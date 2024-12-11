import os
import yaml
from dataclasses import asdict
import numpy as np
from spflows import results_path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from spflows.utils.experiment_files import ExperimentsFiles
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
from spflows.data.datamodules import ForecastingDataModule

from spflows.models.forecasting.score_lightning import ScoreModule
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

def save_hyperparameters_to_yaml(hyperparams: ForecastingModelConfig, file_path: str):
    hyperparams.time_features = None
    with open(file_path, 'w') as file:
        yaml.dump(asdict(hyperparams), file)

def energy_score(forecast, target):
    obs_dist = np.mean(np.linalg.norm((forecast - target), axis=-1))
    pair_dist = np.mean(
        np.linalg.norm(forecast[:, np.newaxis, ...] - forecast, axis=-1)
    )
    return obs_dist - pair_dist * 0.5

class LightningTrainer:
    """ Defines all objects needed to train, check_pointings and evaluation"""

    def __init__(self, config:ForecastingModelConfig):
        self.config = config
        self.experiment_name = "forecasting"

        self.setup_experiment_files()
        self.setup_logger()
        self.setup_callbacks()
        self.setup_datamodule()
        self.setup_model()

    def setup_experiment_files(self):
        self.experiment_files = ExperimentsFiles(experiment_indentifier=None, delete=True)
        self.config.experiment_dir = self.experiment_files.experiment_dir
        self.config.experiment_name = self.experiment_name

    def setup_logger(self):
        ml_flow_folder = os.path.join(results_path, "mlruns")
        self.logger = MLFlowLogger(experiment_name=self.experiment_name,
                                   tracking_uri=f"file:{ml_flow_folder}")

    def setup_callbacks(self):
        self.checkpoint_callback_best = ModelCheckpoint(dirpath=self.experiment_files.checkpoints_dir,
                                                        save_top_k=1,
                                                        monitor="val_loss",
                                                        filename="best-{epoch:02d}")
        self.checkpoint_callback_last = ModelCheckpoint(dirpath=self.experiment_files.checkpoints_dir,
                                                        save_top_k=1,
                                                        monitor=None,
                                                        filename="last-{epoch:02d}")

    def setup_datamodule(self):
        self.datamodule = ForecastingDataModule(self.config)
        self.datamodule.prepare_data()
        self.datamodule.setup()
        self.config = self.datamodule.update_config(self.config)

    def setup_model(self):
        self.model = ScoreModule(self.config)

    def train(self):
        save_hyperparameters_to_yaml(self.config, self.experiment_files.params_yaml)
        trainer = Trainer(
            default_root_dir=self.experiment_files.experiment_dir,
            logger=self.logger,
            max_epochs=self.config.epochs,
            callbacks=[self.checkpoint_callback_best, self.checkpoint_callback_last],
            limit_train_batches=self.config.num_batches_per_epoch,
            limit_val_batches=20,
        )
        trainer.fit(self.model, 
                    train_dataloaders=self.datamodule.train_dataloader(),
                    val_dataloaders=self.datamodule.val_dataloader())
        #self.save_test_samples()

    def save_test_samples(self):
        checkpoint_path = self.experiment_files.get_lightning_checkpoint_path("best")
        self.model = ScoreModule.load_from_checkpoint(checkpoint_path, model_params=self.config, map_location="cuda")
        #sample_and_save_from_test(self.model, self.dataloaders, self.experiment_files)
    
    def evaluate(self):
        # Evaluation
        predictor = self.model
        forecast_it, ts_it = make_evaluation_predictions(dataset=self.datamodule.dataset_test, 
                                                         predictor=predictor, num_samples=100)
        forecasts = list(forecast_it)
        targets = list(ts_it)

        score = energy_score(
            forecast=np.array([x.samples for x in forecasts]),
            target=np.array([x[-self.datamodule.prediction_length:] for x in targets])[:,None,...],
        )

        evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], target_agg_funcs={'sum': np.sum})
        agg_metric, _ = evaluator(targets, forecasts, num_series=len(self.datamodule.dataset_test))

        metrics = dict(
            CRPS=agg_metric['mean_wQuantileLoss'],
            ND=agg_metric['ND'],
            NRMSE=agg_metric['NRMSE'],
            CRPS_sum=agg_metric['m_sum_mean_wQuantileLoss'],
            ND_sum=agg_metric['m_sum_ND'],
            NRMSE_sum=agg_metric['m_sum_NRMSE'],
            energy_score=score,
        )
        metrics = { k: float(v) for k,v in metrics.items() }

        return metrics