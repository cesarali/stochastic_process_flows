import os
import json
import yaml
from dataclasses import asdict
import numpy as np
from spflows import results_path
from lightning.pytorch import Trainer
import lightning.pytorch as pl
from typing import List

from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from spflows.utils.experiment_files import ExperimentsFiles
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
from spflows.data.datamodules import ForecastingDataModule

from spflows.models.forecasting.score_lightning import ScoreModule
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator



def energy_score(forecast, target):
    obs_dist = np.mean(np.linalg.norm((forecast - target), axis=-1))
    pair_dist = np.mean(
        np.linalg.norm(forecast[:, np.newaxis, ...] - forecast, axis=-1)
    )
    return obs_dist - pair_dist * 0.5

class BasicLightningExperiment:
    """ 
    Defines all objects needed to train, check_pointings and evaluation as well as calls to the 
    torch trainer
    """
    experiment_name:str = ""
    experiment_files:ExperimentsFiles
    datamodule:pl.LightningDataModule
    model:pl.LightningModule

    set_to_train:bool = False
    logger:MLFlowLogger = None
    callbacks:List[ModelCheckpoint] = None

    def __init__(self,config:ForecastingModelConfig=None,experiment_dir:str=None,map_location:str="cuda"):
        """
        If experiment dir is provided just load the models
        and all correspondings specificatons else,
        creates a new folder for experiments to be saved there
        if training called
        """
        self.map_location = map_location

        if experiment_dir is not None:
            self.set_to_train = False
            self.setup_experiment_from_dir(experiment_dir)
        else:
            self.set_to_train = True
            self.config = config
            self.experiment_name = config.experiment_name

            self.setup_experiment_files()
            self.setup_logger()
            self.setup_callbacks()
            self.setup_datamodule()
            self.setup_model()

    def setup_experiment_from_dir(self,experiment_dir):
        """"""
        self.experiment_files = ExperimentsFiles(experiment_dir=experiment_dir)
        self.config = ForecastingModelConfig.from_yaml(self.experiment_files.params_yaml)
        checkpoint_path = self.experiment_files.get_lightning_checkpoint_path("best")
        self.config, all_datasets = ForecastingDataModule.get_data_and_update_config(self.config)
        self.datamodule = ForecastingDataModule(self.config,all_datasets)
        self.model = ScoreModule.load_from_checkpoint(checkpoint_path, config=self.config, map_location=self.map_location)        

    def setup_experiment_files(self):
        self.experiment_files = ExperimentsFiles(experiment_indentifier=self.config.experiment_indentifier, delete=True)
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
        self.callbacks = [self.checkpoint_callback_last,self.checkpoint_callback_best]
    
    def setup_datamodule(self):
        self.config, all_datasets = ForecastingDataModule.get_data_and_update_config(self.config)
        self.datamodule = ForecastingDataModule(self.config,all_datasets)

    def setup_model(self):
        self.model = ScoreModule(self.config)

    def train(self):
        self.save_hyperparameters_to_yaml(self.config, self.experiment_files.params_yaml)
        trainer = Trainer(
            default_root_dir=self.experiment_files.experiment_dir,
            logger=self.logger,
            max_epochs=self.config.epochs,
            callbacks=self.callbacks,
            limit_train_batches=self.config.num_batches_per_epoch,
            limit_val_batches=20,
            log_every_n_steps=1,
        )
        trainer.fit(self.model,datamodule=self.datamodule)
        # store metrics save samples
        self.evaluate()

    def save_test_samples(self):
        checkpoint_path = self.experiment_files.get_lightning_checkpoint_path("best")
        self.model = ScoreModule.load_from_checkpoint(checkpoint_path, model_params=self.config, map_location="cuda")
    
    def metrics_evaluations(self,forecasts,targets):
        score = energy_score(
            forecast=np.array([x.samples for x in forecasts]),
            target=np.array([x[-self.datamodule.prediction_length:] for x in targets])[:,None,...],
        )

        evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:], target_agg_funcs={'sum': np.sum})
        agg_metric, _ = evaluator(targets, forecasts, num_series=len(self.datamodule.test_data))

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
    
    def define_predictor_and_sample(self):
        predictor = self.model.create_predictor_network(
            transformation=self.datamodule.transformations,
            prediction_splitter=self.datamodule.create_instance_splitter("test"),
        )
        forecast_it, ts_it = make_evaluation_predictions(dataset=self.datamodule.test_data, 
                                                        predictor=predictor, 
                                                        num_samples=10)
        forecasts = list(forecast_it)
        targets = list(ts_it)
        return forecasts,targets
    
    def evaluate(self):
        # Evaluation
        forecasts,targets = self.define_predictor_and_sample()
        metrics = self.metrics_evaluations(forecasts,targets)
        metrics_file = open(self.experiment_files.metrics_path.format("gluonts_evaluator"),"w")
        json.dump(metrics,metrics_file,indent=4)        
        return metrics
    
    def save_hyperparameters_to_yaml(self,hyperparams: ForecastingModelConfig, file_path: str):
        time_features_ = hyperparams.time_features
        hyperparams.time_features = None
        with open(file_path, 'w') as file:
            yaml.dump(asdict(hyperparams), file)
        hyperparams.time_features = time_features_