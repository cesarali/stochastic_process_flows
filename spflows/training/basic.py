import os
from spflows import results_path
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from spflows.utils.experiment_files import ExperimentsFiles
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
from spflows.data.datamodules import ForecastingDataModule
from spflows.models.forecasting.score_lightning import ScoreModule

class LightningTrainer:
    """ Defines all objects needed to train, check pointing and so on"""

    def __init__(self, config:ForecastingModelConfig):
        self.config = config
        self.experiment_name = "forecasting"

        self.setup_experiment_files()
        self.setup_logger()
        self.setup_callbacks()
        self.setup_dataloaders()
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

    def setup_dataloaders(self):
        self.dataloaders = ForecastingDataModule(self.config)

    def setup_model(self):
        self.model = ScoreModule(self.config)
        #save_hyperparameters_to_yaml(self.config, self.experiment_files.params_yaml)

    def train(self):
        trainer = Trainer(
            default_root_dir=self.experiment_files.experiment_dir,
            logger=self.logger,
            max_epochs=self.config.num_epochs,
            callbacks=[self.checkpoint_callback_best, self.checkpoint_callback_last]
        )
        trainer.fit(self.model, self.dataloaders.train_it, self.dataloaders.validation_it)
        self.save_test_samples()

    def save_test_samples(self):
        checkpoint_path = self.experiment_files.get_lightning_checkpoint_path("best")
        self.model = ScoreModule.load_from_checkpoint(checkpoint_path, model_params=self.config, map_location="cuda")
        #sample_and_save_from_test(self.model, self.dataloaders, self.experiment_files)