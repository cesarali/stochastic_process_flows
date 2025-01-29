import os
import yaml
from typing import List
from dataclasses import asdict
from spflows import results_path
from lightning.pytorch import Trainer

from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from spflows.data.datamodules import ForecastingDataModule
from spflows.utils.experiment_files import ExperimentsFiles
from spflows.models.forecasting.score_lightning import ScoreModule
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
from abc import ABC, abstractmethod


class BasicLightningExperiment(ABC):
    """
    Abstract class defining all objects needed for training, checkpointing,
    and evaluation, as well as calls to the PyTorch Lightning trainer.
    """
    experiment_name: str = ""
    experiment_files: ExperimentsFiles
    datamodule: ForecastingDataModule
    model: ScoreModule
    set_to_train: bool = False
    logger: MLFlowLogger = None
    callbacks: List[ModelCheckpoint] = None

    def __init__(self, config: ForecastingModelConfig = None, experiment_dir: str = None, map_location: str = "cuda"):
        """
        Initializes the experiment.
        If `experiment_dir` is provided, sets up from a saved directory; otherwise, creates a new experiment.
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

    def setup_experiment_files(self):
        """
        Sets up experiment files for storing logs, checkpoints, and configurations.
        """
        self.experiment_files = ExperimentsFiles(experiment_indentifier=self.config.experiment_indentifier, delete=True)
        self.config.experiment_dir = self.experiment_files.experiment_dir
        self.config.experiment_name = self.experiment_name

    def setup_logger(self):
        """
        Sets up the logger for MLFlow.
        """
        ml_flow_folder = os.path.join(results_path, "mlruns")
        self.logger = MLFlowLogger(experiment_name=self.experiment_name,tracking_uri=f"file:{ml_flow_folder}")

    def setup_callbacks(self):
        """
        Sets up model checkpoint callbacks for saving the best and last model states.
        """
        self.checkpoint_callback_best = ModelCheckpoint(dirpath=self.experiment_files.checkpoints_dir,
                                                        save_top_k=1,
                                                        monitor="val_loss",
                                                        filename="best-{epoch:02d}")
        self.checkpoint_callback_last = ModelCheckpoint(dirpath=self.experiment_files.checkpoints_dir,
                                                        save_top_k=1,
                                                        monitor=None,
                                                        filename="last-{epoch:02d}")
        self.callbacks = [self.checkpoint_callback_last, self.checkpoint_callback_best]

    def train(self):
        """
        Trains the model using PyTorch Lightning's `Trainer`.
        """
        self.save_hyperparameters_to_yaml(self.config, self.experiment_files.params_yaml)
        trainer = Trainer(
            default_root_dir=self.experiment_files.experiment_dir,
            logger=self.logger,
            max_epochs=self.config.epochs,
            callbacks=self.callbacks,
            limit_train_batches=self.config.num_batches_per_epoch,
            limit_val_batches=self.config.num_batches_per_epoch_val,
            log_every_n_steps=1,
            gradient_clip_val=self.config.clip_gradient
        )
        trainer.fit(self.model, datamodule=self.datamodule)

    @abstractmethod
    def setup_experiment_from_dir(self, experiment_dir):
        """
        Abstract method to set up an experiment from a saved directory.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def setup_datamodule(self):
        """
        Abstract method to set up the data module.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def setup_model(self):
        """
        Abstract method to set up the model.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def save_test_samples(self):
        """
        Abstract method to save test samples.
        Must be implemented in subclasses.
        """
        pass

    def save_hyperparameters_to_yaml(self, hyperparams: ForecastingModelConfig, file_path: str):
        """
        Saves hyperparameters to a YAML file.
        """
        with open(file_path, "w") as yaml_file:
            yaml.dump(asdict(hyperparams), yaml_file)
