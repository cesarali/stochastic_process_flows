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
from spflows.training.basic_experiments import BasicLightningExperiment

class ForecastingLightningExperiment(BasicLightningExperiment):
    """
    Defines all objects needed to train, check_pointings and evaluation as well as calls to the
    torch trainer
    """
    experiment_name:str = ""
    experiment_files:ExperimentsFiles
    datamodule:ForecastingDataModule
    model:ScoreModule

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
        super().__init__(config,experiment_dir,map_location)

    def setup_experiment_from_dir(self,experiment_dir):
        """"""
        self.experiment_files = ExperimentsFiles(experiment_dir=experiment_dir,results_path=self.config.results_path)
        self.config = ForecastingModelConfig.from_yaml(self.experiment_files.params_yaml)
        checkpoint_path = self.experiment_files.get_lightning_checkpoint_path("best")
        self.config, all_datasets = ForecastingDataModule.get_data_and_update_config(self.config)
        self.datamodule = ForecastingDataModule(self.config,all_datasets)
        self.model = ScoreModule.load_from_checkpoint(checkpoint_path, config=self.config, map_location=self.map_location)

    def setup_datamodule(self):
        self.config, all_datasets = ForecastingDataModule.get_data_and_update_config(self.config)
        self.datamodule = ForecastingDataModule(self.config,all_datasets)

    def setup_model(self):
        self.model = ScoreModule(self.config)

    def save_test_samples(self):
        checkpoint_path = self.experiment_files.get_lightning_checkpoint_path("best")
        self.model = ScoreModule.load_from_checkpoint(checkpoint_path, model_params=self.config, map_location="cuda")

    def save_hyperparameters_to_yaml(self,hyperparams: ForecastingModelConfig, file_path: str):
        time_features_ = hyperparams.time_features
        hyperparams.time_features = None
        with open(file_path, 'w') as file:
            yaml.dump(asdict(hyperparams), file)
        hyperparams.time_features = time_features_
