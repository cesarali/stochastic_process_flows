import os
import yaml
from typing import List
from dataclasses import asdict
from spflows import results_path
from lightning.pytorch import Trainer

from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from spflows.data.datamodules import GeckoDatamodule
from spflows.utils.experiment_files import ExperimentsFiles
from spflows.models.forecasting.score_lightning import ScoreModule
from spflows.configs_classes.gecko_configs import GeckoModelConfig
from spflows.training.basic_experiments import BasicLightningExperiment

class GeckoLightningExperiment(BasicLightningExperiment):
    """
    Defines all objects needed to train, check_pointings and evaluation as well as calls to the
    torch trainer
    """
    experiment_name:str = ""
    experiment_files:ExperimentsFiles
    datamodule:GeckoDatamodule
    model:ScoreModule

    set_to_train:bool = False
    logger:MLFlowLogger = None
    callbacks:List[ModelCheckpoint] = None

    def __init__(self,config:GeckoModelConfig=None,experiment_dir:str=None,map_location:str="cuda"):
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
        self.experiment_files = ExperimentsFiles(experiment_dir=experiment_dir,results_path=self.config.results_path)
        self.config = GeckoModelConfig.from_yaml(self.experiment_files.params_yaml)
        checkpoint_path = self.experiment_files.get_lightning_checkpoint_path("best")
        self.config, all_datasets = GeckoDatamodule.get_data_and_update_config(self.config)
        self.datamodule = GeckoDatamodule(self.config,all_datasets)
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
        self.config, all_datasets = GeckoDatamodule.get_data_and_update_config(self.config)
        self.datamodule = GeckoDatamodule(self.config,all_datasets)

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
            limit_val_batches=self.config.num_batches_per_epoch_val,
            log_every_n_steps=1,
            gradient_clip_val=self.config.clip_gradient
        )
        trainer.fit(self.model,datamodule=self.datamodule)

    def save_test_samples(self):
        checkpoint_path = self.experiment_files.get_lightning_checkpoint_path("best")
        self.model = ScoreModule.load_from_checkpoint(checkpoint_path, model_params=self.config, map_location="cuda")

    def save_hyperparameters_to_yaml(self,hyperparams: GeckoModelConfig, file_path: str):
        time_features_ = hyperparams.time_features
        hyperparams.time_features = None
        with open(file_path, 'w') as file:
            yaml.dump(asdict(hyperparams), file)
        hyperparams.time_features = time_features_
