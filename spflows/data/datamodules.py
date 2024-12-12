from torch.utils.data import DataLoader
from copy import deepcopy
import pytorch_lightning as pl

import numpy as np
from typing import List,Dict

from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    TestSplitSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import get_dataset

from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)

from gluonts.env import env
from gluonts.itertools import maybe_len
from gluonts.transform import SelectFields
from pts.dataset.loader import TransformedIterableDataset
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.common import  Dataset as GluonDataset
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig

from typing import NamedTuple
from torch import nn
from spflows.models.forecasting.score_lightning import ScoreModule

class TrainOutput(NamedTuple):
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor

class ForecastingDataModule(pl.LightningDataModule):
    """Datamodule to train an stochastic process diffusion module"""

    training_iter_dataset:TransformedIterableDataset
    validation_iter_dataset:TransformedIterableDataset
    test_data:GluonDataset

    def __init__(
            self,
            config:ForecastingModelConfig
        ):
        super().__init__()
        self.config = config

        self.train_input_names,_ = ScoreModule.get_networks_inputs(self.config)
        self.dataset = self.config.dataset
        self.batch_size = config.batch_size
        
        # Do not call setup here; let Lightning manage it.
        self.target_dim = None
        self.covariance_dim = None
        self.prediction_length = None
        self.input_size = None
        self.training_data = None
        self.validation_data = None

        self.prepare_data()
        self.setup()
        
    def update_config(self,config:ForecastingModelConfig):
        config.prediction_length = self.prediction_length
        config.context_length = self.context_length
        config.lags_seq = self.lags_seq
        config.time_features = self.time_features
        config.history_length = self.history_length
        config.target_dim = self.target_dim
        config.covariance_dim = self.covariance_dim
        config.input_size = self.input_size
        return config
        
    def prepare_data(self):
        """Download and validate data."""
        # Load and validate data
        dataset = get_dataset(self.dataset, regenerate=False)

        # Store metadata and attributes for later use
        self.covariance_dim = 4 if self.dataset != 'exchange_rate_nips' else -4
        self.prediction_length = dataset.metadata.prediction_length
        self.config.prediction_length = dataset.metadata.prediction_length
        self.target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
        self.config.target_dim = self.target_dim
        self.input_size = self.target_dim * 4 + self.covariance_dim

        # Set the grouped data attributes for later use
        train_grouper = MultivariateGrouper(max_target_dim=min(2000, self.target_dim))
        test_grouper = MultivariateGrouper(
            num_test_dates=int(len(dataset.test) / len(dataset.train)),
            max_target_dim=min(2000, self.target_dim),
        )
        self.training_data = train_grouper(dataset.train)
        self.test_data = test_grouper(dataset.test)

        # Prepare validation data
        val_window = 20 * dataset.metadata.prediction_length
        self.validation_data = [
            {**deepcopy(item), "target": item["target"][:, -val_window:]} for item in self.training_data
        ]
        for item in self.training_data:
            item["target"] = item["target"][:, :-val_window]

    def setup_time_features(self):
        """
        Here we define how we are going to select the different prediction seq2seq elements
        """
        self.lags_seq = (
            self.config.lags_seq
            if self.config.lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=self.config.freq)
        )

        self.time_features = (
            self.config.time_features
            if self.config.time_features is not None
            else fourier_time_features_from_frequency(self.config.freq)
        )

        self.prediction_length = self.config.prediction_length
        self.context_length = self.config.context_length if self.config.context_length is not None else self.prediction_length
        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = self.config.pick_incomplete
        self.input_dim =  self.target_dim * 4 + self.covariance_dim
        self.scaling = self.config.scaling

        # the random selection of points along the path
        # to take the seq2seq segments
        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=10.0,
            min_past=0 if self.config.pick_incomplete else self.history_length,
            min_future=self.prediction_length,
        )

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if self.config.pick_incomplete else self.history_length,
            min_future=self.prediction_length,
        )
    
    def setup(self):
        """set features, transformations, get data, set datasets"""
        self.setup_time_features()
        self.setup_transformations()

        training_lenght = maybe_len(self.training_data)
        with env._let(max_idle_transforms=training_lenght or 0):
            training_instance_splitter = self.create_instance_splitter("training")
        training_transforms = self.transformations + training_instance_splitter + SelectFields(self.train_input_names)

        self.training_iter_dataset = TransformedIterableDataset(
            dataset=self.training_data,
            transform=training_transforms,
            is_train=True,
            shuffle_buffer_length=self.config.shuffle_buffer_length,
            cache_data=self.config.cache_data,
        )

        if self.validation_data is not None:
            validation_lenght = maybe_len(self.validation_data)
            with env._let(max_idle_transforms=validation_lenght or 0):
                validation_instance_splitter = self.create_instance_splitter("validation")
            val_transforms = self.transformations + validation_instance_splitter + SelectFields(self.train_input_names)

            self.validation_iter_dataset = TransformedIterableDataset(
                dataset=self.validation_data,
                transform=val_transforms,
                is_train=True,
                cache_data=self.config.cache_data,
            )
        
    def setup_transformations(self) -> Transformation:
        self.transformations =  Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                # maps the target to (1, T)
                # if the target data is uni dimensional
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.prediction_length,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME],
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )
    
    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.FEAT_TIME,
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )
    
    @staticmethod
    def _worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def train_dataloader(self):
        training_data_loader = DataLoader(
            self.training_iter_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn
        )
        return training_data_loader

    def val_dataloader(self):
        validation_data_loader = DataLoader(
            self.validation_iter_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
        )
        return validation_data_loader

    def test_dataloader(self):
        pass
    
    def get_train_databatch(self):
        return next(self.train_dataloader().__iter__())
    
"""
def get_data(self)->List[Dict[str,np.array]]:
    # Load data
    dataset = get_dataset(self.dataset, regenerate=False)

    # update and set values from dataset
    self.covariance_dim = 4 if self.dataset != 'exchange_rate_nips' else -4
    self.prediction_length = dataset.metadata.prediction_length
    self.config.prediction_length = dataset.metadata.prediction_length
    self.target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
    self.config.target_dim = self.target_dim
    self.input_size = self.target_dim * 4 + self.covariance_dim

    train_grouper = MultivariateGrouper(max_target_dim=min(2000, self.target_dim))
    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test) / len(dataset.train)), max_target_dim=min(2000, self.target_dim))

    training_data = train_grouper(dataset.train)
    self.test_data = test_grouper(dataset.test)

    val_window = 20 * dataset.metadata.prediction_length
    training_data = list(training_data)
    validation_data = []
    for i in range(len(training_data)):
        x = deepcopy(training_data[i])
        x['target'] = x['target'][:,-val_window:]
        validation_data.append(x)
        training_data[i]['target'] = training_data[i]['target'][:,:-val_window]    
    return training_data,self.test_data,validation_data
"""