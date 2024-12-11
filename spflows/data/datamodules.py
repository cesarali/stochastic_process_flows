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
from gluonts.dataset.common import Dataset
from gluonts.transform import SelectFields
from gluonts.itertools import maybe_len

from pts.dataset.loader import TransformedIterableDataset
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig

class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series = self.data[idx]
        context = series[:self.context_length]
        prediction = series[self.context_length:self.context_length + self.prediction_length]
        return {"context": context, "prediction": prediction}

class ForecastingDataModule(pl.LightningDataModule):
    """Datamodule to train an stochastic process diffusion module"""

    training_iter_dataset:TransformedIterableDataset
    validation_iter_dataset:TransformedIterableDataset

    def __init__(
            self,
            config:ForecastingModelConfig
        ):
        super().__init__()
        self.config = config

        # THIS WAS TAKEN FROM THE SIGNATURE OF TimeGradTrainingNetwork_All
        self.input_names = [
            "target_dimension_indicator",
            "past_time_feat",
            "past_target_cdf",
            "past_observed_values",
            "past_is_pad",
            "future_time_feat",
            "future_target_cdf",
            "future_observed_values"
        ]

        self.dataset = self.config.dataset
        self.batch_size = config.batch_size
        self.context_length = config.context_length
        self.prediction_length = config.prediction_length
        self.setup_datasets()

    def get_data(self)->List[Dict[str,np.array]]:
        """downloads data and sets the grouper for multivariates"""
        self.covariance_dim = 4 if self.dataset != 'exchange_rate_nips' else -4
        # Load data
        dataset = get_dataset(self.dataset, regenerate=False)
        self.prediction_length=dataset.metadata.prediction_length
        self.context_length=dataset.metadata.prediction_length

        self.config.prediction_length = self.prediction_length
        self.config.context_length = self.context_length

        target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)

        train_grouper = MultivariateGrouper(max_target_dim=min(2000, target_dim))
        test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test) / len(dataset.train)), max_target_dim=min(2000, target_dim))

        training_data = train_grouper(dataset.train)
        test_data = test_grouper(dataset.test)

        val_window = 20 * dataset.metadata.prediction_length
        training_data = list(training_data)
        validation_data = []
        for i in range(len(training_data)):
            x = deepcopy(training_data[i])
            x['target'] = x['target'][:,-val_window:]
            validation_data.append(x)
            training_data[i]['target'] = training_data[i]['target'][:,:-val_window]    
        return training_data,test_data,validation_data
    
    def setup_time_features(self):
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

        self.history_length = self.context_length + max(self.lags_seq)
        self.pick_incomplete = self.config.pick_incomplete
        self.scaling = self.config.scaling

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if self.config.pick_incomplete else self.history_length,
            min_future=self.config.prediction_length,
        )

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if self.config.pick_incomplete else self.history_length,
            min_future=self.config.prediction_length,
        )
    
    def setup_datasets(self):
        """set features, transformations, get data, set datasets"""
        training_data,test_data,validation_data = self.get_data()
        self.setup_time_features()
        self.setup_transformations()

        with env._let(max_idle_transforms=maybe_len(training_data) or 0):
            validation_instance_splitter = self.create_instance_splitter("validation")
        training_transforms = self.transformations + validation_instance_splitter + SelectFields(self.input_names)

        self.training_iter_dataset = TransformedIterableDataset(
            dataset=training_data,
            transform=training_transforms,
            is_train=True,
            shuffle_buffer_length=self.config.shuffle_buffer_length,
            cache_data=self.config.cache_data,
        )

        if validation_data is not None:
            with env._let(max_idle_transforms=maybe_len(validation_data) or 0):
                validation_instance_splitter = self.create_instance_splitter("validation")
            val_transforms = self.transformations + validation_instance_splitter + SelectFields(self.input_names)

            self.validation_iter_dataset = TransformedIterableDataset(
                dataset=validation_data,
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
    
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass