from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
import lightning.pytorch as pl
from typing import List,Any,Tuple,Optional
import numpy as np

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
from  gluonts.dataset.repository.datasets import TrainDatasets
from pts.dataset.loader import TransformedIterableDataset
from gluonts.dataset.common import  Dataset as GluonDataset
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
from spflows.models.forecasting.score_lightning import ScoreModule
from spflows.data.gecko.gecko_datasets import CoinGeckoDataset
from spflows.configs_classes.gecko_configs import GeckoModelConfig
from spflows.data.gecko.gecko_metadata import AllCoinsMetadata
from spflows.data.gecko.gecko_utils import get_dataframe_with_freq_bitcoin,get_dataframe_with_freq_from_bitcoin
from spflows.data.gecko.gecko_requests import (
    get_key
)

class ForecastingDataModule(pl.LightningDataModule):
    """Datamodule to train an stochastic process diffusion module"""
    #training_iter_dataset:TransformedIterableDataset
    #validation_iter_dataset:TransformedIterableDataset
    #training_data:GluonDataset
    #test_data:GluonDataset
    #validation_data:GluonDataset
    def __init__(
            self,
            config:ForecastingModelConfig,
            all_datasets:List[Any] = None,
        ):
        super(ForecastingDataModule,self).__init__()
        self.config = config

        if all_datasets is None:
            config, all_datasets = self.__class__.get_data_and_update_config(config)

        self.training_data = all_datasets[0]
        self.test_data = all_datasets[1]
        self.validation_data = all_datasets[2]

    @staticmethod
    def get_data_and_update_config(config: ForecastingModelConfig)->Tuple[ForecastingModelConfig,List[GluonDataset]]:
        """
        this is all the config information that is obtained from 
        the data metadata as well as the config, that is requiered by the 
        model initialization, but should be called independently of the "setup"
        and "prepare_data" functions which are handled by lightning datamodules
        inside trainer calls
        """
        dataset = get_dataset(config.dataset_str_name, regenerate=False)

        config.covariance_dim = 4 if config.dataset_str_name != 'exchange_rate_nips' else -4
        config.prediction_length = dataset.metadata.prediction_length
        config.target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
        config.input_size = config.target_dim * 4 + config.covariance_dim
        config.context_length = config.context_length if config.context_length is not None else config.prediction_length
        config.freq=dataset.metadata.freq

        config.lags_seq = (
            config.lags_seq
            if config.lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=config.freq)
        )
        config.time_features = (
            config.time_features
            if config.time_features is not None
            else fourier_time_features_from_frequency(config.freq)
        )

        # If context is not provided in the config, the prediction length is used as context
        config.history_length = config.context_length + max(config.lags_seq)

        # Assuming all_datasets is computed elsewhere
        all_datasets = ForecastingDataModule.grouper_datasets(dataset, config)
        return config, all_datasets

    @staticmethod
    def grouper_datasets(dataset:TrainDatasets,config:ForecastingModelConfig)->List[GluonDataset]:
        """Download and validate data."""
        # Set the grouped data attributes for later use
        train_grouper = MultivariateGrouper(max_target_dim=min(2000, config.target_dim))
        test_grouper = MultivariateGrouper(
            num_test_dates=int(len(dataset.test) / len(dataset.train)),
            max_target_dim=min(2000, config.target_dim),
        )
        training_data = train_grouper(dataset.train)
        test_data = test_grouper(dataset.test)

        # Prepare validation data
        val_window = 20 * dataset.metadata.prediction_length
        validation_data = [
            {**deepcopy(item), "target": item["target"][:, -val_window:]} for item in training_data
        ]
        for item in training_data:
            item["target"] = item["target"][:, :-val_window]
        return [training_data,test_data,validation_data]

    def setup_from_config(self):
        """function get_data_and_update_config was called before"""
        config:ForecastingModelConfig = self.config
        self.prediction_length = config.prediction_length
        self.context_length = config.context_length if config.context_length is not None else self.prediction_length
        self.history_length = self.context_length + max(config.lags_seq)
        self.target_dim = config.target_dim
        self.covariance_dim = 4 if config.dataset_str_name != 'exchange_rate_nips' else -4
        self.input_dim = self.target_dim * 4 + self.covariance_dim

        self.lags_seq = config.lags_seq
        self.time_features = config.time_features

        self.pick_incomplete = config.pick_incomplete
        self.scaling = config.scaling

        self.train_input_names,_ = ScoreModule.get_networks_inputs(self.config)
        self.dataset_str_name = self.config.dataset_str_name
        self.batch_size = config.batch_size

    def setup_samplers(self):
        """
        Here we define how we are going to select the different prediction seq2seq elements
        """
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

    def setup(self,stage: Optional[str] = None):
        """set features, transformations, get data, set datasets"""
        self.setup_from_config()
        self.setup_samplers()
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
            persistent_workers=True,
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
            persistent_workers=True,
            worker_init_fn=self._worker_init_fn,
        )
        return validation_data_loader

    def test_dataloader(self):
        pass

    def get_train_databatch(self):
        return next(self.train_dataloader().__iter__())

    def send_tensors_to_device(self,databatch, device):
        """
        Sends all tensor values in a dictionary or named tuple to the specified device.

        Args:
            data (dict or namedtuple): The input data containing tensors.
            device (torch.device): The target device to send the tensors to.

        Returns:
            dict: A new dictionary with tensors moved to the specified device.
        """
        if isinstance(databatch, dict):
            return {key: value.to(device) for key, value in databatch.items()}
        elif hasattr(databatch, '_fields'):  # Check if it's a namedtuple
            return type(databatch)(*(value.to(device) if hasattr(value, 'to') else value for value in databatch))
        else:
            raise TypeError("Input data must be a dictionary or namedtuple.")

class GeckoDatamodule(ForecastingDataModule):
    """Datamodule to train an stochastic process diffusion module"""
    #training_iter_dataset:TransformedIterableDataset
    #validation_iter_dataset:TransformedIterableDataset
    #training_data:GluonDataset
    #test_data:GluonDataset
    #validation_data:GluonDataset
    def __init__(
            self,
            config:GeckoModelConfig,
            all_datasets:List[Any] = None,
        ):
        super(GeckoDatamodule,self).__init__(config,all_datasets)

    @staticmethod
    def get_coingecko_datasets(config:GeckoModelConfig,regenerate=False)->List[CoinGeckoDataset]:
        coingecko_key = get_key()
        all_coins_metadata = AllCoinsMetadata(date_string=config.date_str,
                                              coingecko_key=coingecko_key)
        all_coins_metadata.load_existing_timeseries()
        df_bitcoin = all_coins_metadata.df_time_series["bitcoin"]
        df_bitcoin_freq = get_dataframe_with_freq_bitcoin(df_bitcoin)

        all_datasets = []
        not_bitcoin_coins = [coin for coin in all_coins_metadata.df_time_series.keys() if coin != "bitcoin"]
        print("Setting Data Frequencies")
        for not_bitcoin_coin in tqdm(not_bitcoin_coins):
            df = all_coins_metadata.df_time_series[not_bitcoin_coin]
            df_freq = get_dataframe_with_freq_from_bitcoin(df,df_bitcoin_freq)
            if df_freq is not None:
                dataset = CoinGeckoDataset(coin_id=not_bitcoin_coin,df_freq=df_freq,df_bitcoin_freq=df_bitcoin_freq)
                all_datasets.append(dataset)
            else:
                pass
        return all_datasets

    @staticmethod
    def get_data_and_update_config(config: GeckoModelConfig)->Tuple[GeckoModelConfig,List[GluonDataset]]:
        """
        this is all the config information that is obtained from
        the data metadata as well as the config, that is requiered by the
        model initialization, but should be called independently of the "setup"
        and "prepare_data" functions which are handled by lightning datamodules
        inside trainer calls
        """
        datasets = GeckoDatamodule.get_coingecko_datasets(config, regenerate=False)
        dataset = datasets[0]
        config.covariance_dim = 4 if config.date_str != 'exchange_rate_nips' else -4
        config.prediction_length = dataset.metadata.prediction_length
        config.target_dim = int(dataset.metadata.feat_static_cat[0].cardinality)
        config.input_size = config.target_dim * 4 + config.covariance_dim
        config.context_length = config.context_length if config.context_length is not None else config.prediction_length
        config.freq=dataset.metadata.freq

        config.lags_seq = (
            config.lags_seq
            if config.lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=config.freq)
        )
        config.time_features = (
            config.time_features
            if config.time_features is not None
            else fourier_time_features_from_frequency(config.freq)
        )

        # If context is not provided in the config, the prediction length is used as context
        config.history_length = config.context_length + max(config.lags_seq)

        # Assuming all_datasets is computed elsewhere
        all_datasets = GeckoDatamodule.grouper_all_datasets(datasets, config)
        return config, all_datasets

    @staticmethod
    def grouper_all_datasets(datasets: List[TrainDatasets], config: ForecastingModelConfig) -> List[List[GluonDataset]]:
        """
        Download and validate data for a list of datasets.

        Args:
        ----
        datasets: List[TrainDatasets]
            A list of datasets containing training and test data.
        config: ForecastingModelConfig
            Configuration object with target dimensions and other settings.

        Returns:
        -------
        grouped_datasets: List[List[GluonDataset]]
            A list containing grouped training, test, and validation datasets for all input datasets.
        """
        all_training_data = []
        all_test_data = []
        all_validation_data = []

        for dataset in datasets:
            # Initialize groupers for each dataset
            train_grouper = MultivariateGrouper(max_target_dim=min(2000, config.target_dim))
            test_grouper = MultivariateGrouper(
                num_test_dates=int(len(dataset.test) / len(dataset.train)),
                max_target_dim=min(2000, config.target_dim),
            )

            # Apply the groupers
            training_data = train_grouper(dataset.train)
            test_data = test_grouper(dataset.test)

            # Prepare validation data
            val_window = 20 * dataset.metadata.prediction_length
            validation_data = [
                {**deepcopy(item), "target": item["target"][:, -val_window:]} for item in training_data
            ]
            for item in training_data:
                item["target"] = item["target"][:, :-val_window]

            # Collect grouped data
            all_training_data.append(training_data[0])
            all_test_data.append(test_data[0])
            all_validation_data.append(validation_data[0])

        return [all_training_data, all_test_data, all_validation_data]


