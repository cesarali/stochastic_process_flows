import pytest
from spflows.configs_classes.gecko_configs import GeckoModelConfig
from spflows.data.datamodules import GeckoDatamodule
from spflows.data.gecko.gecko_datasets import CoinGeckoDataset
from spflows.data.gecko.gecko_metadata import AllCoinsMetadata,CoinMetadata
from spflows.data.gecko.gecko_utils import get_dataframe_with_freq_bitcoin,get_dataframe_with_freq_from_bitcoin
from spflows.data.gecko.gecko_requests import (
    get_key
)
from spflows.training.gecko_experiments import GeckoLightningExperiment

def test_allcoins_load():
    coingecko_key = get_key()
    date_string="2024-12-18"
    all_coins_metadata = AllCoinsMetadata(date_string=date_string,coingecko_key=coingecko_key)
    all_coins_metadata.load_existing_timeseries()
    print(all_coins_metadata.df_time_series["bitcoin"].head())

def test_gecko_dataset():
    coingecko_key = get_key()
    date_string="2024-12-18"
    all_coins_metadata = AllCoinsMetadata(date_string=date_string,coingecko_key=coingecko_key)
    all_coins_metadata.load_existing_timeseries()
    coins_not_bitcoin = [coin for coin in all_coins_metadata.df_time_series.keys() if coin != "bitcoin"]
    df = all_coins_metadata.df_time_series["solana"]
    df_bitcoin = all_coins_metadata.df_time_series["bitcoin"]

    df_bitcoin_freq = get_dataframe_with_freq_bitcoin(df_bitcoin)
    df_freq = get_dataframe_with_freq_from_bitcoin(df,df_bitcoin_freq)

    dataset = CoinGeckoDataset(coin_id=coins_not_bitcoin[0],df_freq=df_freq,df_bitcoin_freq=df_bitcoin_freq)
    print(dataset.metadata)
    assert dataset.metadata.prediction_length == 96

def test_frequency_alignement():
    config = GeckoModelConfig()
    config,all_datasets = GeckoDatamodule.get_data_and_update_config(config)
    assert all_datasets is not None

def test_gecko_dataloader():
    config = GeckoModelConfig()
    config,all_datasets = GeckoDatamodule.get_data_and_update_config(config)
    datamodule = GeckoDatamodule(config,all_datasets)
    datamodule.setup()
    databatch = datamodule.get_train_databatch()
    print(databatch.keys())

def test_forward():
    from spflows.models.forecasting.score_lightning import ScoreModule
    config = GeckoModelConfig()
    config,all_datasets = GeckoDatamodule.get_data_and_update_config(config)
    datamodule = GeckoDatamodule(config,all_datasets)
    datamodule.setup()
    databatch = datamodule.get_train_databatch()
    module = ScoreModule(config)
    inputs = list(databatch.values())
    loss = module.train_dynamical_module(*inputs)
    print(loss)

def test_load_experiment():
    import torch
    experiment_dir = r"C:\Users\cesar\Desktop\Projects\BirthDeathPortfolio\stochastic_process_flows\results\1734628682"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment = GeckoLightningExperiment(experiment_dir=experiment_dir,map_location=device)
    databatch = experiment.datamodule.get_train_databatch()
    databatch = experiment.datamodule.send_tensors_to_device(databatch,device)
    inputs = list(databatch.values())
    loss = experiment.model.train_dynamical_module(*inputs)
    print(loss)

if __name__ == "__main__":
    test_forward()
