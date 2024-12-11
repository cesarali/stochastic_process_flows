import torch
import pprint
import pytest

def test_dataloader():
    from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
    from spflows.data.datamodules import ForecastingDataModule

    config = ForecastingModelConfig()
    datamodule = ForecastingDataModule(config)
    assert datamodule is not None

if __name__=="__main__":
    test_dataloader()