import torch
import pprint
import pytest

def test_dataloader():
    from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
    from spflows.data.datamodules import ForecastingDataModule

    config = ForecastingModelConfig()
    datamodule = ForecastingDataModule(config)
    config = datamodule.update_config(config)

    # Test the get_train_databatch function
    databatch = datamodule.get_train_databatch()
    # Assertions for sanity checks
    assert databatch is not None, "Data batch should not be None"
    assert len(databatch) > 0, "Data batch should contain elements"


if __name__=="__main__":
    test_dataloader()