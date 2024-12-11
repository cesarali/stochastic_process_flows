import torch
import pprint
import pytest

def test_dataloader():
    from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
    from spflows.data.datamodules import ForecastingDataModule

    config = ForecastingModelConfig()
    datamodule = ForecastingDataModule(config)

    for i,data_entry in enumerate(datamodule.validation_iter_dataset):
        if i % 5000 == 0:
            print(data_entry.keys())
            break
    
    for i,data_entry in enumerate(datamodule.validation_iter_dataset):
        if i % 5000 == 0:
            print(data_entry.keys())
            break
        

if __name__=="__main__":
    test_dataloader()