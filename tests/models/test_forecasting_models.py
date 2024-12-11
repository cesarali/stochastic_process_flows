import torch
import pprint
import pytest

def test_forward():
    from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
    from spflows.data.datamodules import ForecastingDataModule
    from spflows.models.forecasting.score_lightning import ScoreModule

    config = ForecastingModelConfig()
    datamodule = ForecastingDataModule(config)
    config = datamodule.update_config(config)
    databatch = datamodule.get_train_databatch()
    module = ScoreModule(config)
    inputs = [v for v in databatch.values()]
    loss = module.train_dynamical_module(*inputs)

    print(loss)



if __name__=="__main__":
    test_forward()