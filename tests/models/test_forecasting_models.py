import torch
import pprint
import pytest

def test_forward():
    from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
    from spflows.data.datamodules import ForecastingDataModule
    from spflows.models.forecasting.score_lightning import ScoreModule

    config = ForecastingModelConfig(residual_layers=2,residual_channels=2)
    config, all_data = ForecastingDataModule.get_data_and_update_config(config)
    datamodule = ForecastingDataModule(config,all_data)
    datamodule.setup()
    databatch = datamodule.get_train_databatch()
    module = ScoreModule(config)
    inputs = list(databatch.values())
    loss = module.train_dynamical_module(*inputs)
    assert loss is not None

def test_predictor():
    from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
    from spflows.data.datamodules import ForecastingDataModule
    from spflows.models.forecasting.score_lightning import ScoreModule
    from gluonts.evaluation.backtest import make_evaluation_predictions

    config = ForecastingModelConfig(residual_layers=2,residual_channels=2)
    config, all_data = ForecastingDataModule.get_data_and_update_config(config)
    datamodule = ForecastingDataModule(config,all_data)
    datamodule.setup()
    model = ScoreModule(config)

    predictor = model.create_predictor_network(
        transformation=datamodule.transformations,
        prediction_splitter=datamodule.create_instance_splitter("test"),
    )
    forecast_it, ts_it = make_evaluation_predictions(dataset=datamodule.test_data,
                                                     predictor=predictor,
                                                     num_samples=10)
    forecasts = list(forecast_it)
    targets = list(ts_it)
    assert targets is not None
    assert forecasts is not None

if __name__=="__main__":
    test_predictor()
