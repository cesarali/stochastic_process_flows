import torch
import pprint
import pytest

def test_forward():
    from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
    from spflows.data.datamodules import ForecastingDataModule
    from spflows.models.forecasting.score_lightning import ScoreModule

    config = ForecastingModelConfig()
    datamodule = ForecastingDataModule(config)
    datamodule.prepare_data()
    datamodule.setup()
    config = datamodule.update_config(config)
    databatch = datamodule.get_train_databatch()
    module = ScoreModule(config)
    inputs = [v for v in databatch.values()]
    loss = module.train_dynamical_module(*inputs)
    assert loss is not None

def test_predictor():
    from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
    from spflows.data.datamodules import ForecastingDataModule
    from spflows.models.forecasting.score_lightning import ScoreModule
    from gluonts.evaluation.backtest import make_evaluation_predictions

    config = ForecastingModelConfig(diffusion_steps=10)
    datamodule = ForecastingDataModule(config)
    config = datamodule.update_config(config)
    module = ScoreModule(config)

    predictor = module.create_predictor_network(
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