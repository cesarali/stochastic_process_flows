from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
import torch
from spflows.configs_classes.forecasting_configs import ForecastingModelConfig
from torch import nn

from spflows.models.forecasting import (
    TimeGradTrainingNetwork_AutoregressiveOld, TimeGradPredictionNetwork_AutoregressiveOld,
    TimeGradTrainingNetwork_Autoregressive, TimeGradPredictionNetwork_Autoregressive,
    TimeGradTrainingNetwork_All, TimeGradPredictionNetwork_All,
    TimeGradTrainingNetwork_RNN, TimeGradPredictionNetwork_RNN,
    TimeGradTrainingNetwork_Transformer, TimeGradPredictionNetwork_Transformer,
    TimeGradTrainingNetwork_CNN, TimeGradPredictionNetwork_CNN,
)
from spflows.utils import NotSupportedModelNoiseCombination

class ScoreModule(LightningModule):
    """
    """
    train_dynamical_module:nn.Module

    def __init__(
        self,
        config:ForecastingModelConfig,
    ):
        super(ScoreModule,self).__init__()

        self.config = config  # Store the config for potential future reference
        self.epochs = config.epochs
        self.network = config.network
        self.batch_size = config.batch_size
        self.num_batches_per_epoch = config.num_batches_per_epoch
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.maximum_learning_rate = config.maximum_learning_rate
        
        # Additional attributes for training state
        self.best_loss = float("inf")
        self.waiting = 0
        self.best_net_state = None

        # Initialize other model-specific components as needed
        self.set_networks_class()
        self.train_dynamical_module =  self.create_training_network()

    def set_networks_class(self):
        """defines the network class to be initialized"""
        network  = self.config.network
        noise = self.config.noise
        # Load model
        if network == 'timegrad':
            if noise != 'normal':
                raise NotSupportedModelNoiseCombination
            train_dynamic_class, prediction_dynamic_class = TimeGradTrainingNetwork_Autoregressive, TimeGradPredictionNetwork_Autoregressive
        elif network == 'timegrad_old':
            if noise != 'normal':
                raise NotSupportedModelNoiseCombination
            train_dynamic_class, prediction_dynamic_class = TimeGradTrainingNetwork_AutoregressiveOld, TimeGradPredictionNetwork_AutoregressiveOld
        elif network == 'timegrad_all':
            train_dynamic_class, prediction_dynamic_class = TimeGradTrainingNetwork_All, TimeGradPredictionNetwork_All
        elif network == 'timegrad_rnn':
            train_dynamic_class, prediction_dynamic_class = TimeGradTrainingNetwork_RNN, TimeGradPredictionNetwork_RNN
        elif network == 'timegrad_transformer':
            train_dynamic_class, prediction_dynamic_class = TimeGradTrainingNetwork_Transformer, TimeGradPredictionNetwork_Transformer
        elif network == 'timegrad_cnn':
            train_dynamic_class, prediction_dynamic_class = TimeGradTrainingNetwork_CNN, TimeGradPredictionNetwork_CNN
        
        self.train_dynamic_class = train_dynamic_class
        self.prediction_dynamic_class = prediction_dynamic_class
    
    def create_training_network(self):
        """initilizes the dynamical model"""
                
        return self.train_dynamic_class(
            noise=self.config.noise,
            input_size=self.config.input_size,
            target_dim=self.config.target_dim,
            num_layers=self.config.num_layers,
            num_cells=self.config.num_cells,
            cell_type=self.config.cell_type,
            history_length=self.config.history_length,
            context_length=self.config.context_length,
            prediction_length=self.config.prediction_length,
            dropout_rate=self.config.dropout_rate,
            cardinality=self.config.cardinality,
            embedding_dimension=self.config.embedding_dimension,
            diff_steps=self.config.diff_steps,
            loss_type=self.config.loss_type,
            beta_end=self.config.beta_end,
            beta_schedule=self.config.beta_schedule,
            residual_layers=self.config.residual_layers,
            residual_channels=self.config.residual_channels,
            dilation_cycle_length=self.config.dilation_cycle_length,
            lags_seq=self.config.lags_seq,
            scaling=self.config.scaling,
            conditioning_length=self.config.context_length,
            time_feat_dim=self.config.time_feat_dim,
        )

    # ------------------------------ LITHNING IMPLEMENTATION -------------------------------------
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.maximum_learning_rate,
            steps_per_epoch=self.num_batches_per_epoch,
            epochs=self.epochs,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        inputs = [v.to(self.device) for v in batch.values()]
        loss = self(*inputs)
        if isinstance(loss, (list, tuple)):
            loss = loss[0]

        # Gradient clipping
        if self.clip_gradient is not None:
            clip_grad_norm_(self.parameters(), self.clip_gradient)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = [v.to(self.device) for v in batch.values()]
        with torch.no_grad():
            output = self(*inputs)
        loss = output[0] if isinstance(output, (list, tuple)) else output
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("val_loss", float("inf"))

        # Early stopping logic
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_net_state = deepcopy(self.state_dict())
            self.waiting = 0
        elif self.waiting >= self.patience:
            self.trainer.should_stop = True
        else:
            self.waiting += 1

    def on_train_end(self):
        if self.best_net_state:
            self.load_state_dict(self.best_net_state)