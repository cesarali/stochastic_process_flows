from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class ForecastingModelConfig:

    # Data configuration
    dataset: str = "electricity_nips"
    freq: str = "H"  
    prediction_length: int = 24 
    context_length: Optional[int] = None # updated at datamodule
    history_length: Optional[int] = None # updated at datamodule
    lags_seq: Optional[List[int]] = None # updated at datamodule 
    time_features: Optional[List[Any]] = None # updated at datamodule
    covariance_dim: int = 1 # updated at datamodule
    input_size: int = 1  # Replace with actual input size
    target_dim: int = 1  # Replace with actual target dimension

    batch_size: int = 64
    pick_incomplete: bool = True

    scaling: bool = True
    shuffle_buffer_length: Optional[int] = None
    cache_data: bool = False
    weight_decay: float = 1e-6
    maximum_learning_rate: float = 1e-2
    num_workers: int = 2
    prefetch_factor: int = 3

    # Training configuration
    epochs: int = 100
    num_batches_per_epoch: int = 50
    learning_rate: float = 1e-3
    seed: int = 1

    #Denoising Models
    noise: str = "gp"
    diffusion_steps:int = 100
    diff_steps: int = 100
    beta_end: float = 0.1
    beta_schedule: str = "linear"
    loss_type: str = "l2"

    # Model architecture
    network: str = "timegrad_rnn"
    num_layers: int = 2
    num_cells: int = 100
    cell_type: str = "GRU"
    num_parallel_samples: int = 100
    dropout_rate: float = 0.1
    cardinality: List[int] = field(default_factory=lambda: [1])
    embedding_dimension: int = 5
    hidden_dim: int = 100

    residual_layers: int = 8
    residual_channels: int = 8
    dilation_cycle_length: int = 2

    old: bool = False
    time_feat_dim: int = 4

    extra_args: dict = field(default_factory=dict)

