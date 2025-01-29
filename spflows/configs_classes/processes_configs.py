from dataclasses import dataclass, field
from typing import List, Optional, Any
import yaml

@dataclass
class ProcessesConfig:
    experiment_dir:str = None
    experiment_name:str = "processes"
    experiment_indentifier:Optional[str] = None
    results_path:Optional[str] = None

    # Data configuration
    process_data_file : str = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\data\external\coarse_obs_systems_data_5000_points\20250129_systems_coarse_observations.json"
    tau:float=0.002 # 0.02, 0.01, 0.002
    num_divisions:int = 25
    dataset_str_name: str = "Duffing" # Duffing, Hopf, Damped Linear, Syn Drift, Damped Cubic, Glycolysis, Double Well, Wang
    date_str: str = "2024-12-18"
    freq: str = "H"

    # Data Shapes
    prediction_length: int = 24 # updated at datamodule
    context_length: Optional[int] = None # updated at datamodule
    history_length: Optional[int] = None # updated at datamodule
    lags_seq: Optional[List[int]] = None # updated at datamodule
    time_features: Optional[List[Any]] = None # updated at datamodule
    covariance_dim: int = 1 # updated at datamodule
    input_size: int = 1  # updated at datamodule
    target_dim: int = 1  # updated at datamodule

    batch_size: int = 64
    pick_incomplete: bool = True
    scaling: bool = True
    shuffle_buffer_length: Optional[int] = None
    cache_data: bool = False
    weight_decay: float = 1e-6
    maximum_learning_rate: float = 1e-2
    num_workers: int = 4
    prefetch_factor: int = 3

    # Training configuration
    epochs: int = 100
    num_batches_per_epoch: int = 50
    num_batches_per_epoch_val: int = 30
    learning_rate: float = 1e-3
    seed: int = 1
    patience:Optional[int] = 10
    clip_gradient:Optional[float] = None

    #Denoising Models
    noise: str = "gp"
    diffusion_steps:int = 100
    beta_end: float = 0.1
    beta_schedule: str = "linear"
    loss_type: str = "l2"
    num_samples:int = 10

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

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ProcessesConfig':
        with open(yaml_path, 'r') as file:
            params_dict = yaml.safe_load(file)
        return cls(**params_dict)
