import json
import torch

def split_paths(observations: torch.Tensor, num_divisions: int):
    num_experiments, num_time_steps, dim = observations.shape
    assert num_time_steps % num_divisions == 0, "num_time_steps must be divisible by num_divisions"

    new_t = num_time_steps // num_divisions  # New time steps per division
    reshaped_paths = observations.view(num_experiments, num_divisions, new_t, dim)  # Reshape to group divisions
    reshaped_paths = reshaped_paths.permute(0, 1, 2, 3).reshape(num_experiments * num_divisions, new_t, dim)  # Merge first two dims

    return reshaped_paths

def get_data_paths(data_file:str,dataset_str_name: str = "Duffing",tau:float=0.002,num_divisions:int=25):
    """
    Args:
        full_data
    """
    with open(data_file,"r") as f:
        full_data = json.load(f)

    for data in full_data:
        if data["name"] == dataset_str_name and data["tau"] == tau:
            observations = data["observations"]
    observations = torch.Tensor(observations).squeeze()
    paths_observations = split_paths(observations,num_divisions)
    return paths_observations
