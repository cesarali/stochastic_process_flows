
import pytest
from spflows.configs_classes.processes_configs import ProcessesConfig
from spflows.data.datamodules import ProcessesDatamodule

def test_processes_datasets(process_file):
    config = ProcessesConfig(process_data_file=process_file)
    datasets  = ProcessesDatamodule.get_processes_datasets(config)

def test_processes_datamodule(process_file):
    config = ProcessesConfig(process_data_file=process_file)
    config,all_datasets  = ProcessesDatamodule.get_data_and_update_config(config)
    datamodule = ProcessesDatamodule(config,all_datasets)
    datamodule.setup()
    databatch = datamodule.get_train_databatch()
    print(databatch.keys())

if __name__=="__main__":
    process_file = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\data\external\coarse_obs_systems_data_5000_points\20250129_systems_coarse_observations.json"
    test_processes_datamodule(process_file)
