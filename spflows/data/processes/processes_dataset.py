import torch
import pandas as pd
import numpy as np
from gluonts.dataset.common import (
    CategoricalFeatureInfo,
    MetaData,
)
from gluonts.dataset.artificial import ArtificialDataset


class ProcessesDataset(ArtificialDataset):
    """
    This dataset is set to behave exactly as the dataset
    obtained from get_dataset in gluon_ts, it will generate 6 dimensions
    index 0-2 bitcoin price, market_caps, total_volumes
    index 3-5 altcoin price, market_caps, total_volumes

    for portfolio creation prediction length is set to 96 hours (4 days)
    freq in hours H
    """
    def __init__(self,
                 path_tensor,
                 prediction_length: int = 20,  # change for days (portfolio sensitivity)
                 freq_str: str = "H"):
        """
        df_freq: pd.DataFrame constructed with frequencies given by bitcoin init
        df_bitcoin_freq: pd.DataFrame constructed with frequencies given by bitcoin init
        """
        super().__init__(freq_str)
        self.path_tensor:torch.Tensor = path_tensor
        self.dimensions = str(path_tensor.size(1))
        self.prediction_length = prediction_length
        self.freq_str = freq_str
        self._set_data_entries()

    @property
    def metadata(self):
        return MetaData(freq=self.freq_str,
                        feat_static_cat=[CategoricalFeatureInfo(name='feat_static_cat_0', cardinality=self.dimensions)],
                        prediction_length=self.prediction_length)

    def _set_data_entries(self):
        """
        Set data entries for the dataset.
        """
        dimensions = self.path_tensor.size(1)
        timestamp = pd.Timestamp('2024-09-19 16:01:55.551000', freq='H')

        self.data_list = []
        for dimension in range(dimensions):
            period = timestamp.to_period(freq='H')
            self.data_list.append({'target': self.path_tensor[:,dimension].numpy(),
                                   'start': period,
                                   'feat_static_cat': np.array([0]),
                                   'item_id': dimension})
        print(self.data_list)

    @property
    def train(self):
        return self.data_list

    @property
    def test(self):
        return self.data_list
