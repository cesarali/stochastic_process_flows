import numpy as np
from gluonts.dataset.common import (
    CategoricalFeatureInfo,
    MetaData,
)
from gluonts.dataset.artificial import ArtificialDataset
from spflows.data.gecko.gecko_metadata import AllCoinsMetadata,CoinMetadata

class CoinGeckoDataset(ArtificialDataset):
    """
    This dataset is set to behave exactly as the dataset
    obtained from get_dataset in gluon_ts, it will generate 6 dimensions
    index 0-2 bitcoin price, market_caps, total_volumes
    index 3-5 altcoin price, market_caps, total_volumes

    for portfolio creation prediction length is set to 96 hours (4 days)
    freq in hours H
    """
    def __init__(self,
                 coin_id,
                 df_freq,
                 df_bitcoin_freq,
                 prediction_length: int = 96,  # change for days (portfolio sensitivity)
                 freq_str: str = "H",
                 include_market_cap: bool = True,
                 include_volumes: bool = True):
        """
        df_freq: pd.DataFrame constructed with frequencies given by bitcoin init
        df_bitcoin_freq: pd.DataFrame constructed with frequencies given by bitcoin init
        """
        super().__init__(freq_str)
        self.coin_id = coin_id
        self.df_freq = df_freq
        self.df_bitcoin_freq = df_bitcoin_freq
        self.prediction_length = prediction_length
        self.freq_str = freq_str
        self.include_market_cap = include_market_cap
        self.include_volumes = include_volumes
        self._set_data_entries()

    @property
    def metadata(self):
        return MetaData(freq=self.freq_str,
                        feat_static_cat=[CategoricalFeatureInfo(name='feat_static_cat_0', cardinality='6')],
                        prediction_length=self.prediction_length)

    def _set_data_entries(self):
        """
        Set data entries for the dataset.
        """
        alt_coin_prices = self.df_freq["prices"].values
        alt_coin_market_caps = self.df_freq["market_caps"].values
        alt_coin_total_volumes = self.df_freq["total_volumes"].values

        bitcoin_prices = self.df_bitcoin_freq["prices"].values
        bitcoin_market_caps = self.df_bitcoin_freq["market_caps"].values
        bitcoin_total_volumes = self.df_bitcoin_freq["total_volumes"].values

        timestamp = self.df_bitcoin_freq.index[0]
        period = timestamp.to_period(freq='H')

        self.data_list = [
            {'target': bitcoin_prices, 'start': period, 'feat_static_cat': np.array([0]), 'item_id': 0},
            {'target': bitcoin_market_caps, 'start': period, 'feat_static_cat': np.array([0]), 'item_id': 1},
            {'target': bitcoin_total_volumes, 'start': period, 'feat_static_cat': np.array([0]), 'item_id': 2},
            {'target': alt_coin_prices, 'start': period, 'feat_static_cat': np.array([0]), 'item_id': 3},
            {'target': alt_coin_market_caps, 'start': period, 'feat_static_cat': np.array([0]), 'item_id': 4},
            {'target': alt_coin_total_volumes, 'start': period, 'feat_static_cat': np.array([0]), 'item_id': 5}
        ]
        print(self.data_list)

    @property
    def train(self):
        return self.data_list

    @property
    def test(self):
        return self.data_list
