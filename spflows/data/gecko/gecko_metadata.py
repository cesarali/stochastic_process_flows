import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from typing import List,Dict
from dataclasses import dataclass, fields,field
from tqdm import tqdm


from spflows.data.gecko.gecko_utils import (
    RateLimitedRequester,
    parse_raw_prices_to_dataframe
)

from spflows.data.gecko.gecko_requests import (
    get_key,
    get_coins_to_download,
    get_one_coin_metadata,
    get_coin_timeseries_raw
)

import pickle
from typing import Optional

@dataclass
class CoinMetadata:
    """
    After preprocessing stores values from
    """
    id:Optional[str] = None
    contract:Optional[str] = None
    symbol:Optional[str] = None
    name:Optional[str] = None
    #SENTIMENT
    sentiment_votes_up_percentage:Optional[int] = None
    watchlist_portfolio_users:Optional[int] = None
    market_cap_rank:Optional[int] = None
    #MARKET VALUES
    price_change_percentage_24h: Optional[float] = None
    price_change_percentage_7d: Optional[float] = None
    price_change_percentage_14d: Optional[float] = None
    price_change_percentage_30d: Optional[float] = None
    price_change_percentage_60d: Optional[float] = None
    price_change_percentage_200d: Optional[float] = None
    price_change_percentage_1y: Optional[float] = None
    price_change_percentage_1h_in_currency: Optional[float] = None
    price_change_percentage_24h_in_currency: Optional[float] = None
    price_change_percentage_7d_in_currency: Optional[float] = None
    price_change_percentage_14d_in_currency: Optional[float] = None
    price_change_percentage_30d_in_currency: Optional[float] = None
    price_change_percentage_60d_in_currency: Optional[float] = None
    price_change_percentage_200d_in_currency: Optional[float] = None
    price_change_percentage_1y_in_currency: Optional[float] = None
    current_price: Optional[float] = None
    total_value_locked: Optional[float] = None
    mcap_to_tvl_ratio: Optional[float] = None
    market_cap: Optional[float] = None
    uniswap:Optional[bool] = False

#===========================================================================================
# FUNCTIONS UTILS DICT TO CoinMetadata CLASS
#===========================================================================================
def obtain_tickers(data_tickers):
    bid_ask_spread_percentage = None
    uniswap = False
    if isinstance(data_tickers,list):
        for ticker in data_tickers:
            if isinstance(ticker,dict):
                if "market" in ticker.keys():
                    if 'uniswap' in ticker["market"]["identifier"]:
                        uniswap = True
                bid_ask_spread_percentage = ticker['bid_ask_spread_percentage']
    return {"uniswap":uniswap,
            "bid_ask_spread_percentage":bid_ask_spread_percentage}

def filter_dict_for_dataclass(input_dict, dataclass_type,currency="usd"):
    dataclass_fields = {f.name for f in fields(dataclass_type)}
    filtered_dict = {}
    for k, v in input_dict.items():
        if k in dataclass_fields:
            if not isinstance(v,dict):
                filtered_dict[k] = v
            else:
                if currency in v.keys():
                    filtered_dict[k] = v[currency]
    return filtered_dict

def prepare_dict_for_dataclass(data:dict,dataclass_type:CoinMetadata,currency="usd")->dict:
    """
    here data comes from the gecko api, the idea is to prepare the dict such that
    we are able to initialize the classes like:

    PriceChangeData(**filter_data_dict)
    """
    data_dict = {}
    data_dict.update(data)
    if "market_data"in data.keys():
        data_dict.update(data["market_data"])
    if "tickers" in data.keys():
        tickers_data = obtain_tickers(data["tickers"])
        data_dict.update(tickers_data)
    data_dict = filter_dict_for_dataclass(data_dict,dataclass_type,currency)
    return data_dict

@dataclass
class AllCoinsMetadata:
    """
    Manages the process of downloading, tracking, and storing metadata and time series data for cryptocurrencies.

    This class centralizes the workflow for:
    1. Tracking downloaded coins and their metadata.
    2. Keeping a record of coins that failed to download.
    3. Storing and managing time series data for coins.

    Attributes:
    ----------
    downloaded_coins : dict
        A dictionary mapping coin IDs to their metadata objects (e.g., {coin_id: CoinMetadata}).
    fail_to_download : List[str]
        A list of coin IDs that failed to download metadata.
    missing_time_series : List[str]
        A list of coin IDs for which time series data is missing.
    df_time_series : Dict
        A dictionary storing DataFrame objects representing time series data for each coin.
    date_string : str
        The current date (default: today's date) used for organizing file directories.
    coingecko_key : str
        API key for accessing CoinGecko services.
    all_coins_date_pathdir : Path
        The directory path for storing metadata and time series data for the specified date.
    downloaded_file : Path
        The file path where the downloaded coins metadata is stored.
    not_downloaded_file : Path
        The file path where the list of coins that failed to download is stored.
    num_total_downloads : int
        The total number of successfully downloaded coins.
    redo_names : bool
        A flag for managing deprecated price change objects.

    Methods:
    -------
    __post_init__():
        Initializes file paths and loads existing data from disk if available.
    _load_pickle(filepath, default):
        Loads a pickle file from the specified filepath. If the file does not exist, returns the default value.
    save_lists():
        Saves the downloaded and failed lists to their respective pickle files.
    download_coins_metadata(coins_to_download=None, trials=10000):
        Downloads metadata for a list of specified coins. Handles rate limiting and retries for failed downloads.
    download_df_timeseries():
        Downloads time series data for all downloaded coins, storing them as DataFrames.
    """
    downloaded_coins: dict = field(default_factory=dict)
    fail_to_download: List[str] = field(default_factory=list)
    missing_time_series: List[str] = field(default_factory=list)
    df_time_series: Dict = field(default_factory=dict)

    date_string: str = None
    coingecko_key: str = None

    all_coins_date_pathdir: Path = None
    downloaded_file: Path = None
    not_downloaded_file: Path = None

    num_total_downloads: int = 0
    redo_names: bool = False

    def __post_init__(self):
        """
        Initializes the class instance by setting up necessary directories and loading any existing data.

        This method:
        - Assigns the current date if `date_string` is not provided.
        - Creates a directory structure for organizing raw metadata and time series data.
        - Loads previously saved metadata and failure records from disk.
        """
        from spflows import data_path
        self.date_string = self.date_string or str(datetime.now().date())
        self.all_coins_date_pathdir = Path(os.path.join(data_path, "raw", "gecko", self.date_string))
        self.all_coins_date_pathdir.mkdir(parents=True, exist_ok=True)

        self.downloaded_file = self.all_coins_date_pathdir / f"downloaded_metadata_{self.date_string}.pck"
        self.not_downloaded_file = self.all_coins_date_pathdir / f"not_downloaded_{self.date_string}.pck"

        self.downloaded_coins = self._load_pickle(self.downloaded_file, default={})
        self.fail_to_download = self._load_pickle(self.not_downloaded_file, default=[])
        self.num_total_downloads = len(self.downloaded_coins)

    def _load_pickle(self, filepath: Path, default):
        """
        Loads a pickle file from the given filepath. If the file does not exist, returns the provided default value.

        Args:
        -----
        filepath : Path
            The file path to load the pickle file from.
        default : Any
            The default value to return if the file does not exist.

        Returns:
        --------
        Any
            The data loaded from the pickle file or the default value.
        """
        if filepath.exists():
            with open(filepath, "rb") as file:
                return pickle.load(file)
        return default

    def save_lists(self):
        """
        Saves the downloaded coins metadata and failed download records to their respective files.
        """
        with open(self.downloaded_file, "wb") as file:
            pickle.dump(self.downloaded_coins, file)

        with open(self.not_downloaded_file, "wb") as file:
            pickle.dump(self.fail_to_download, file)

    def download_coins_metadata(self, coins_to_download=None, trials=10000):
        """
        Downloads metadata for a given list of coin IDs. Tracks successful and failed downloads, with rate limiting.

        Args:
        -----
        coins_to_download : list, optional
            A list of coin IDs to download metadata for. If not provided, retrieves all available coin IDs.
        trials : int, optional
            The maximum number of download attempts (default is 10,000).

        Behavior:
        ---------
        - Utilizes a rate-limited API client to download data.
        - Saves progress after every 100 trials or when trials exceed the limit.
        - Updates internal records of downloaded and failed coins.
        """
        if coins_to_download is None:
            coins_to_download = get_coins_to_download()

        print(f"Coins to Download: {len(coins_to_download)}")
        rate_limiter = RateLimitedRequester()

        trial = 0
        for coin_id in coins_to_download:
            if coin_id not in self.downloaded_coins and coin_id not in self.fail_to_download:
                print(f"Downloading trial {trial}, coin ID: {coin_id}")
                rate_limiter.wait_for_rate_limit()

                coin_data_downloaded = get_one_coin_metadata(coin_id, key=self.coingecko_key)

                if coin_data_downloaded["response"]:
                    rate_limiter.up_one_download()
                    coin_data_dict = prepare_dict_for_dataclass(coin_data_downloaded, CoinMetadata)
                    coin_data_object = CoinMetadata(**coin_data_dict)

                    self.downloaded_coins[coin_id] = coin_data_object
                else:
                    self.fail_to_download.append(coin_id)
                    rate_limiter.up_one_fail()

                trial += 1
                if trial % 100 == 0 or trial >= trials:
                    self.save_lists()
                    if trial >= trials:
                        break

        self.num_total_downloads = len(self.downloaded_coins)
        self.save_lists()

    def download_df_timeseries(self):
        """
        Downloads time series data for all downloaded coins and stores it as DataFrames in `df_time_series`.

        Behavior:
        ---------
        - Checks if a time series file exists for each coin. If not, downloads it.
        - Stores downloaded time series in a DataFrame and saves it to disk.
        - Updates the `missing_time_series` list for coins without available data.
        """
        rate_limiter = RateLimitedRequester()
        for coin_id in tqdm(self.downloaded_coins.keys()):
            ts_filename = f"{coin_id}.csv"
            ts_coin_pathdir = self.all_coins_date_pathdir / ts_filename
            if not ts_coin_pathdir.exists():
                print(f"Downloading time series for: {coin_id}")
                rate_limiter.wait_for_rate_limit()
                timeseries_dict = get_coin_timeseries_raw(coin_id, key=self.coingecko_key, number_of_days=90)
                if timeseries_dict:
                    ts = parse_raw_prices_to_dataframe(timeseries_dict)
                    ts.to_csv(ts_coin_pathdir, index=True)
                    self.df_time_series[coin_id] = ts
                else:
                    self.missing_time_series.append(coin_id)
            else:
                ts = pd.read_csv(ts_coin_pathdir, index_col=0)
                ts.index = pd.to_datetime(ts.index)
                self.df_time_series[coin_id] = ts

        print(f"Obtained {len(self.df_time_series)} time series. Missing: {len(self.missing_time_series)}")

    def load_existing_timeseries(self):
        """
        Loads available time series data for coins from existing files in the designated directory.

        Behavior:
        ---------
        - Iterates through downloaded coins and attempts to load their time series files.
        - If the file exists, loads it into a DataFrame and stores it in `df_time_series`.
        - Does not attempt to handle exceptions or download missing data.
        """
        for coin_id in tqdm(self.downloaded_coins.keys()):
            ts_filename = f"{coin_id}.csv"
            ts_coin_pathdir = self.all_coins_date_pathdir / ts_filename
            if ts_coin_pathdir.exists():
                ts = pd.read_csv(ts_coin_pathdir, index_col=0)
                ts.index = pd.to_datetime(ts.index)  # Convert the index to datetime format
                self.df_time_series[coin_id] = ts

        print(f"Loaded {len(self.df_time_series)} existing time series.")

if __name__=="__main__":
    coingecko_key = get_key()
    date_string="2024-12-18"

    number_of_coins_to_download = 4000
    selected_coins = get_coins_to_download(date_string=None,
                                           key=coingecko_key,
                                           number_of_coins_to_download=number_of_coins_to_download,
                                           percentage_on_top=.25,
                                           number_of_pages=8,
                                           redo=True)
    all_coins_metadata = AllCoinsMetadata(date_string=date_string,
                                          coingecko_key=coingecko_key)

    some_coins_to_download = selected_coins[:number_of_coins_to_download]
    all_coins_metadata.download_coins_metadata(coins_to_download=some_coins_to_download)
    all_coins_metadata.download_df_timeseries()
