from tqdm import tqdm
import requests
from dataclasses import field,fields
import time
import random
import os
import random
import pickle
from datetime import datetime
from spflows import data_path

def get_key():
    from spflows import project_path
    from random import choice
    keys_file = os.path.join(project_path,"KEYS.txt")
    key = choice(open(keys_file,"r").read().split("\n"))
    return key

def get_request(url,coin_gecko_key=None):
    if coin_gecko_key is not None:
        headers = {
            "x-cg-demo-api-key": coin_gecko_key,
        }
        # Sending a GET request to the URL
        response = requests.get(url,headers=headers)
    else:
        response = requests.get(url)

    # Checking if the request was successful
    if response.status_code == 200:
        # Convert the JSON response into a Python dictionary
        data = response.json()
        return data
    else:
        return None

def filter_coin_id_and_contract_deprecated(coin_ids,contract="ethereum"):
    if "id" in coin_ids.keys() and "platforms" in coin_ids.keys():
        if contract in coin_ids["platforms"]:
            return coin_ids["id"]
        else:
            return None
    else:
        return None

def filter_coin_id_and_contract(coin_ids, contract="ethereum"):
    """
    Simplified filter function to accept any coin data without filtering by platform.

    Args:
        coin_ids (dict): Dictionary containing coin information.
        contract (str, optional): Unused parameter, kept for compatibility. Defaults to "ethereum".

    Returns:
        str: The coin ID if present, otherwise None.
    """
    return coin_ids.get("id", None)

def get_all_coins_and_contracts_data(date_string,key):
    if date_string is None:
        date_string = str(datetime.now().date())
    coins_pathdir = data_path / "raw" / "gecko" / date_string
    filename = coins_pathdir / "all_coins.pck"

    if not os.path.exists(coins_pathdir):
        os.makedirs(coins_pathdir)

    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    if not os.path.exists(filename):
        data = get_request(url,key)
        if data:
            pickle.dump(data,open(filename,"wb"))
            return data
        else:
            return None
    else:
        data = pickle.load(open(filename,"rb"))
        return data

def get_all_coins_and_markets(date_string,key=None,number_of_pages=3,tor=False):
    if date_string is None:
        date_string = str(datetime.now().date())
    coins_pathdir = data_path / "raw" / "gecko" / date_string
    if not os.path.exists(coins_pathdir):
        os.makedirs(coins_pathdir)

    url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
    data = []
    for page in range(number_of_pages):
        filename = f"all_coins_markets_{page}.pck"
        filename = coins_pathdir / filename
        if not os.path.exists(filename):
            url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}&sparkline=false&locale=en"
            data_ = get_request(url,key)

            if data_:
                pickle.dump(data_,open(filename,"wb"))
                data.extend(data_)
        else:
            data.extend(pickle.load(open(filename,"rb")))
    return data

def get_coins_to_download(date_string=None,
                          key=None,
                          number_of_coins_to_download=3000,
                          percentage_on_top = .1,
                          number_of_pages=4,
                          redo=True):
    """
    Retrieves a list of coins to download, prioritizing the top-performing coins and adding random ones
    to meet the specified number. The function saves the selected coins to a file or reloads them
    if they already exist (unless `redo` is set to True).

    Args:
        date_string (str, optional): The date string used to organize data. Defaults to today's date.
        key (str, optional): API key or authentication token for accessing coin data. Defaults to None.
        number_of_coins_to_download (int, optional): The total number of coins to select for downloading. Defaults to 3000.
        percentage_on_top (float, optional): Proportion of top-performing coins to include. Defaults to 0.1 (10%).
        number_of_pages (int, optional): Number of pages to fetch from the top coins' market data. Defaults to 4.
        redo (bool, optional): If True, reselects coins even if the selection file exists. Defaults to True.

    Returns:
        list: A list of filtered coin IDs and their corresponding contract data for downloading.

    Workflow:
        1. Determine the storage directory and file path based on the `date_string`.
        2. If the file with selected coins exists and `redo` is False:
           - Load the coin data from the file.
        3. Otherwise:
           - Fetch all available coins and their contract data.
           - Shuffle the coins to ensure randomness.
           - Fetch top-performing coins' data from the market and sort them.
           - Select a portion of the top coins based on `percentage_on_top`.
           - Fill the remainder with randomly selected coins from the shuffled list.
           - Filter out coins that don't meet certain criteria using `filter_coin_id_and_contract`.
           - Save the final list of selected coins to a file.
        4. Return the list of selected coins.

    Notes:
        - The function relies on external functions:
            - `get_all_coins_and_contracts_data(date_string, key)`: Fetches all available coin data.
            - `get_all_coins_and_markets(date_string, key, number_of_pages)`: Fetches market data for the top coins.
            - `filter_coin_id_and_contract(coin_data)`: Filters and validates coin data based on custom criteria.
        - The selected coins are saved in a pickle file at `data_path/raw/gecko/<date_string>/selected_coins.pck`.
        - If the directory does not exist, it is created automatically.
    """
    # If no date string is provided, use the current date
    if date_string is None:
        date_string = str(datetime.now().date())

    # Define the path for storing coin data, organized by date
    coins_pathdir = data_path / "raw" / "gecko" / date_string

    # Create the directory if it does not already exist
    if not os.path.exists(coins_pathdir):
        os.makedirs(coins_pathdir)

    # Define the filename for storing selected coins
    filename = "selected_coins.pck"
    filename = coins_pathdir / filename

    # If the file doesn't exist or a redo is requested, generate the list of coin IDs to download
    if not os.path.exists(filename) or redo:
        # Determine how many coins to select based on top ranking and random selection
        number_on_top = int(percentage_on_top * number_of_coins_to_download)
        number_on_random = number_of_coins_to_download - number_on_top

        # Retrieve all coins and their associated contract data
        data_1 = get_all_coins_and_contracts_data(date_string, key)
        random.shuffle(data_1)  # Shuffle the data for random selection

        # Display the total number of available coins
        print(f"Coins available: {len(data_1)}")
        print(len(data_1))

        # Create a dictionary mapping coin IDs to their contract data
        id_to_contracts = {coin["id"]: coin for coin in data_1}

        # Retrieve market data for coins and sort the IDs by rank
        data_2 = get_all_coins_and_markets(date_string, key, number_of_pages)
        sorted_ids = [coin["id"] for coin in data_2]

        # Initialize the list of IDs to download
        ids_to_download = []

        # Select the top-ranked coins
        from_top = 0
        while from_top < min(len(sorted_ids), number_on_top):
            top_id = sorted_ids[from_top]
            coin_ids_and_contract = id_to_contracts[top_id]  # Get the contract data for the top coin
            filtered_ = filter_coin_id_and_contract(coin_ids_and_contract)  # Apply filtering
            if filtered_:
                ids_to_download.append(filtered_)  # Add to the download list if it passes the filter
            from_top += 1

        # Calculate how many additional coins need to be selected randomly
        number_on_random = number_of_coins_to_download - len(ids_to_download)
        on_random = 0

        # Select random coins until the required number is reached
        for coin in data_1:
            coin_id = coin["id"]
            if coin_id not in ids_to_download:  # Ensure the coin is not already selected
                coin_ids_and_contract = id_to_contracts[coin_id]  # Get the contract data
                filtered_ = filter_coin_id_and_contract(coin_ids_and_contract)  # Apply filtering
                if filtered_:
                    ids_to_download.append(filtered_)  # Add to the download list if it passes the filter
                    on_random += 1
                if on_random > number_on_random:  # Stop if enough random coins are selected
                    break

        # Save the selected IDs to a file for future use
        pickle.dump(ids_to_download, open(filename, "wb"))
    else:
        # Load the existing list of IDs from the file
        ids_to_download = pickle.load(open(filename, "rb"))

    # Return the list of IDs to download
    return ids_to_download

def get_one_coin_metadata(id="archangel-token",key=None):
    # URL of the CoinGecko API for the Archangel Token contract details
    url = f"https://api.coingecko.com/api/v3/coins/{id}?tickers=true&market_data=true&community_data=true&sparkline=true"

    data = get_request(url,key)
    if data:
        data.update({"response":True,"id":id})
        return data
    else:
        data = {"response":False,"id":id}
        return data

def get_coin_timeseries_raw(coin_id,key=None,number_of_days=90,tor=False):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={number_of_days}"
    data = get_request(url,key)
    return data

if __name__=="__main__":
    date_string = None
    coingecko_key = get_key()

    number_of_coins_to_download = 15

    selected_coins = get_coins_to_download(date_string=None,
                                           key=coingecko_key,
                                           number_of_coins_to_download=number_of_coins_to_download,
                                           percentage_on_top=.1,
                                           number_of_pages=5)
    print(selected_coins)
