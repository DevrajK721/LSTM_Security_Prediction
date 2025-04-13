# Data Processing Class 

# Core Imports 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 

# Additional Imports 
from binance.client import Client as BC 
import json 
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import * 

class DataProcessor:
    def __init__(self, secrets_path: str = '../secrets/secrets.json'):
        # Verify existence of secrets.json file 
        if not os.path.exists(secrets_path):
            raise FileNotFoundError(f"secrets.json file not found at {secrets_path}. Please ensure the provided path is correct (Hint: Use `pwd` in command line to verify current directory)")
        else:
            print(f"secrets.json file found at {secrets_path}. Beginning initialization of DataProcessor class.")
            self.secrets_path = secrets_path
        
        # Load secrets from the JSON file
        with open(self.secrets_path, 'r') as file:
            vals = json.load(file)
            self.binance_api_key = vals['BINANCE_API_KEY']
            self.binance_api_secret = vals['BINANCE_API_SECRET']
            self.frequency = vals['Trading Frequency (Yearly/Monthly/Weekly/Daily)']
            self.start_date = vals['Starting Date (YYYY-MM-DD)']
            self.end_date = vals['Ending Date (YYYY-MM-DD)']
            self.base_currency = vals['Base Currency']
            self.tickers = vals['Tickers of Interest']

        # Check whether all information has been loaded correctly 
        if not all([self.binance_api_key, self.binance_api_secret, self.start_date, self.end_date, self.base_currency, self.tickers]):
            raise ValueError("One or more required fields are missing in the secrets.json file.")
        
        # Initialize the Binance Client
        self.binance_client = BC(self.binance_api_key, self.binance_api_secret)
        print("Binance client initialized successfully.")

        print("All required fields loaded successfully from secrets.json.")
        print(f"{len(self.tickers)} tickers loaded successfully.")
        print(f"Frequency: {self.frequency}")
        print(f"Starting date: {self.start_date}")
        print(f"Ending date: {self.end_date}")
        print(f"Base currency: {self.base_currency}")
        print(f"Tickers: {self.tickers}")
        print("Initialization of DataProcessor class completed successfully.")

        if self.frequency == 'Daily':
            interval = BC.KLINE_INTERVAL_1DAY
        elif self.frequency == 'Weekly':
            interval = BC.KLINE_INTERVAL_1WEEK
        elif self.frequency == 'Monthly':
            interval = BC.KLINE_INTERVAL_1MONTH
        elif self.frequency == 'Yearly':
            interval = BC.KLINE_INTERVAL_1YEAR
        else:
            raise ValueError("Invalid frequency. Choose from 'Daily', 'Weekly', 'Monthly', or 'Yearly'.")

        # Convert tickers to trading pairs 
        self.tickers = [f"{ticker}{self.base_currency}" for ticker in self.tickers]
        

        # Fetch all the crypto data requested 
        self.crypto_data = {}
        # Find the path to the project root 
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        for ticker in tqdm(self.tickers, desc="Fetching Crypto Data", unit="pair",
              ncols=80, bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
              colour="green", leave=True, dynamic_ncols=True):
            # Check if the ticker data already exists
            ticker_file_path = os.path.join(data_dir, f"{ticker}.csv")
            if os.path.exists(ticker_file_path):
                print(f"Data for {ticker} already exists at {ticker_file_path}. Skipping download.")
                # Load existing data
                data = pd.read_csv(ticker_file_path)
                self.crypto_data[ticker] = data
                continue
            
            symbol = ticker 
            columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
                'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
                'Taker Buy Quote Asset Volume', 'Ignore']
            data = pd.DataFrame(self.binance_client.get_historical_klines(symbol, interval, self.start_date, self.end_date), columns=columns)
            data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
            data['Close'] = data['Close'].astype(float)
            data['Open'] = data['Open'].astype(float)
            data['High'] = data['High'].astype(float)
            data['Low'] = data['Low'].astype(float)
            data['Volume'] = data['Volume'].astype(float)
            data = data[['Open Time', 'Close']]

            # Compute log daily returns column
            data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data.dropna(inplace=True)

            # Compute windowed returns column (n = 30 (Adjust when testing))
            windowed_data = self.df_to_windowed_df(dataframe=data)
            if windowed_data is None:
                print(f"Warning: Could not create windowed data for {ticker}. Skipping.")
                continue
            data = windowed_data
            data.dropna(inplace=True)

            # Save the processed data to CSV
            data.to_csv(ticker_file_path, index=False)
            self.crypto_data[ticker] = data
            print(f"Data for {ticker} saved successfully to {ticker_file_path}")
        
        print("All crypto data fetched successfully.")
        # Make sure crypto_data isn't empty
        if not self.crypto_data:
            print("Warning: No crypto data was successfully processed.")
        
        print("Data processing completed successfully.")
            


    # Convert string date to datetime object
    def str_to_datetime(self, date_str: str) -> datetime:
        return datetime.strptime(date_str, '%Y-%m-%d')
    
    # Compute windowed returns column 
    def df_to_windowed_df(self, dataframe: pd.DataFrame, n: int = 30):
        first_date = self.str_to_datetime(self.start_date)
        last_date  = self.str_to_datetime(self.end_date)

        self.n = n
        dataframe = dataframe.set_index('Open Time')
        sorted_dates = sorted(dataframe.index)
        if len(sorted_dates) <= n:
            print(f"Error: Not enough data points ({len(sorted_dates)}) for window size {n}")
            return None

        target_date_index = n  # Start where we have n previous points
        dates = []
        X, Y = [], []

        while target_date_index < len(sorted_dates):
            target_date = sorted_dates[target_date_index]

            df_subset = dataframe.loc[:target_date].tail(n+1)
            if len(df_subset) != n+1:
                print(f'Error: Window of size {n} is too large for date {target_date}')
                target_date_index += 1
                continue

            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

            # Update target_date_index to process the next date
            target_date_index += 1

        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates
        X = np.array(X)
        for i in range(n):
            ret_df[f'Lag-{n - i}'] = X[:, i]
        ret_df['Target'] = Y

        self.windowed_df = ret_df

        return self.windowed_df
    
    
if __name__ == "__main__":
    dp = DataProcessor() # One-Liner is all it takes to initialize the DataProcessor class)
    # For example, let's print "BTCUSDT" head data 
    print(dp.crypto_data['BTCUSDC'].head())
    





            


            






        

    