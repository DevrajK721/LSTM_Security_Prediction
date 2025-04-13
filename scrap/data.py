
# Data Processing Class for LSTM 
"""
This module contains the DataProcessor class, which is responsible for fetching, cleaning, and processing stock data for LSTM model training. 
It is used for the following:
- Fetching stock data 
- Fetching crypto data from Binance
- Cleaning the data into a usable dataframe
- Converting to dataframe in clean format
- Setting up the windowed configuration for the LSTM 
- Computing Log-Normalized Returns (Potentially, haven't done this yet)
"""

import yfinance as yf
from binance.client import Client as BC
import pandas as pd
import numpy as np 
import os 
import sys
import matplotlib.pyplot as plt
from typing import *
import datetime 
import json
import os

class DataProcessor:
    def __init__(self, ticker, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_stock_data(self):
        # Fetch the stock data
        stock_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        # Save the data to a CSV file
        self.csv_file = f"data/{self.ticker}_data.csv"
        stock_data.to_csv(self.csv_file)
        
        print(f"Stock data for {self.ticker} saved to {self.csv_file}")

        # Remove the top 3 lines and replace the top line with column headers
        with open(self.csv_file, 'r') as file:
            lines = file.readlines()[3:]  # Skip the first 3 lines

        # Replace the first line with the desired headers
        lines[0] = "Date,Open,High,Low,Close,Volume\n"

        # Write the modified lines back to the file
        with open(self.csv_file, 'w') as file:
            file.writelines(lines)

    def initialize_binance_client(self):
        with open('secrets/secrets.json', 'r') as file:
            secrets = json.load(file)
            self.binance_api_key = secrets['BINANCE_API_KEY']
            self.binance_api_secret = secrets['BINANCE_API_SECRET']
            self.binance_client = BC(self.binance_api_key, self.binance_api_secret)

    def fetch_crypto_data(self):
        self.initialize_binance_client() # Get the API keys and initialize the client

        symbol = 'SOLUSDT'
        interval = BC.KLINE_INTERVAL_1DAY

        columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
                   'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore']
        self.df = pd.DataFrame(self.binance_client.get_historical_klines(symbol, interval, self.start_date, self.end_date), columns=columns)
        self.df['Open Time'] = pd.to_datetime(self.df['Open Time'], unit='ms')
        self.df['Close'] = self.df['Close'].astype(float)
        self.df['Open'] = self.df['Open'].astype(float)
        self.df['High'] = self.df['High'].astype(float)
        self.df['Low'] = self.df['Low'].astype(float)
        self.df['Volume'] = self.df['Volume'].astype(float)
        self.df = self.df[['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def clean_data(self):
        # Load DataFrame Object from CSV
        self.df = pd.read_csv(self.csv_file)
        self.df = self.df[['Date', 'Close']] # Select only the Date and Close columns

        self.df['Date'] = pd.to_datetime(self.df['Date']) # Convert the Date column to datetime
        self.df.index = self.df.pop('Date')

    def str_to_datetime(self, s: str):
        split = s.split('-')
        year, month, day = int(split[0]), int(split[1]), int(split[2])
        return datetime.datetime(year=year, month=month, day=day)

    def df_to_windowed_df(self, dataframe, first_date_str, last_date_str, n: int):
        first_date = self.str_to_datetime(first_date_str)
        last_date  = self.str_to_datetime(last_date_str)

        self.n = n

        target_date = first_date
        
        dates = []
        X, Y = [], []

        last_time = False
        while True:
            df_subset = dataframe.loc[:target_date].tail(n+1)
            
            if len(df_subset) != n+1:
                print(f'Error: Window of size {n} is too large for date {target_date}')
                return

            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

            next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
            
            if last_time:
                break
            
            target_date = next_date

            if target_date == last_date:
                last_time = True
            
        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates
        
        X = np.array(X)
        for i in range(0, n):
            X[:, i]
            ret_df[f'Target-{n-i}'] = X[:, i]
        
        ret_df['Target'] = Y

        self.windowed_df = ret_df

        return self.windowed_df
    

    def windowed_df_to_numpy(self):
        df_as_numpy = self.windowed_df.to_numpy()
        self.dates = df_as_numpy[:, 0]

        middle_matrix = df_as_numpy[:, 1:-1]
        self.X = middle_matrix.reshape(len(self.dates), middle_matrix.shape[1], 1)
        self.X = self.X.astype(np.float32)

        self.Y = df_as_numpy[:, -1]
        self.Y = self.Y.astype(np.float32)

        return self.dates, self.X, self.Y
    

if __name__ == "__main__":
    # Example Usage 
    x = DataProcessor('AAPL', '2020-01-01', '2023-10-01')
    x.fetch_crypto_data()

    print(x.df.head())