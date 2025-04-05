import yfinance as yf 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from typing import *
import datetime
import os 

class DataProcessor:
    def __init__(self, ticker, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
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
    

    


        