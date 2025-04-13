# Base level time series models for analytical comparisons to LSTM 
"""
This module contains the base level time series models for analytical comparisons to LSTM.
It includes:
- ARIMA 
- GARCH
- Plotting Features for the autocorrelation function (ACF)
"""

from statsmodels.tsa.arima_model import ARIMA # Might write my own
from arch import arch_model # Might write my own 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from typing import *
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf


class TS_Models:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def acf_plot(self):
        if 'Close' not in self.df.columns:
            raise ValueError("DataFrame does not contain the 'Close' column.")
        
        # Import statsmodels function for autocorrelation
        
        # Compute autocorrelation
        LAGS = 50
        closes = np.array(self.df.Close, dtype=np.float64)
        autocorr = acf(closes, nlags=LAGS-1, fft=True)
        
        # Plot the autocorrelation function
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 6))
        
        # Create stem plot (lines with circles)
        markerline, stemlines, baseline = plt.stem(
            range(len(autocorr)), autocorr, linefmt='r-', markerfmt='ro', basefmt='r-')
        plt.setp(markerline, 'markerfacecolor', 'r')
        plt.setp(markerline, 'markeredgecolor', 'r')
        plt.setp(stemlines, 'color', 'r', 'alpha', 0.7)
        
        # Plot confidence interval
        plt.axhspan(-1.96/np.sqrt(len(closes)), 1.96/np.sqrt(len(closes)), alpha=0.2, color='grey')
        
        # Add labels and formatting
        plt.title('ACF Plot', color='white')
        plt.xlabel(r'Lag ($k$)', color='white')
        plt.ylabel(r'Correlation ($\text{Corr}(x_t, x_{t-k})$)', color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Create a sample DataFrame for sample stock data 
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    np.random.seed(721)
    w = np.random.normal(0, 1, 1000)
    x = np.zeros(1000)
    for t in range(1, 1000):
        # AR(1) process
        x[t] = x[t - 1] + w[t]
    
    df = pd.DataFrame({'Date': dates, 'Close': x})
    df.set_index('Date', inplace=True)
    df = df.astype(np.float64)
    df = df.dropna()

    # Plot the ACF for the Random Walk Data 
    ts_model = TS_Models(df)
    ts_model.acf_plot() # Plot the ACF for the random walk data should be quite high for all the lags

    # We can now use the differencing operator make the data stationary
    # Create a differenced version of the data
    diff_data = np.diff(df['Close'])
    df_diff = pd.DataFrame({'Date': dates[1:], 'Close': diff_data})
    df_diff.set_index('Date', inplace=True)
    df_diff = df_diff.astype(np.float64)

    ts_model_diff = TS_Models(df_diff)
    ts_model_diff.acf_plot() # Now nearly all the lags should be within the confidence interval
    plot_acf(df_diff['Close'], lags=50) # This is the statsmodels version of the ACF plot for validation
    plt.show()




        