# Class for building LSTM models 

# Imports
import pandas as pd 
import numpy as np 
import datetime 
import os 
import DataProcessor as dp 
import QuantCoreStats as qcs  
import tensorflow as tf

class LSTM_Models:
    def __init__(self):
        # Run the data processor 
        self.data_processor = dp.DataProcessor()
    
    def windowed_df_to_numpy(self, windowed_df):
        df_as_numpy = windowed_df.to_numpy()
        dates = df_as_numpy[:, 0]

        middle_matrix = df_as_numpy[:, 1:-1]
        X = middle_matrix.reshape(len(dates), middle_matrix.shape[1], 1)

        Y = df_as_numpy[:, -1]

        return dates, X.astype(np.float32), Y.astype(np.float32)

    

