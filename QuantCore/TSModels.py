import os
import pandas as pd
import numpy as np
import QuantCoreStats as qcs  # Your PyBind11 module
import DataProcessor as dp

class ARIMAGARCHModel:
    def __init__(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        self.data_folder = data_dir
        self.TS_Models = {}
        self.data_processor = dp.DataProcessor()
        self.recent_returns = self.data_processor.crypto_live_data['BTCUSDT']

    def run(self):
        for file in os.listdir(self.data_folder):
            if file.endswith(".csv"):
                ticker = os.path.splitext(file)[0]
                file_path = os.path.join(self.data_folder, file)
                try:
                    ts_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                except Exception as e:
                    print(f"Failed to load {ticker}: {e}")
                    continue

                if "Target" in ts_data.columns:
                    returns = ts_data["Target"].values
                elif "LogReturns" in ts_data.columns:
                    returns = ts_data["LogReturns"].values
                else:
                    print(f"No 'Target' or 'LogReturns' column found for {ticker}. Skipping.")
                    continue

                try:
                    model = qcs.grid_search_arima_garch(returns)
                except Exception as e:
                    print(f"Failed to build model for {ticker}: {e}")
                    continue

                try:
                    prediction = qcs.predict_next_return(model, self.recent_returns)
                    certainty = model.get("certainty", None)
                except Exception as e:
                    print(f"Failed to predict for {ticker}: {e}")
                    continue

                self.TS_Models[ticker] = {"Prediction": prediction, "Certainty": certainty}
        return self.TS_Models

if __name__ == "__main__":
    arb_model = ARIMAGARCHModel()
    results = arb_model.run()

    # Extract the predictions and certainties which are beneficial
    for ticker, data in results.items():
        prediction = data["Prediction"]
        certainty = data["Certainty"]
        if certainty > 0.5 and np.exp(prediction) - 1 > 0:
            print(f"Ticker: {ticker}, Prediction: {(np.exp(prediction) - 1) * 100}% Return, Certainty: {certainty}")

    print("All models have been run.")
