# LSTM_Security_Prediction
**Note: This project is purely for interest and research purposes. I am not responsible for any losses that occur as a result of using this GitHub Repository, please use at your own risk.**

Multi-Disciplinary Quantitative Trading Strategy invoking the usage of:
- Classic Time Series Modeling (ARIMA-GARCH Joint Model) `C++` with `PyBind11` for efficiency
- Long-Short-Term-Memory (LSTM) Model (Form of Recurrent Neural Network (RNN)) `tensorflow` with `metal` support as well as `cuda`
- Analysis of correlations between assets to build low-risk portfolios. 

## Hidden Files
To be able to fetch Cryptocurrency Historical Data, you are required to provide a Binance API Key and a Secret Key. These are available for free by simply switching your Binance account to a PRO account. 

Then, you need to set up the `secrets.json` file which will be located at `$PROJ_ROOT/secrets/secrets.json`. The file structure is shown below:

`secrets.json`

```json
    "BINANCE_API_KEY": "Enter Binance API Key Here", 
    "BINANCE_API_SECRET": "Enter Binance Secret Key Here",
    "Trading Frequency (Yearly/Monthly/Weekly/Daily/Hourly/Minutely)": "Daily",
    "Starting Date (YYYY-MM-DD)": "2015-01-01", 
    "Ending Date (YYYY-MM-DD)": "2025-04-01",
    "Base Currency": "USDT", 
    "Tickers of Interest": [
        "BTC", "ETH", "XRP", "SOL", "LTC", "DOGE", "BNB", 
        "ADA", "TRX", "LINK", "XLM", "FIL", "MATIC", "DOT",
        "ALGO", "ICP", "BCH", "UNI", "ETC", "VET", "EOS", 
        "AAVE", "NEAR", "ATOM", "FTM", "THETA", "XTZ", 
        "ZEC", "MANA", "SAND", "BAT", "CHZ", "HBAR", "LDO",
        "CRV", "KSM", "XEM", "ZRX", "QTUM", "DGB", "WAVES",
        "HNT", "XVG", "DASH", "ZIL", "NANO", "OMG", "REN",
        "1INCH", "SUSHI", "YFI", "COMP", "SNX", "LRC", "STMX",
        "FET", "STPT", "LEND", "BAND", "ENJ", "LPT", "RLC"
    ],
    "Window Size": 40
```

After this has ben set up, calling the DataProcessor Class initializer will automatically use the data you have provided in the secrets.json file to populate and compute the models for usage. I would recommend against using Hourly Data (Very noisy and remains non-stationary even after differencing operator applied) and Minutely Data (Same issue as Hourly Data but with additional extremely long runtime for fetching data using Binance API). 

