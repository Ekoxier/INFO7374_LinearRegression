from datetime import datetime

import numpy as np
import yfinance as yf
import pandas as pd
import statsmodels.api as sm
ticker = 'TSLA'
start = '2022-3-20'
end = '2023-3-20'
file_name = "F-F_Research_Data_Factors_daily.CSV"
rows_to_skip = 4
dateparse = lambda x: datetime.strptime(x, '%Y%m%d')
# Process stock return
stock_data = yf.download(ticker, start, end)
# stock_returns = stock_data['Adj Close'].pct_change().dropna()
# stock_returns.name = "Daily_Rtn"

# Process famafrench3factors
ff3_factors = pd.read_csv(file_name, skiprows=rows_to_skip, parse_dates=['Date'], date_parser=dateparse)
# ff3_data = ff3_factors.merge(stock_returns, on='Date')

ff3_factors['Mkt'] = ff3_factors['Mkt-RF'] + ff3_factors['RF']
tmp_X = ff3_factors[['Mkt', 'SMB', 'HML', 'Date']]
# Expected excess return
tmp_Y = (np.log(stock_data['Adj Close']) - np.log(stock_data['Adj Close'].shift(1)))
tmp_XY = tmp_X.merge(tmp_Y, on='Date')
X = tmp_XY[['Mkt', 'SMB', 'HML']][:-1].reset_index(drop=True)
Y = tmp_XY['Adj Close'][1:].reset_index(drop=True)
print(X)
print(Y)
X = sm.add_constant(X)
ff_model = sm.OLS(Y, X).fit()
print(ff_model.summary())