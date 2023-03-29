from datetime import datetime

import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import getFamaFrenchFactors as gff

ticker = 'TSLA'
start = '2022-3-20'
end = '2023-3-20'
stock_data = yf.download(ticker, start, end)

stock_returns = stock_data['Adj Close'].pct_change().dropna()
stock_returns.name = "Asset"
file_name = "F-F_Research_Data_Factors_daily.CSV"
rows_to_skip = 4
dateparse = lambda x: datetime.strptime(x, '%Y%m%d')

ff3_factors = pd.read_csv(file_name, skiprows=rows_to_skip, parse_dates=['Date'], date_parser=dateparse)
ff3_data = ff3_factors.merge(stock_returns, on='Date')
ff3_data['Asset-RF'] = ff3_data['Asset'] - ff3_data['RF']
# risk-free rate of return
X = ff3_data['Mkt-RF']
Y = ff3_data['Asset-RF']
#
# X = ff3_data['Mkt-RF'][:-1].reset_index(drop=True)
# Y = ff3_data['Asset-RF'][1:].reset_index(drop=True)

X = sm.add_constant(X)
capm_model = sm.OLS(Y, X)
result = capm_model.fit()
print(result.summary())