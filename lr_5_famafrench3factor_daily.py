from datetime import datetime
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
stock_returns = stock_data['Adj Close'].pct_change().dropna()
stock_returns.name = "Daily_Rtn"

# Process famafrench3factors
ff3_factors = pd.read_csv(file_name, skiprows=rows_to_skip, parse_dates=['Date'], date_parser=dateparse)
ff3_data = ff3_factors.merge(stock_returns, on='Date')

# print(ff3_data)
# Mkt-RF: total market portfolio return at time t
# - risk-free rate of return at time t
# = excess return on the market portfolio
print(ff3_data)
X = ff3_data[['Mkt-RF', 'SMB', 'HML']][:-1].reset_index(drop=True)
# Expected excess return
Y = (ff3_data['Daily_Rtn'] - ff3_data['RF'])[1:].reset_index(drop=True)

X2 = ff3_data[['Mkt-RF', 'SMB', 'HML']].reset_index(drop=True)
# Expected excess return
Y2 = (ff3_data['Daily_Rtn'] - ff3_data['RF']).reset_index(drop=True)
X2 = sm.add_constant(X2)
ff_model = sm.OLS(Y2, X2).fit()
print(ff_model.summary())

# print(X)
# print(Y)
# X = sm.add_constant(X)
# ff_model = sm.OLS(Y, X).fit()
# print(ff_model.summary())