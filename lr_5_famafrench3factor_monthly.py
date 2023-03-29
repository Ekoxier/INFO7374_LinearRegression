import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import getFamaFrenchFactors as gff

# http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
ticker = 'TSLA'
start = '2022-3-20'
end = '2023-3-20'

stock_data = yf.download(ticker, start, end)

ff3_monthly = gff.famaFrench3Factor(frequency='m')
ff3_monthly.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
ff3_monthly.set_index('Date', inplace=True)

stock_returns = stock_data['Adj Close'].resample('M').last().pct_change().dropna()
stock_returns.name = "Month_Rtn"
ff_data = ff3_monthly.merge(stock_returns, on='Date')

X = ff_data[['Mkt-RF', 'SMB', 'HML']].reset_index(drop=True)
# Mkt-RF: total market portfolio return at time t - risk-free rate of return at time t
# = excess return on the market portfolio
Y = (ff_data['Month_Rtn'] - ff_data['RF']).reset_index(drop=True) # Expected excess return
X = sm.add_constant(X)
print(X)
print(Y)
ff_model = sm.OLS(Y, X).fit()
print(ff_model.summary())
intercept, b1, b2, b3 = ff_model.params

rf = ff_data['RF'].mean()
market_premium = ff3_monthly['Mkt-RF'].mean()
size_premium = ff3_monthly['SMB'].mean()
value_premium = ff3_monthly['HML'].mean()

# expected_monthly_return = rf + b1 * market_premium + b2 * size_premium + b3 * value_premium
# expected_yearly_return = expected_monthly_return * 12
# print("Expected yearly return: " + str(expected_yearly_return))