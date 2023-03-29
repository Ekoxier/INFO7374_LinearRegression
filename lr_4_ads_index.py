import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import getFamaFrenchFactors as gff
import matplotlib.pyplot as plt

from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%Y:%m:%d')

ticker = 'TSLA'
start = '2022-3-10'
end = '2023-3-10'
ads_data = pd.read_excel("ads_slice.xlsx", parse_dates=['Date'], date_parser=dateparse)
stock_data = yf.download(ticker, start, end)
new_ads_data = ads_data[['Date', 'ADS_INDEX_031623']].set_index('Date')
tmp_x = stock_data.merge(new_ads_data, on="Date")[['Open','ADS_INDEX_031623']][:-1]
print(tmp_x)
X = tmp_x.reset_index(drop=True)
X = sm.add_constant(X)
print(stock_data['Open'][1:])
Y = stock_data['Open'][1:].reset_index(drop=True)
ff_model = sm.OLS(Y, X).fit()
print(ff_model.summary())

# fig, ax = plt.subplots()
# fig = sm.graphics.plot_fit(ff_model, 0, ax=ax)
# ax.set_ylabel("X")
# ax.set_xlabel("Y")
# ax.set_title("Linear Regression")
# plt.plot();
# ff_model.
# print(merge_data)
