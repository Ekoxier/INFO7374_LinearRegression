import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import getFamaFrenchFactors as gff
import matplotlib.pyplot as plt


ticker = 'TSLA'
start = '2022-3-20'
end = '2023-3-20'

stock_data = yf.download(ticker, start, end)
tmp_x = stock_data.copy();
tmp_x['Open-1'] = stock_data['Open'].shift(1)
tmp_x['Open-2'] = stock_data['Open'].shift(2)
tmp_x['Open-3'] = stock_data['Open'].shift(3)
X = tmp_x[['Open-1','Open-2','Open-3']][3:].reset_index(drop=True)
# X = stock_data[['Open']][:len(stock_data) - 1].reset_index(drop=True)
Y = stock_data['Open'][3:].reset_index(drop=True)
X = sm.add_constant(X)
ff_model = sm.OLS(Y, X).fit()
print(ff_model.summary())
#
# fig, ax = plt.subplots()
# fig = sm.graphics.plot_fit(ff_model, 0, ax=ax)
# ax.set_ylabel("X")
# ax.set_xlabel("Y")
# ax.set_title("Linear Regression")
# plt.show();
# ff_model.
# print(merge_data)
