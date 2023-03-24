import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import getFamaFrenchFactors as gff
import matplotlib.pyplot as plt


ticker = 'TSLA'
start = '2022-3-20'
end = '2023-3-20'

def OBV(close_series,volume_series,days):
    OBV_LABEL ="OBV"
    day_index = -1 * days
    volume_data = pd.DataFrame(volume_series)[day_index:]
    volume_data["Adj Close"] = pd.DataFrame(close_series)[day_index:]

    for i, (index, row) in enumerate(volume_data.iterrows()):
        if i > 0:
            # Not the first row, so adjust OBV based on the price action
            prev_obv = volume_data.loc[volume_data.index[i - 1], OBV_LABEL]
            if row["Adj Close"] > volume_data.loc[volume_data.index[i - 1], "Adj Close"]:
                # Up day
                obv = prev_obv + row["Adj Close"]
            elif row["Adj Close"] < volume_data.loc[volume_data.index[i - 1], "Adj Close"]:
                # Down day
                obv = prev_obv - row["Adj Close"]
            else:
                # Equals, so keep the previous OBV value
                obv = prev_obv
        else:
            # First row, set prev_obv to zero
            obv = row["Adj Close"]
            prev_obv = 0

        # Assign the obv value to the correct row
        volume_data.at[index, OBV_LABEL] = obv
    return volume_data


stock_data = yf.download(ticker, start, end)
# print(stock_data)
obv_data = OBV(stock_data['Adj Close'],stock_data['Volume'], len(stock_data))
merge_data = stock_data.merge(obv_data,on="Date")

X = merge_data[['Open', 'OBV']][:len(merge_data) - 1].reset_index(drop=True)
Y = merge_data['Open'][1:].reset_index(drop=True)
X = sm.add_constant(X)
print(X)
print(Y)
ff_model = sm.OLS(Y, X).fit()
print(ff_model.summary())
#
# fig, ax = plt.subplots()
# fig = sm.graphics.plot_fit(ff_model, 0, ax=ax)
# ax.set_ylabel("X")
# ax.set_xlabel("Y")
# ax.set_title("Linear Regression")
# plt.plot();
# ff_model.
# print(merge_data)
