import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import getFamaFrenchFactors as gff
import statsmodels.api as sm
import seaborn as sns
from datetime import datetime
dateparse = lambda x: datetime.strptime(x, '%Y:%m:%d')

# Download Tesla stock data for 1000 days
start_date = datetime(2021, 1, 1)
end_date = datetime(2022, 1, 1)
ticker = "TSLA"
df = yf.download(ticker, start=start_date, end=end_date)
df_sp500 = yf.download('SPY',start_date,end_date)


# ads_data = pd.read_excel("ads_slice.xlsx", parse_dates=['Date'], date_parser=dateparse)
# print(ads_data.head(5))
# # stock_data = yf.download(ticker, start, end)
# new_ads_data = ads_data[['Date', 'ADS_INDEX_031623']].set_index('Date')

# new_ads_data.tz_localize(None)
# new_ads_data.reset_index(inplace=True)
# new_ads_data['Date'] = pd.to_datetime(new_ads_data['Date']).dt.date

# print(tmp_x)
# X = tmp_x.reset_index(drop=True)
# X = sm.add_constant(X)

# Calculate On Balance Volume (OBV)
df['daily_return'] = df['Adj Close'].pct_change()
df['direction'] = np.where(df['daily_return'] >= 0, 1, -1)
df['direction'][0] = 0
df['vol_adjusted'] = df['Volume'] * df['direction']
df['OBV'] = df['vol_adjusted'].cumsum()
df['SP500_ADJ_CLOSE'] = df_sp500['Adj Close']

# Calculate Fama French 3 factors
ff_data = gff.famaFrench3Factor(frequency='m') 


ff_data.rename(columns={"date_ff_factors": 'Date'}, inplace=True)
ff_data.set_index('Date',inplace=True)
ff_data = ff_data.resample('D').interpolate()
ff_data.tz_localize(None)
ff_data.reset_index(inplace=True)
df.reset_index(inplace=True)

df['Date'] = pd.to_datetime(df['Date']).dt.date
ff_data['Date'] = pd.to_datetime(ff_data['Date']).dt.date

# Merge data together to create dataframe
df = ff_data.merge(df,on='Date')

# df = new_ads_data.merge(df, on="Date")


df['Fama_French_Mkt_RF'] = ff_data['Mkt-RF']
df['Fama_French_SMB'] = ff_data['SMB']
df['Fama_French_HML'] = ff_data['HML']

# df.head()
# # Create dataframe for PCA
closing_price = df['Adj Close']

df_pca = pd.DataFrame({'Closing Price 1': closing_price.shift(1), 
                       'Closing Price 2': closing_price.shift(2), 
                       'Closing Price 3': closing_price.shift(3), 
                       'OBV': df['OBV'].shift(1), 
                       'Mkt-RF': df['Mkt-RF'].shift(1), 
                       'SMB': df['SMB'].shift(1), 
                       'HML': df['HML'].shift(1)})
#                         'SP500_ADJ_CLOSE': df['ADS_INDEX_031623'].shift(1)})

# Dropping null values
df_pca.dropna(inplace=True)

print(df_pca.head(5))

# Applying PCA
scaler = StandardScaler()
pca = PCA(n_components=3)

# Fit the scaler and PCA on the data
X = scaler.fit_transform(df_pca)
X = sm.add_constant(X)
pca.fit(X)

# Extract the top 3 principal components
top_3_pca = pca.transform(X)[:, :3]

# Splitting data into training and testing sets
train_size = int(len(top_3_pca) * 0.7)
train_data, test_data = top_3_pca[:train_size], top_3_pca[train_size:]

train_y, test_y = closing_price.values[3:train_size+3], closing_price.values[train_size+3:]

# Training the linear regression model
reg = LinearRegression()
reg.fit(train_data, train_y)

# Predicting on test data
test_pred = reg.predict(test_data)


test_y = test_y.reshape(-1, 1)
test_pred = test_pred.reshape(-1, 1)

# Calculating regression metrics
rmse = np.sqrt(mean_squared_error(test_y, test_pred))
r2 = r2_score(test_y, test_pred)

print("RMSE: ", rmse)
print("R Squared: ", r2)

pred_all = reg.predict(top_3_pca)
df_plot = df[['Date','Adj Close']]
df_plot = df_plot.iloc[3:]
df_plot['Predicted Close'] = pred_all
df_plot['Diff Close'] = df_plot['Adj Close'] - df_plot['Predicted Close']

sum(df_plot['Diff Close'])

sns.lineplot(data=df_plot[['Adj Close','Predicted Close']])
