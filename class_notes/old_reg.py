# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:07:20 2019
OLS REGRESSION
@author: yizhen zhao
"""
import numpy as np
import scipy.stats as ss
import numpy as np
import scipy.stats as ss
import pandas as pd
from pandas_datareader import DataReader
from datetime import datetime

import yfinance as yf


start_date = datetime(2020,1,1)
end_date = datetime(2020,12,31)
MSFT = yf.download('MSFT',start_date ,end_date)
Y = np.diff(np.log(MSFT['Adj Close'].values))
T = Y.shape[0];

SPY = yf.download('SPY',start_date ,end_date)
F1 = np.diff(np.log(SPY['Adj Close'].values))
QQQ = yf.download('QQQ',start_date ,end_date)
F2 = np.diff(np.log(QQQ['Adj Close'].values))

'Add Constant to X'
X = np.column_stack([np.ones((T,1)), F1, F2])
N = X.shape[1]

'OLS REGRESSION STARTS'
'Linear Regression of Y: T x 1 on'
'Regressors X: T x N'
invXX = np.linalg.inv(X.transpose()@X)
'OLS estimates for coefficients: X x 1'
beta_hat = invXX@X.transpose()@Y
'Predictive value of Y using OLS'
y_hat = X@beta_hat
'Residuals from OLS'
residuals = Y - y_hat
'Variance of residuals'
sigma2 = (1/T)*residuals.transpose()@residuals
'standard deviation of Y or residuals'
sigma = np.sqrt(sigma2)

'variance-covariance matrix of beta_hat'
varcov_beta_hat = (sigma2)*invXX
std_beta_hat = np.sqrt(T*np.diag(varcov_beta_hat))

'Calculate R-square'
R_square = 1- (residuals.transpose()@residuals)/(T*np.var(Y))
adj_R_square = 1-(1-R_square)*(T-1)/(T-N)

'Test Each Coefficient: beta_i'
'Null Hypothesis: beta_i = 0'
t_stat = (beta_hat.transpose()-0)/std_beta_hat
p_val_t = 1-ss.norm.cdf(t_stat)

'Test of Joint Significance of Model'
F_stat = (beta_hat.transpose()@np.linalg.inv(varcov_beta_hat)@beta_hat/N)/\
         (residuals.transpose()@residuals/(T-N))

p_val_F = 1-ss.f.cdf(F_stat,N-1,T-N)


REPORT = np.column_stack([beta_hat, t_stat,p_val_t])
print('Regression Statistics')
print('------------------------\n')
print(' REGRESSION STATISTICS  \n') 
print('------------------------\n')
print('beta             t_stat            p_val\n')
print(REPORT)
print('\n Joint significance of all coefficients\n',[F_stat,p_val_F])
print('R-Square is       \n',R_square)
print('Adjusted R Square \n',adj_R_square)
print('Standard Error    \n',sigma)
print('Observations      \n',T) 
print('-------------------------\n')