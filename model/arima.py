import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
from pandas.plotting import lag_plot
import pmdarima as pmd

#freq: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

def check_adfuller(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    
    if result[1] <= 0.05:
        print("Reject the null hypothesis. Data is stationary")
    else:
        print("Do not reject the null hypothesis. Data is not stationary ")

# tic = yf.Ticker('^set.bk')
tic = yf.Ticker('ptt.bk')
data = tic.history(interval='1d', period='3y')
close = data[['Close']]
print(close['Close'])
close = close.iloc[4:]
print(close['Close'])
close = close.asfreq('b').ffill()
print(close['Close'])
lag_plot(close['Close'], lag=1)
fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10,5))
# close.plot()
ax1.plot(close)
# plt.show()

ssn = seasonal_decompose(close)
ssn.plot()
s = ssn.seasonal
print(s[s>4])
# print(ssn.seasonal)
# plt.show()

# print(data.columns)
# close.plot()
# plt.show()

check_adfuller(close)

diff1 = close['Close'] - close['Close'].shift(1)

diff1 = diff1.dropna()
# diff1.plot()
ax2.plot(diff1)
# print(diff1)
# print(type(diff1))

check_adfuller(diff1)

acf = plot_acf(close.dropna())
pacf = plot_pacf(close.dropna(), method='ywmle')
acf = plot_acf(diff1.dropna())
pacf = plot_pacf(diff1.dropna(), method='ywmle')
# plt.show()


index = int(len(close)-10)
# index = int(len(close)*9/10)
train = close[:index].dropna()
test = close[index:].dropna()

# index = int(len(diff1)*9/10)
# train = diff1[:index].dropna()
# test = diff1[index:].dropna()
# train = diff1[:index]
# test = diff1[index:]

############################################ train model ##############################################################

#advanc 1b 3y 2 1 1 2 1 1 5 : mae 2.144 mape 0.010 mse 9.027 rmse 3.004 corr 0.714  
#advanc 5b 3y 3 1 3 2 1 2 5 : mae 2.880 mape 0.014 mse 12.302 rmse 3.507 corr 0.758  
#advanc M 3y 5 1 1 1 1 0 12 : mae 3.854 mape 0.019 mse 22.346 rmse 4.727 corr 0.817
#cpn 1b 3y 2 1 2 1 1 2 5 : mae 2.144 mape 0.010 mse 9.027 rmse 3.004 corr 0.714  
#cpn 5b 3y 4 1 4 1 1 1 5 : mae 1.566 mape 0.022 mse 3.486 rmse 1.867 corr -0.200 
#cpn M 3y 2 1 2 0 1 0 12 : mae 4.326 mape 0.062 mse 22.914 rmse 4.786 corr -0.322
#ptt 1b 3y 0 1 1 0 1 1 5 : mae 0.487 mape 0.015 mse 0.314 rmse 0.560 corr 0.271  
#ptt 5b 3y 2 1 2 1 1 1 5 : mae 0.818 mape 0.025 mse 0.889 rmse 0.943 corr -0.431 
#ptt M 3y 2 0 2 1 0 1 12 : mae 1.349 mape 0.041 mse 2.095 rmse 1.447 corr 0.543

##################

p = 0
d = 1
q = 1

P = 0
D = 1
Q = 1
S = 5

# min_aic = 9999
# for i in range(0,3):
#     for j in range(0,3):
#         model = ARIMA(train, order=(i,d,j))
#         result = model.fit()
# # print(result.summary())
#         print(f"{result.aic}, {i}, {j}")
#         if min_aic >= result.aic:
#             min_aic = result.aic
#             p = i
#             q = j
# print(p,d,q)
# model = ARIMA(train, order=(p,d,q)) # for backtest
# model = ARIMA(close, order=(p,d,q)) # for forecast
# result = model.fit()
# result = model.fit(method='innovations_mle')

# min_aic = 9999
# for i in range(0,3):
#     for j in range(0,3):
#         for k in range(0,3):
#             for l in range(0,3):
#                 model = ARIMA(train, order=(i,d,j), seasonal_order=(k,D,l,S))
#                 result = model.fit()
#                 print(f"{result.aic}, {i}, {j}, {k}, {l}")
#                 if min_aic >= result.aic:
#                     min_aic = result.aic
#                     p = i
#                     q = j
#                     P = k
#                     Q = l
print(p,d,q,P,D,Q,S)
model = ARIMA(train, order=(p,d,q), seasonal_order=(P,D,Q,S)) # for backtest
# model = ARIMA(close, order=(p,d,q), seasonal_order=(P,D,Q,S)) # for forecast
result = model.fit()

# model = pmd.auto_arima(train, start_p=1, start_q=1,
#                          test='adf',
#                          max_p=3, max_q=3, 
#                          m=7, #12 is the frequncy of the cycle
#                          start_P=1, 
#                          seasonal=True, #set to seasonal
#                          #d=1, 
#                          D=1, #order of the seasonal differencing
#                          trace=True,
#                          error_action='ignore',  
#                          suppress_warnings=True, 
#                          stepwise=True)
# # result = model.fit()
# res = model.predict(len(test))
# fig3, (ax5,ax6) = plt.subplots(2, 1, figsize=(10,5))
# ax5.plot(res)
# ax6.plot(test)
# plt.show()
# print(res)
############################################# export trained model #######################################################

# pickle.dump(result, open('model\\1mo_ADVANC_arima.pkl', 'wb'))
# pickle.dump(result, open('model\\1wk_ADVANC_arima.pkl', 'wb'))
# pickle.dump(result, open('model\\1d_ADVANC_arima.pkl', 'wb'))
# pickle.dump(result, open('model\\1mo_CPN_arima.pkl', 'wb'))
# pickle.dump(result, open('model\\1wk_CPN_arima.pkl', 'wb'))
# pickle.dump(result, open('model\\1d_CPN_arima.pkl', 'wb'))
# pickle.dump(result, open('model\\1mo_PTT_arima.pkl', 'wb'))
# pickle.dump(result, open('model\\1wk_PTT_arima.pkl', 'wb'))
# pickle.dump(result, open('model\\1d_PTT_arima.pkl', 'wb'))

# mean = close['Close'].mean()
# sd = close['Close'].std()
# lenn = len(close)
# txt = f"{mean},{sd},{lenn}"
# pickle.dump(txt, open('arima_data.pkl', 'wb'))

################################################ backtest ################################################################

closee = close.copy()
closee['forecast'] = result.forecast(len(test))
# diff = pd.DataFrame(diff1)
# diff['forecast'] = result.forecast(len(test))

closee.plot()
# print(closee)
# fig2, (ax3,ax4) = plt.subplots(2, 1, figsize=(10,5))
# diff.plot()
# ax3.plot(closee)
# print(diff)
# plt.show()


# diff['compute'] = diff['Close']
# diff['compute'].iloc[index:] = diff['forecast'].dropna()
# diff['cumsum'] = diff['compute'].cumsum()
# print(diff)

# close['forecast'] = close['Close'].iloc[0] + diff['cumsum']
# print(close)

# close.iloc[index:].plot()
# ax4.plot(closee)
# plt.show()

################################################ test forecast ################################################################

# closee = pd.DataFrame(close)
# x = result.forecast(5)
# print(x)
# print(type(x))
# res = pd.DataFrame(result.forecast(10))
# res.columns = ['Close']
# print(res)
# ret = pd.concat([closee,res])
# print(ret)

# closee[['forecast']] = pd.concat([closee,res])
# closee['Close'] = pd.concat([closee[['Close']],x])
# closee['forecast'] = res

# ret.plot()
# plt.show()

################################################ evaluate model ######################################################

# mae = mean_absolute_error(test, closee['forecast'].dropna())
# mape = mean_absolute_percentage_error(test, closee['forecast'].dropna())
# mse = mean_squared_error(test, closee['forecast'].dropna())
# mae = mean_absolute_error(test, diff['forecast'].dropna())
# mape = mean_absolute_percentage_error(test, diff['forecast'].dropna())
# mse = mean_squared_error(test, diff['forecast'].dropna())
mae = mean_absolute_error(close['Close'].iloc[index:], closee['forecast'].iloc[index:])
mape = mean_absolute_percentage_error(close['Close'].iloc[index:], closee['forecast'].iloc[index:])
mse = mean_squared_error(close['Close'].iloc[index:], closee['forecast'].iloc[index:])
rmse = np.sqrt(mse)
corr = close['Close'].iloc[index:].corr(closee['forecast'].iloc[index:])

print(p,d,q,P,D,Q,S)
print(f'mae = {mae}')
print(f'mape = {mape}')
print(f'mse = {mse}')
print(f'rmse = {rmse}')
print(f'corr = {corr}')

plt.show()

###################### best ################################
# 1d -> 2mo -> 0 1 1 1 1 0 7
#   mae = 0.1746242515811061, mape = 0.005586975989057016, mse = 0.08214207635260742, rmse = 0.2866043899744165

########################################################## callable function ##########################################

def train_arima(close):
    price = close.copy()