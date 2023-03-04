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
data = tic.history(interval='1d', period='2mo')
close = data[['Close']]
close = close.asfreq('d').ffill()
# print(close['Close'])
close.plot()
plt.show()

ssn = seasonal_decompose(close)
ssn.plot()
# print(ssn.seasonal)
plt.show()

# print(data.columns)
# close.plot()
# plt.show()

check_adfuller(close)

diff1 = close['Close'] - close['Close'].shift(1)

diff1 = diff1.dropna()
# diff1.plot()
# print(diff1)
# print(type(diff1))

check_adfuller(diff1)

# acf = plot_acf(close.dropna())
# pacf = plot_pacf(close.dropna(), method='ywmle')
acf = plot_acf(diff1.dropna())
pacf = plot_pacf(diff1.dropna(), method='ywmle')
plt.show()


# index = int(len(close)*9/10)
# train = close[:index].dropna()
# test = close[index:].dropna()

index = int(len(diff1)*9/10)
# train = diff1[:index].dropna()
# test = diff1[index:].dropna()
train = diff1[:index]
test = diff1[index:]

############################################ train model ##############################################################

p = 0
d = 1
q = 1

P = 1
D = 1
Q = 0
S = 7

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
# model = ARIMA(train, order=(p,d,q), seasonal_order=(P,D,Q,S)) # for backtest
model = ARIMA(close, order=(p,d,q), seasonal_order=(P,D,Q,S)) # for forecast
result = model.fit()
############################################# export trained model #######################################################

# pickle.dump(result, open('arima_trained_backtest.pkl', 'wb'))
# pickle.dump(result, open('arima_trained_forecast.pkl', 'wb'))
mean = close['Close'].mean()
sd = close['Close'].std()
lenn = len(close)
txt = f"{mean},{sd},{lenn}"
pickle.dump(txt, open('arima_data.pkl', 'wb'))

################################################ backtest ################################################################

# closee = pd.DataFrame(close)
# closee['forecast'] = result.forecast(len(test))
# diff = pd.DataFrame(diff1)
# diff['forecast'] = result.forecast(len(test))

# closee.plot()
# diff.plot()
# print(diff)
# plt.show()
# print(closee)


# diff['compute'] = diff['Close']
# diff['compute'].iloc[index:] = diff['forecast'].dropna()
# diff['cumsum'] = diff['compute'].cumsum()
# print(diff)

# close['forecast'] = close['Close'].iloc[0] + diff['cumsum']
# print(close)

# close.iloc[index:].plot()
# plt.show()

################################################ test forecast ################################################################

closee = pd.DataFrame(close)
# x = result.forecast(5)
# print(x)
# print(type(x))
res = pd.DataFrame(result.forecast(10))
res.columns = ['Close']
print(res)
ret = pd.concat([closee,res])
print(ret)

# closee[['forecast']] = pd.concat([closee,res])
# closee['Close'] = pd.concat([closee[['Close']],x])
# closee['forecast'] = res

ret.plot()
plt.show()

################################################ evaluate model ######################################################

# mae = mean_absolute_error(test, closee['forecast'].dropna())
# mape = mean_absolute_percentage_error(test, closee['forecast'].dropna())
# mse = mean_squared_error(test, closee['forecast'].dropna())
# mae = mean_absolute_error(test, diff['forecast'].dropna())
# mape = mean_absolute_percentage_error(test, diff['forecast'].dropna())
# mse = mean_squared_error(test, diff['forecast'].dropna())
# mae = mean_absolute_error(close['Close'].iloc[index:], close['forecast'].iloc[index:])
# mape = mean_absolute_percentage_error(close['Close'].iloc[index:], close['forecast'].iloc[index:])
# mse = mean_squared_error(close['Close'].iloc[index:], close['forecast'].iloc[index:])
# rmse = np.sqrt(mse)

# print(p,d,q,P,D,Q,S)
# print(f'mae = {mae}')
# print(f'mape = {mape}')
# print(f'mse = {mse}')
# print(f'rmse = {rmse}')

###################### best ################################
# 1d -> 2mo -> 0 1 1 1 1 0 7
#   mae = 0.1746242515811061, mape = 0.005586975989057016, mse = 0.08214207635260742, rmse = 0.2866043899744165