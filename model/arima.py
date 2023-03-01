import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
data = tic.history(interval='1wk', period='2y')
close = data[['Close']]
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
train = diff1[:index].dropna()
test = diff1[index:].dropna()

############################################ train model ##############################################################

p = 3
d = 1
q = 5

# min_aic = 9999
# for i in range(0,6):
#     for j in range(0,6):
#         model = ARIMA(train, order=(i,d,j))
#         result = model.fit()
# # print(result.summary())
#         print(f"{result.aic}, {i}, {j}")
#         if min_aic >= result.aic:
#             min_aic = result.aic
#             p = i
#             q = j
print(p,d,q)
# model = ARIMA(train, order=(p,d,q)) # for backtest
model = ARIMA(close, order=(p,d,q)) # for forecast
result = model.fit()

############################################# export trained model #######################################################

# pickle.dump(result, open('arima_trained_backtest.pkl', 'wb'))
pickle.dump(result, open('arima_trained_forecast.pkl', 'wb'))
mean = close['Close'].mean()
sd = close['Close'].std()
lenn = len(close)
txt = f"{mean},{sd},{lenn}"
pickle.dump(txt, open('arima_data.pkl', 'wb'))

################################################ backtest ################################################################

# closee = pd.DataFrame(close)
# closee['forecast'] = result.forecast(len(test), alpha=0.05)
# diff = pd.DataFrame(diff1)
# diff['forecast'] = result.forecast(len(test), alpha=0.05)

# closee.plot()
# diff.plot()
# plt.show()
# print(closee)


# diff['compute'] = diff['Close']
# diff['compute'].iloc[index:] = diff['forecast'].dropna()
# diff['cumsum'] = diff['compute'].cumsum()
# print(diff)

# close['forecast'] = close['Close'].iloc[0] + diff['cumsum']
# print(close)

# close.plot()
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
# mse = mean_squared_error(test, closee['forecast'].dropna())
# mae = mean_absolute_error(test, diff['forecast'].dropna())
# mse = mean_squared_error(test, diff['forecast'].dropna())
# rmse = np.sqrt(mse)

# print(mae)
# print(mse)
# print(rmse)