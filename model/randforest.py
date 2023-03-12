import matplotlib.pyplot as plt
import pandas as pd
from indicator.highlow_graph import highlow
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import yfinance as yf
from datetime import timedelta
import pickle

def test():
    
    # data = pd.read_csv('test_data\ptt.bk.csv')
    # data = price.copy()

    tic = yf.Ticker('advanc.bk')
    data = tic.history(interval='1d', period='3y')
    # data = data.ffill()
    # print(data)

    # X = data[['Open','Close','High','Low','Volume']].iloc[:-5]
    X = data[['Close']].iloc[:-5]
    # X = data[['Close']].shift(-1).dropna()
    y = data[['Close']].shift(5).dropna()
    # print(X)
    # print(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    X_train = X.iloc[:int(len(data)*9/10)]
    X_test = X.iloc[int(len(data)*9/10):]
    y_train = y.iloc[:int(len(data)*9/10)]
    y_test = y.iloc[int(len(data)*9/10):] 

    # data.index = data['Date']
    # data.plot()

    # temp = highlow(data)
    # temp = temp.replace(np.nan, 99)
    # crit = temp[['crit']]
    # temp = temp.drop(columns=['crit'])
    # print(temp)
    # print(crit)

    # X = temp[['Close']].where(temp['crit'] == 0).dropna()
    # y = temp[['Close']].where(temp['crit'] == 1).dropna()
    # X_train = temp.iloc[:int(len(data)*9/10)]
    # X_test = temp.iloc[int(len(data)*9/10):]
    # y_train = crit.iloc[:int(len(data)*9/10)]
    # y_test = crit.iloc[int(len(data)*9/10):] 
    # print(X_train)
    # print(y_train['crit'])

    ## backtest ##
    # rf = RandomForestClassifier()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    result = rf.fit(X_train, y_train)
    # print(result)
    # pickle.dump(result, open('model\\ADVANC_random_forest.pkl', 'wb'))

    # # print(X_test)
    # X_test.index = X_test.index
    # print(X_test)
    # y_pred = rf.predict(X_test)
    # pred = pd.DataFrame(y_pred)
    # # print(pred)
    # pred.columns = ['Forecast']
    # pred.index = y_test.index

    # print(y_test)
    # print(pred)

    # y_test.plot()
    # pred.plot()
    # plt.show()

    # ## forecast ##
    # X_test = data[['Open','Close','High','Low','Volume']].iloc[-5:]
    X_test = data[['Close']].iloc[-5:]
    print(X_test)
    y_pred = rf.predict(X_test)
    pred = pd.Series(y_pred)
    # print(pred)
    # # pred.columns = ['Close']
    fut = []
    for i in range(5):
        tmr = y_test.index[-1] + timedelta(days=3+i)
        fut.append(tmr)
    # print(fut)
    # print(tmr)
    # print(type(tmr))
    # ind = y_test.index[-5:]
    # print(type(ind))
    # print(type(ind[0]))
    # print(ind.append(pd.DatetimeIndex([tmr])))

    # pred.index = ind[1:].append(pd.DatetimeIndex([tmr]))
    pred.index = fut
    print(y_test)
    print(pred)

    y_test.plot()
    # plt.show()
    pred.plot()
    plt.show()

    # result = rf.predict(X_test)
    # print(X_test)
    # print(result)
    # print(y_test)
    # X_train.plot()
    # X_test.plot()
    # plt.show()

    # y_train.plot()
    # res = pd.Series(result)
    # res.plot()
    # y_test.plot()
    
    # # crit.plot()
    # plt.show()
    
    # return crit

    # mae = mean_absolute_error(y_test, pred)
    # mape = mean_absolute_percentage_error(y_test, pred)
    # mse = mean_squared_error(y_test, pred)
    # rmse = np.sqrt(mse)
    # corr = y_test['Close'].corr(pred['Forecast'])

    # print(f'mae = {mae}')
    # print(f'mape = {mape}')
    # print(f'mse = {mse}')
    # print(f'rmse = {rmse}')
    # print(f'corr = {corr}')

    # mae = 0.02025257644653351
    # mape = 0.0005701874408488805
    # mse = 0.0026427504663121626
    # rmse = 0.051407688785940986