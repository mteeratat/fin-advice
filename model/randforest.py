import matplotlib.pyplot as plt
import pandas as pd
from indicator.highlow_graph import highlow
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import yfinance as yf

def test():
    
    # data = pd.read_csv('test_data\ptt.bk.csv')
    # data = price.copy()

    tic = yf.Ticker('ptt.bk')
    data = tic.history(interval='1d', period='1y')
    # print(data)

    X = data[['Open','Close','High','Low','Volume']].shift(-1).dropna()
    # X = data[['Close']].shift(-1).dropna()
    y = data[['Close']].iloc[:-1]
    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,)

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

    # rf = RandomForestClassifier()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    pred = pd.Series(y_pred)
    pred.index = y_test.index

    # print(y_test)
    # print(pred)

    y_test.plot()
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
    
    # crit.plot()
    # plt.show()
    
    # return crit

    mae = mean_absolute_error(y_test, pred)
    mape = mean_absolute_percentage_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)

    print(f'mae = {mae}')
    print(f'mape = {mape}')
    print(f'mse = {mse}')
    print(f'rmse = {rmse}')