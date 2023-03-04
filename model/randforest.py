import matplotlib.pyplot as plt
import pandas as pd
from indicator.highlow_graph import highlow
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
def test(price):
    
    # data = pd.read_csv('test_data\ptt.bk.csv')
    data = price.copy()
    # print(data)
    # data.index = data['Date']
    # data.plot()

    temp = highlow(data)
    temp = temp.replace(np.nan, 99)
    crit = temp[['crit']]
    temp = temp.drop(columns=['crit'])
    # print(temp)
    # print(crit)

    # X = temp[['Close']].where(temp['crit'] == 0).dropna()
    # y = temp[['Close']].where(temp['crit'] == 1).dropna()
    X_train = temp.iloc[:int(len(data)*9/10)]
    X_test = temp.iloc[int(len(data)*9/10):]
    y_train = crit.iloc[:int(len(data)*9/10)]
    y_test = crit.iloc[int(len(data)*9/10):] 
    # print(X_train)
    # print(y_train['crit'])

    # rf = RandomForestClassifier()
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train['crit'])

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
    
    crit.plot()
    plt.show()
    
    return crit