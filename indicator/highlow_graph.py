import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def highlow(price):
    temp = price[['Close']]
    temp['crit'] = pd.Series(data=[1 for i in range(temp.shape[0])], index=temp.index)
    days = 10
    high_dif = temp.copy()
    low_dif = temp.copy()
    for i in range(days):
        high_dif['fdiff'] = temp['Close'].diff(i)
        high_dif['bdiff'] = temp['Close'].diff(-i)
        # temp[f'high_fdiff{i}'] = high_dif['fdiff'].copy()
        # temp[f'high_bdiff{i}'] = high_dif['bdiff'].copy()

        low_dif['fdiff'] = temp['Close'].diff(i)
        low_dif['bdiff'] = temp['Close'].diff(-i)
        # temp[f'low_fdiff{i}'] = low_dif['fdiff'].copy()
        # temp[f'low_bdiff{i}'] = low_dif['bdiff'].copy()

        high_dif['crit'].loc[high_dif['fdiff']<0] = 0
        high_dif['crit'].loc[high_dif['bdiff']<0] = 0

        low_dif['crit'].loc[low_dif['fdiff']>0] = 0
        low_dif['crit'].loc[low_dif['bdiff']>0] = 0

    temp['crit'] = high_dif['crit'] + low_dif['crit']
    temp['highs'] = high_dif['crit']
    temp['lows'] = low_dif['crit']

    # temp['highs'].plot()
    # temp['lows'].plot()
    # plt.show()

    # price = price.sort_values(by=['Close'])
    # temp = price[['Close']].diff(5)
    # temp['crit'] = np.where(temp.iloc[:, 0] < 0.3, 1.0, 0.0)
    # temp = temp.sort_values(by=['Date'])
    # price = price.sort_values(by=['Date'])
    # price['crit'] = temp['crit']

    return temp