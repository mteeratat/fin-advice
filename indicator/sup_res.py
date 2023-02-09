import pandas as pd

def supres(price):
    temp = price[['Close']]
    temp['crit'] = pd.Series(data=[1 for i in range(temp.shape[0])], index=temp.index)
    days = 10
    dif = temp.copy()
    dif2 = temp.copy()
    for i in range(1, days):
        dif['fdiff'] = temp['Close'].diff(i)
        dif['bdiff'] = temp['Close'].diff(-i)
        # cut out -> use 'not' in loc
        dif['crit'].loc[dif['fdiff']<0] = 0
        dif['crit'].loc[dif['bdiff']<0] = 0
    for i in range(days-1):
        dif2['fdiff'] = temp['Close'].diff(i)
        dif2['bdiff'] = temp['Close'].diff(-i)
        # cut out -> use 'not' in loc
        dif2['crit'].loc[dif2['fdiff']>0] = 0
        dif2['crit'].loc[dif2['bdiff']>0] = 0

    temp['crit'] = dif['crit'] + dif2['crit']

    # price = price.sort_values(by=['Close'])
    # temp = price[['Close']].diff(5)
    # temp['crit'] = np.where(temp.iloc[:, 0] < 0.3, 1.0, 0.0)
    # temp = temp.sort_values(by=['Date'])
    # price = price.sort_values(by=['Date'])
    # price['crit'] = temp['crit']

    return temp

def supresret():
    print('x')