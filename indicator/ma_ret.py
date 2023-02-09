import numpy as np
import pandas as pd

def cross_pos(price):
    price['signal'] = np.where(price.iloc[:, 0] < price.iloc[:, 1], 1.0, 0.0)
    price['position'] = price['signal'].diff()

    return price

def cross_return(price):
    price = cross_pos(price)
    price['spend'] = price['Close'] * price['position']
    price['cash'] = price['Close'].mean()*10 - price['spend'].cumsum()

    return price[['cash']]

def cross_bs(price):
    price = cross_pos(price)
    price['b/s'] = pd.Series('-', index=price.index)
    price['b/s'] = price['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == -1 else '-'))

    return price[['b/s']]

def bol_pos(price):
    # signal -> 0-middle, 1-below lband, 2-above uband
    # position -> 0->1=1=buy, 1->0=-1=neutral, 0->2=2=sell, 2->0=-2=neutral
    price['signal'] = np.where(price.iloc[:, 0] < price.iloc[:, 2], 1.0, np.where(price.iloc[:, 0] > price.iloc[:, 1], 2.0, 0.0))
    price['position'] = price['signal'].diff()

    return price

def bol_return(price):
    price = bol_pos(price)
    bs = price[['position']]
    bs.loc[bs['position']==-1] = 0
    bs.loc[bs['position']==-2] = 0
    bs.loc[bs['position']==2] = -1
    price['spend'] = price['Close'] * bs['position']
    price['cash'] = price['Close'].mean()*10 - price['spend'].cumsum()

    return price[['cash']]

def bol_bs(price):
    price = bol_pos(price)
    price['b/s'] = pd.Series('-', index=price.index)
    price['b/s'] = price['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == 2 else '-'))

    return price[['b/s']]