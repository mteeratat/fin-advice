import numpy as np
import pandas as pd

def cross_pos(price, col1, col2):
    price['signal'] = np.where(price.iloc[:, col1] < price.iloc[:, col2], 1.0, 0.0)
    price['position'] = price['signal'].diff()

    return price

def cross_return(price, col1, col2):
    price = cross_pos(price,col1,col2)
    price['spend'] = price['Close'] * price['position']
    price['cash'] = price['Close'].mean()*10 - price['spend'].cumsum()

    return price[['cash']]

def cross_bs(price, col1, col2):
    price = cross_pos(price,col1,col2)
    price['b/s'] = pd.Series('-', index=price.index)
    price['b/s'] = price['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == -1 else '-'))

    return price[['b/s']]