import pandas as pd
import numpy as np

def rsi_pos(price):
    # signal -> 0-middle, 1-below lband, 2-above uband
    # position -> 0->1=1=buy, 1->0=-1=neutral, 0->2=2=sell, 2->0=-2=neutral
    price['signal'] = np.where(price.iloc[:, 0] < 30, 1.0, np.where(price.iloc[:, 0] > 70, 2.0, 0.0))
    state = 0
    price['position'] = price['signal'].diff()
    for i in range(len(price['signal'])):
        # print(i)
        if state == 0 and price['signal'].iloc[i] == 1:
            price['position'].iloc[i] = 1
            state = 1
        elif state == 1 and price['signal'].iloc[i] == 2:
            price['position'].iloc[i] = -1
            state = 0
        else:
            price['position'].iloc[i] = 0
    # print(price)
        
    return price

def rsi_return(price):
    temp = rsi_pos(price[['rsi']])
    bs = temp[['position']]
    # bs.loc[bs['position']==-1] = 0
    # bs.loc[bs['position']==-2] = 0
    # bs.loc[bs['position']==2] = -1
    temp['spend'] = price['Close'] * bs['position']
    temp['cash'] = price['Close'].mean()*10 - temp['spend'].cumsum()
    # print(temp)

    return temp[['cash']]

def rsi_bs(price):
    temp = rsi_pos(price[['rsi']])
    temp['b/s'] = pd.Series('-', index=price.index)
    # price['b/s'] = price['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == 2 else '-'))
    price['b/s'] = temp['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == -1 else '-'))

    return temp[['b/s']]