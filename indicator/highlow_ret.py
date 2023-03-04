import numpy as np
import pandas as pd

def highlow_pos(crit):
    # signal -> 0-middle, 1-below lband, 2-above uband
    # position -> 0->1=1=buy, 1->0=-1=neutral, 0->2=2=sell, 2->0=-2=neutral
    crit['signal'] = np.where(crit.iloc[:, 1] == 1, 1.0, np.where(crit.iloc[:, 2] == 1, 2.0, 0.0))
    state = 0
    crit['position'] = crit['signal'].diff()
    for i in range(len(crit['signal'])):
        # print(i)
        if state == 0 and crit['signal'].iloc[i] == 1:
            crit['position'].iloc[i] = 1
            state = 1
        elif state == 1 and crit['signal'].iloc[i] == 2:
            crit['position'].iloc[i] = -1
            state = 0
        else:
            crit['position'].iloc[i] = 0
        
    return crit

def highlow_ret(price):
    temp = highlow_pos(price[['crit','lows','highs']])
    bs = temp[['position']]
    # bs.loc[bs['position']==-1] = 0
    # bs.loc[bs['position']==-2] = 0
    # bs.loc[bs['position']==2] = -1
    temp['spend'] = price['Close'] * bs['position']
    temp['cash'] = price['Close'].mean()*10 - temp['spend'].cumsum()
    # print(temp)

    return temp[['cash']]

def highlow_bs(price):
    temp = highlow_pos(price[['crit','lows','highs']])
    temp['b/s'] = pd.Series('-', index=price.index)
    # price['b/s'] = price['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == 2 else '-'))
    price['b/s'] = temp['position'].apply(lambda x: 'buy' if x == 1 else ('sell' if x == -1 else '-'))

    return temp[['b/s']]