import numpy as np
import pandas as pd

def SMA(price):
    data = price.rolling(5).mean()
    return data

def EMA(price):
    ema = []
    for i in price['Close'][:4]:
        ema.append(np.NaN)
    ema.append(price[:5].mean()[0])
    for i in price['Close'][5:]:
        ema.append(i*2/(1+5) + ema[-1]*(1-2/(1+5)))
    ema = pd.Series(ema).T
    ema.index = price.index
    res = ema.copy()
    return res

def bolband(price):
    sma = SMA(price)
    std = price.std()
    factor = 1
    price['uband'] = pd.DataFrame(sma + factor*std)
    price['lband'] = pd.DataFrame(sma - factor*std)

    return price