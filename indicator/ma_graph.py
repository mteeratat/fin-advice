import numpy as np
import pandas as pd

def SMA(price, days):
    data = price.rolling(int(days)).mean()
    return data

def EMA(price, days):
    ema = []
    for i in price['Close'][:int(days)-1]:
        ema.append(np.NaN)
    ema.append(price[:int(days)].mean()[0])
    for i in price['Close'][int(days):]:
        ema.append(i*2/(1+int(days)) + ema[-1]*(1-2/(1+int(days))))
    ema = pd.Series(ema).T
    ema.index = price.index
    res = ema.copy()
    return res

def bolband(price, days, factor):
    sma = SMA(price, days)
    std = price.std()
    # factor = 1
    price['uband'] = pd.DataFrame(sma + int(factor)*std)
    price['lband'] = pd.DataFrame(sma - int(factor)*std)

    return price

def APO(price):
    print('x')