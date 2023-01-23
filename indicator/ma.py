import numpy as np
import pandas as pd

def SMA(df_price):
    data = df_price.rolling(5).mean()
    # print(data)
    # print(data)
    # print(data['Close'])
    # print(data[['Close']])
    return data[['Close']]

def EMA(df_price):
    ema = []
    for i in df_price['Close'][:4]:
        ema.append(np.NaN)
        # print(df_price[:5].mean()[0])
        # print(type(df_price[:5].mean()[0]))
    ema.append(df_price[:5].mean()[0])
    for i in df_price['Close'][5:]:
        # print(i)
        # print(type(i))
        # print(ema[-1])
        ema.append(i*2/(1+5) + ema[-1]*(1-2/(1+5)))
    # print(ema)
    ema = pd.Series(ema).T
    # print(ema)
    # print(type(ema))
    # print(df_price)
    # print(type(df_price))
    # ema = ema.reindex(index=df_price.index)
    ema.index = df_price.index
    # print(ema)
    # df_price.loc[:, 'Close'] = ema
    # print(df_price.index)
    df_price['Close'] = ema.copy()
    # print(df_price)
    return df_price