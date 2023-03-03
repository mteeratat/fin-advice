import pandas as pd
from indicator.ma_graph import SMA

def bolband(price, days, factor):
    sma = SMA(price, days)
    std = price.std()
    # factor = 1
    price['uband'] = pd.DataFrame(sma + float(factor)*std)
    price['lband'] = pd.DataFrame(sma - float(factor)*std)

    return price