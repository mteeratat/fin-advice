def rsi(price, n=14):
    # Calculate price change from previous day
    delta = price.diff().dropna()

    # Define up and down moves
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss over period
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    # print(rsi)

    return rsi