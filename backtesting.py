
import pandas as pd
import numpy as np

def backtest_strategy(prices, signals):
    capital = 10000
    position = 0
    equity = []
    for i in range(len(prices)):
        if signals[i] == 1 and capital > prices[i]:
            position += 1
            capital -= prices[i]
        elif signals[i] == -1 and position > 0:
            capital += position * prices[i]
            position = 0
        equity.append(capital + position * prices[i])
    return pd.Series(equity, index=prices.index)
