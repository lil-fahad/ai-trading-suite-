
import numpy as np

def apply_momentum_strategy(predictions):
    preds = np.array(predictions)
    signal = np.sign(np.diff(preds[:, -1], prepend=preds[0, -1]))
    return signal

def apply_rsi_strategy(rsi_array, low=30, high=70):
    return np.where(rsi_array < low, 1, np.where(rsi_array > high, -1, 0))

def apply_sma_crossover(df):
    return np.where(df["SMA_10"] > df["EMA_20"], 1, -1)
