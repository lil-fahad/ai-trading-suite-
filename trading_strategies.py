
def apply_momentum_strategy(preds):
    return np.where(np.diff(preds[:, -1], prepend=preds[0, -1]) > 0, 1, -1)

def apply_rsi_strategy(rsi_values, low_thresh=30, high_thresh=70):
    return np.where(rsi_values < low_thresh, 1, np.where(rsi_values > high_thresh, -1, 0))

def apply_sma_crossover(df):
    short = df['SMA_10']
    long = df['EMA_20']
    return np.where(short > long, 1, -1)
