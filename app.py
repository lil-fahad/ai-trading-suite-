import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, EMAIndicator, CCIIndicator, ADXIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("نظام تنبؤ احترافي للأسهم بالتواريخ والتوصيات")

symbol = st.sidebar.text_input("ادخل رمز السهم", value="AAPL").upper()
future_days = st.sidebar.slider("عدد الأيام للتنبؤ:", 1, 10, 3)
run = st.sidebar.button("ابدأ التحليل")

if run:
    df = yf.download(symbol, period="5y")
    if df.empty or len(df) < 100:
        st.error("بيانات غير كافية لهذا الرمز.")
        st.stop()

    df["Close"] = df["Close"].astype(float).squeeze()
    df["High"] = df["High"].astype(float).squeeze()
    df["Low"] = df["Low"].astype(float).squeeze()
    df["Volume"] = df["Volume"].astype(float).squeeze()

    # مؤشرات فنية مصححة 1D
    df["SMA_10"] = SMAIndicator(close=df["Close"].squeeze(), window=10).sma_indicator()
    df["EMA_20"] = EMAIndicator(close=df["Close"].squeeze(), window=20).ema_indicator()
    df["RSI_14"] = RSIIndicator(close=df["Close"].squeeze(), window=14).rsi()
    df["CCI_20"] = CCIIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze(), window=20).cci()
    df["ADX_14"] = ADXIndicator(high=df["High"].squeeze(), low=df["Low"].squeeze(), close=df["Close"].squeeze(), window=14).adx()
    df["OBV"] = OnBalanceVolumeIndicator(close=df["Close"].squeeze(), volume=df["Volume"].squeeze()).on_balance_volume()

    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_20', 'RSI_14', 'CCI_20', 'ADX_14', 'OBV']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    target_idx = features.index("Close")

    window = 60
    X, y = [], []
    for i in range(window, len(scaled) - future_days):
        X.append(scaled[i-window:i])
        y.append([scaled[i + d, target_idx] for d in range(future_days)])
    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(future_days)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    preds = model.predict(X_test)
    last_prediction = preds[-1]

    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]

    actuals = [None]*len(last_prediction)
    direction = ["صاعد" if i > last_prediction[idx - 1] else "هابط" if idx > 0 else "-" for idx, i in enumerate(last_prediction)]
    recommendation = ["شراء" if d == "صاعد" else "بيع" for d in direction]

    results_df = pd.DataFrame({
        "التاريخ": forecast_dates,
        "السعر المتوقع": np.round(last_prediction, 2),
        "الاتجاه": direction,
        "التوصية": recommendation
    })

    st.subheader("تنبؤات الأسعار حسب التواريخ:")
    st.dataframe(results_df)

    st.subheader("الرسم البياني للمقارنة")
    plt.figure(figsize=(10, 4))
    plt.plot([x[-1] for x in y_test[-200:]], label="Actual")
    plt.plot([x[-1] for x in preds[-200:]], label="Predicted")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
