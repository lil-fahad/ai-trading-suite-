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

st.set_page_config(layout="wide")
st.title("تطبيق التنبؤ الذكي للأسهم والعملات - النسخة المطورة")

symbol = st.sidebar.text_input("ادخل رمز السهم أو العملة", value="AAPL").upper()
future_days = st.sidebar.slider("عدد الأيام للتنبؤ:", 1, 10, 3)
run = st.sidebar.button("تشغيل النموذج")

if run:
    df = yf.download(symbol, period="5y")

    if df.empty or len(df) < 70:
        st.error("لم يتم العثور على بيانات كافية لهذا الرمز. يرجى تجربة رمز آخر.")
        st.stop()

    # مؤشرات فنية متقدمة
    df["SMA_10"] = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["RSI_14"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["CCI_20"] = CCIIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=20).cci()
    df["ADX_14"] = ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"], window=14).adx()
    df["OBV"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_20', 'RSI_14', 'CCI_20', 'ADX_14', 'OBV']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    target_idx = features.index("Close")

    # تجهيز البيانات
    X, y = [], []
    window = 60
    for i in range(window, len(scaled) - future_days):
        X.append(scaled[i-window:i])
        y.append([scaled[i + d, target_idx] for d in range(future_days)])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    # بناء النموذج
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
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    direction_acc = np.mean(np.sign(np.diff(y_test[:, -1])) == np.sign(np.diff(preds[:, -1]))) * 100

    st.subheader("نتائج التقييم:")
    st.write(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | دقة الاتجاه: {direction_acc:.2f}%")

    st.subheader("مقارنة التنبؤ بالواقع (آخر 200 نقطة):")
    fig, ax = plt.subplots()
    ax.plot([x[-1] for x in y_test[-200:]], label="Actual")
    ax.plot([x[-1] for x in preds[-200:]], label="Predicted")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
