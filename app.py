
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, EMAIndicator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta

st.set_page_config(layout="wide")
st.title("نظام تنبؤ وتوصية احترافي بالأسهم والخيارات")

symbol = st.sidebar.text_input("ادخل رمز السهم", value="AAPL").upper()
future_days = st.sidebar.slider("عدد الأيام للتنبؤ:", 1, 10, 3)
run = st.sidebar.button("ابدأ التنبؤ والتحليل")

if run:
    df = yf.download(symbol, period="5y")

    if df.empty or len(df) < 100:
        st.error("بيانات غير كافية لهذا الرمز.")
        st.stop()

    df["Close"] = df["Close"].astype(float).squeeze()
    df["High"] = df["High"].astype(float).squeeze()
    df["Low"] = df["Low"].astype(float).squeeze()
    df["Volume"] = df["Volume"].astype(float).squeeze()

    df["SMA_10"] = SMAIndicator(close=df["Close"], window=10).sma_indicator()
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'EMA_20']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    target_idx = features.index("Close")

    X, y = [], []
    window = 60
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

    st.subheader("تنبؤات السعر حسب التاريخ")
    st.dataframe(results_df)

    st.subheader("الرسم البياني للتوقع مقابل الفعلي")
    plt.figure(figsize=(10, 4))
    plt.plot([x[-1] for x in y_test[-200:]], label="Actual")
    plt.plot([x[-1] for x in preds[-200:]], label="Predicted")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # توصيات خيارات مرتبة حسب أعلى ربحية نظرية (نموذجية)
    st.subheader("أفضل توصيات عقود الخيارات (افتراضية)")
    base_price = df["Close"].iloc[-1]
    option_scenarios = []
    for strike_offset in range(-10, 11, 5):
        strike_price = round(base_price + strike_offset, 2)
        option_price = 2.5 + abs(strike_offset) * 0.5
        expected_profit = (last_prediction[-1] - strike_price - option_price) if strike_price < last_prediction[-1] else 0
        profit_pct = (expected_profit / option_price) * 100 if option_price > 0 else 0
        option_scenarios.append({
            "Strike": strike_price,
            "Premium": option_price,
            "Expected Return %": round(profit_pct, 2)
        })

    ranked = pd.DataFrame(option_scenarios).sort_values(by="Expected Return %", ascending=False)
    st.dataframe(ranked)
