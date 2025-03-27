
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polygon_data import get_stock_data, get_options_chain

st.set_page_config(layout="wide")
st.title("منصة التداول الذكي باستخدام Polygon.io")

symbol = st.sidebar.text_input("ادخل رمز السهم", value="AAPL").upper()
days = st.sidebar.slider("عدد الأيام للبيانات:", 30, 720, 365)
load_data = st.sidebar.button("جلب البيانات")

if load_data:
    with st.spinner("جاري تحميل البيانات..."):
        df = get_stock_data(symbol, days=days)

    if df.empty or len(df) < 50:
        st.error("البيانات غير كافية.")
        st.stop()

    st.subheader(f"عرض بيانات السهم: {symbol}")
    st.line_chart(df["close"])

    st.write("آخر 5 أيام:")
    st.dataframe(df[["open", "high", "low", "close", "volume"]].tail(5))

    st.subheader("عقود الخيارات المتاحة")
    options_df = get_options_chain(symbol, limit=20)
    st.dataframe(options_df)
