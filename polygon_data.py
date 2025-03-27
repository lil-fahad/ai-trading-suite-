
import streamlit as st
from polygon import RESTClient
import pandas as pd
from datetime import datetime, timedelta

# استخدام مفتاح API من st.secrets
api_key = st.secrets["POLYGON_API_KEY"]
client = RESTClient(api_key)

def get_stock_data(symbol, days=365):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    aggs = client.get_aggs(ticker=symbol, multiplier=1, timespan="day", from_=start_date, to=end_date)
    df = pd.DataFrame(aggs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def get_options_chain(symbol, exp_date=None, limit=10):
    if not exp_date:
        today = datetime.now().strftime('%Y-%m-%d')
        exp_date = today
    results = client.list_options_contracts(underlying_ticker=symbol, expiration_date=exp_date, limit=limit)
    contracts = []
    for res in results:
        contracts.append({
            'symbol': res.symbol,
            'type': res.contract_type,
            'strike_price': res.strike_price,
            'expiration_date': res.expiration_date,
            'exercise_style': res.exercise_style
        })
    return pd.DataFrame(contracts)
