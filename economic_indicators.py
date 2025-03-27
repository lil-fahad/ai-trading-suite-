
import requests
import pandas as pd

def get_fred_data(api_key, indicator='FEDFUNDS'):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={indicator}&api_key={api_key}&file_type=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['observations']
        df = pd.DataFrame(data)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df[['value']]
    else:
        return pd.DataFrame()
