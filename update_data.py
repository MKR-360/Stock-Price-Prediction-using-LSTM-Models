import requests
import pandas as pd
import os

def update_data():
    """
    Fetches the latest stock data for Asian Paints from Alpha Vantage,
    processes it, and merges it with the existing data.
    """
    # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
    # It is recommended to use an environment variable for the API key.
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    symbol = 'ASIANPAINT.BSE'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
    
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alpha Vantage: {e}")
        return

    if 'Time Series (Daily)' not in data:
        print("Error: 'Time Series (Daily)' not in API response. The response was:")
        print(data)
        return

    new_df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

    new_df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    }, inplace=True)
    
    new_df.index.name = 'Date'
    new_df.reset_index(inplace=True)
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        new_df[col] = new_df[col].astype('float64')

    pickle_file_path = os.path.join(os.path.dirname(__file__), 'data', 'asian_paints.pkl')

    if os.path.exists(pickle_file_path):
        existing_df = pd.read_pickle(pickle_file_path)
        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
    else:
        existing_df = pd.DataFrame()

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    
    combined_df.sort_values('Date', inplace=True)

    combined_df.to_pickle(pickle_file_path)
    
    print(f"Data updated and saved to {pickle_file_path}")
    print(f"Total records: {len(combined_df)}")
    print("Latest data point:")
    print(combined_df.tail(1))

if __name__ == '__main__':
    update_data()
