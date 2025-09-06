import pandas as pd
import numpy as np
from tensorflow import keras
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler

def update_scalers_if_needed(df: pd.DataFrame, folder_path="./model/"):
    last_update_file = folder_path + "last_update.json"
    last_df_date_str = df['Date'].iloc[-1].strftime('%Y-%m-%d')
    
    last_updated_date = ""
    try:
        with open(last_update_file, 'r') as f:
            data = json.load(f)
            last_updated_date = data.get('last_updated_date')
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if last_updated_date != last_df_date_str:
        print("New data found. Updating scalers...")
        window_size = 90
        X = []
        y = []

        for i in range(window_size, len(df)):
            X.append(df['Close'].iloc[i-window_size:i].values)
            y.append(df['Close'].iloc[i])

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        scaler_inp = StandardScaler()
        scaler_out = StandardScaler()

        X_reshaped = X.reshape(-1, 1)
        scaler_inp.fit(X_reshaped)

        scaler_out.fit(y)

        joblib.dump(scaler_inp, folder_path + "x_scaler.pkl")
        joblib.dump(scaler_out, folder_path + "y_scaler.pkl")

        with open(last_update_file, 'w') as f:
            json.dump({'last_updated_date': last_df_date_str}, f)
        print("Scalers updated and saved.")

def predict_next_day(df: pd.DataFrame) -> float:
    folder_path = "./model/"
    update_scalers_if_needed(df, folder_path)

    model = keras.models.load_model(folder_path + "LSTM_model_1.keras")
    x_scaler = joblib.load(folder_path + "x_scaler.pkl")
    y_scaler = joblib.load(folder_path + "y_scaler.pkl")

    window_size = 90

    test_data = df['Close'].iloc[-window_size:]
    X_test = test_data.to_numpy()

    X_test_scaled = x_scaler.transform(X_test.reshape(-1, 1))

    X_test_reshaped = X_test_scaled.reshape(1, window_size, 1)

    prediction_scaled = model.predict(X_test_reshaped)


    prediction = y_scaler.inverse_transform(prediction_scaled)

    return prediction[0][0]

if __name__ == '__main__':
    try:
        df = pd.read_pickle("./data/asian_paints.pkl")
        df['Date'] = pd.to_datetime(df['Date'])
        prediction = predict_next_day(df)
        print(f"Predicted next day's Close price: {prediction}")
    except FileNotFoundError:
        print("Error: Make sure './data/asian_paints.pkl', './model/LSTM_model_1.keras', './model/x_scaler.pkl', and './model/y_scaler.pkl' exist.")
    except Exception as e:
        print(f"An error occurred: {e}")
