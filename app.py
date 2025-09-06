from flask import Flask, render_template, request, Response, jsonify
import pandas as pd
from plot_utils import create_plot
from update_data import update_data as update_stock_data
from predictor import predict_next_day
import os

app = Flask(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'asian_paints.pkl')

def load_data():
    df = pd.read_pickle(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

df = load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    return df.head().to_json(orient='table')

@app.route('/update', methods=['POST'])
def update():
    global df
    update_stock_data()
    df = load_data()
    return jsonify(success=True, message="Data updated successfully")

@app.route('/plot')
def plot():
    time_range = request.args.get('time_range', 'max')
    graph_json_str = create_plot(df, time_range)
    return Response(graph_json_str, mimetype='application/json')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        prediction = predict_next_day(df)
        return jsonify(success=True, prediction=f'{prediction:.2f}')
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify(success=False, message=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
