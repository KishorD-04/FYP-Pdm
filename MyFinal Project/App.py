from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import make_prediction as mp

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Convert string values to float
        sensor_data = {
            "Voltage (V)": float(data['voltage']),
            "Current (A)": float(data['current']),
            "Temperature (°C)": float(data['temperature']),
            "Power (W)": float(data['power']),
            "Humidity (%)": float(data['humidity']),
            "Vibration (m/s²)": float(data['vibration']),
            "Ambient Temperature (°C)": float(data['ambient_temp']),
            "Ambient Humidity (%)": float(data['ambient_humidity']),
            "X": float(data['x_axis']),
            "Y": float(data['y_axis']),
            "Z": float(data['z_axis'])
        }
        
        prediction = mp.make_prediction(sensor_data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)