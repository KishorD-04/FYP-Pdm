from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def make_prediction(sensor_data):
    # Load saved
    rf_trigger = joblib.load("model_artifacts/rf_trigger_model.pkl")
    rf_maintenance = joblib.load("model_artifacts/rf_maintenance_model.pkl")
    ann = tf.keras.models.load_model("model_artifacts/hybrid_ann.h5")
    scaler = joblib.load("model_artifacts/scaler.pkl")
    le_maintenance = joblib.load("model_artifacts/maintenance_encoder.pkl")

    #  sensor data to DataFrame
    new_data = pd.DataFrame([sensor_data])

    # Standardize
    X_new_scaled = scaler.transform(new_data)
    rf_new_features = rf_trigger.predict_proba(new_data)

    # Combine
    X_new_hybrid = np.hstack((X_new_scaled, rf_new_features))
    predictions = ann.predict(X_new_hybrid)
    maintenance_trigger = (predictions > 0.5).astype(int)[0][0]

    # Predict Fault
    fault_detected = rf_trigger.predict(new_data)[0]

    # Predict Repair Time
    repair_time = rf_trigger.predict(new_data)[0] if fault_detected else 0

    # Predict Maintenance
    maintenance_type = None
    if maintenance_trigger:
        maintenance_code = rf_maintenance.predict(new_data)[0]
        maintenance_type = le_maintenance.inverse_transform([maintenance_code])[0]

    #  output dictionary
    result = {
        "predictive_maintenance_trigger": bool(maintenance_trigger),
        "maintenance_needed": "Yes" if maintenance_trigger else "No",
        "fault_detected": bool(fault_detected),
        "repair_time_hrs": float(repair_time),
        "maintenance_type": maintenance_type if maintenance_trigger else "None",
        **sensor_data
    }

    return result
