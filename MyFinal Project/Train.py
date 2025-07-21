import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,                          confusion_matrix, roc_curve, auc)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os
import evaluation_plots as ep

# directory
os.makedirs("model_artifacts", exist_ok=True)
os.makedirs("model_artifacts/plots", exist_ok=True)



def train_models():
    # Load data
    df = pd.read_csv("sensor_maintenance_data.csv")

    sensor_cols = [
        "Voltage (V)", "Current (A)", "Temperature (°C)", "Power (W)", 
        "Humidity (%)", "Vibration (m/s²)", "Ambient Temperature (°C)", 
        "Ambient Humidity (%)", "X", "Y", "Z"
    ]
    target_cols = ["Predictive Maintenance Trigger", "Fault Detected", "Repair Time (hrs)", "Maintenance Type"]
    df = df[sensor_cols + target_cols]

    # Encode Maintenance
    le_maintenance = LabelEncoder()
    df["Maintenance Type"] = le_maintenance.fit_transform(df["Maintenance Type"])

    # Features and labels
    X = df[sensor_cols]
    y_trigger = df["Predictive Maintenance Trigger"]
    y_fault = df["Fault Detected"]
    y_repair = df["Repair Time (hrs)"].fillna(0)
    y_maintenance = df["Maintenance Type"]

    # Split data
    X_train, X_test, y_trigger_train, y_trigger_test = train_test_split(
        X, y_trigger, test_size=0.2, random_state=42, stratify=y_trigger
    )
    _, _, y_maintenance_train, _ = train_test_split(
        X, y_maintenance, test_size=0.2, random_state=42, stratify=y_maintenance
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("\nTraining Random Forest models with cross-validation...")
    rf_trigger = RandomForestClassifier(
        n_estimators=150, 
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in skf.split(X_train, y_trigger_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_trigger_train.iloc[train_idx], y_trigger_train.iloc[val_idx]
        
        rf_trigger.fit(X_train_fold, y_train_fold)
        val_pred = rf_trigger.predict(X_val_fold)
        cv_scores.append(accuracy_score(y_val_fold, val_pred))
    
    print(f"RF Cross-Validation Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    rf_trigger.fit(X_train, y_trigger_train)

    ep.plot_feature_importance(rf_trigger, sensor_cols, "RF_Trigger")

    rf_maintenance = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=10,
        random_state=42
    )
    rf_maintenance.fit(X_train, y_maintenance_train)
    ep.plot_feature_importance(rf_maintenance, sensor_cols, "RF_Maintenance")

    # Extract RF Predictions for hybrid model
    rf_train_features = rf_trigger.predict_proba(X_train)
    rf_test_features = rf_trigger.predict_proba(X_test)

    # Combine RF features with original inputs for ANN
    X_train_hybrid = np.hstack((X_train_scaled, rf_train_features))
    X_test_hybrid = np.hstack((X_test_scaled, rf_test_features))

    # Enhanced ANN architecture
    ann = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_hybrid.shape[1],), 
              kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.6),
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', 
                metrics=['accuracy', tf.keras.metrics.AUC()])

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=0.001
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    print("\nTraining hybrid ANN model...")
    history = ann.fit(
        X_train_hybrid, y_trigger_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test_hybrid, y_trigger_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Save models
    joblib.dump(rf_trigger, "model_artifacts/rf_trigger_model.pkl")
    joblib.dump(rf_maintenance, "model_artifacts/rf_maintenance_model.pkl")
    ann.save("model_artifacts/hybrid_ann.h5")
    joblib.dump(scaler, "model_artifacts/scaler.pkl")
    joblib.dump(le_maintenance, "model_artifacts/maintenance_encoder.pkl")
    
    print("\nModel training completed. Saved all models and encoders.")

    # Enhanced evaluation
    print("\n=== Final Model Evaluation ===")
    loss, accuracy, auc_score = ann.evaluate(X_test_hybrid, y_trigger_test, verbose=0)
    print(f"Hybrid  Test Loss: {loss:.4f}")
    print(f"Hybrid  Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Hybrid  Test AUC: {auc_score * 100:.2f}%")

    # Generate predictions for plots
    y_pred_ann = (ann.predict(X_test_hybrid) > 0.5).astype("int32")
    y_probs_ann = ann.predict(X_test_hybrid).ravel()
    
    # Save  evaluation plots
    ep.save_plots(history, y_trigger_test, y_pred_ann, y_probs_ann, "Hybrid_ANN")

    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_trigger_test, y_pred_ann))

    rf_pred = rf_trigger.predict(X_test)
    rf_probs = rf_trigger.predict_proba(X_test)[:, 1]
    ep.save_plots(None, y_trigger_test, rf_pred, rf_probs, "RandomForest")


    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[sensor_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("model_artifacts/plots/feature_correlation.png")
    plt.close()

if __name__ == "__main__":
    train_models()