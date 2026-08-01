import joblib
import pandas as pd
import os
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features

# Matches sklearn's LabelEncoder alphabetical ordering used during training
# (fit on ['L','M','H'] -> classes_ sorted as ['H','L','M'] -> 0,1,2)
TYPE_MAPPING = {"H": 0, "L": 1, "M": 2}


def load_artifacts(base_dir):
    model = joblib.load(os.path.join(base_dir, "models/best_model.joblib"))
    scaler = joblib.load(os.path.join(base_dir, "models/scaler.joblib"))
    feature_columns = joblib.load(os.path.join(base_dir, "models/feature_columns.joblib"))
    return model, scaler, feature_columns


def make_prediction(data: dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model, scaler, feature_columns = load_artifacts(base_dir)

    df = pd.DataFrame([data])

    # Encode Type exactly the way it was encoded at training time
    if df["Type"].dtype == object:
        df["Type"] = df["Type"].map(TYPE_MAPPING)

    df = engineer_features(df)

    # Guarantee the exact column order the scaler/model were fit on
    df = df[feature_columns]

    X_scaled = scaler.transform(df)

    probability = float(model.predict_proba(X_scaled)[:, 1][0])
    prediction = "Failure" if probability > 0.5 else "No Failure"

    return {"failure_probability": probability, "prediction": prediction}


if __name__ == "__main__":
    sample_data = {
        "Type": "L",
        "Air_temperature_K": 298.0,
        "Process_temperature_K": 308.0,
        "Rotational_speed_rpm": 1500,
        "Torque_Nm": 42.0,
        "Tool_wear_min": 180,
    }
    result = make_prediction(sample_data)
    logging.info(f"Prediction result: {result}")