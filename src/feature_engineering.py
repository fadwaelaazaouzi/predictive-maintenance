import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def engineer_features(df):
    """Creates new features for the model. Expects the snake_case column
    names produced by preprocessing.py (e.g. 'Air_temperature_K')."""
    logging.info("Starting feature engineering...")
    df = df.copy()

    # Temperature Difference: Process temp - Air temp
    df["Temp_Diff"] = df["Process_temperature_K"] - df["Air_temperature_K"]

    # Power: Torque * Rotational speed
    df["Power"] = df["Torque_Nm"] * df["Rotational_speed_rpm"]

    # Tool Wear x Torque: a simple interaction term
    df["Tool_Wear_Torque"] = df["Tool_wear_min"] * df["Torque_Nm"]

    logging.info("Feature engineering complete.")
    return df


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data/processed/preprocessed_data.csv")
    output_path = os.path.join(base_dir, "data/processed/engineered_data.csv")

    df = pd.read_csv(input_path)
    df = engineer_features(df)
    df.to_csv(output_path, index=False)
    logging.info(f"Engineered data saved to {output_path}")