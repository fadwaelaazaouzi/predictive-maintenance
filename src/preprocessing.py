import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Maps the original AI4I column names (with spaces/brackets) to clean
# snake_case names used by every script downstream of this one.
RENAME_MAP = {
    'Air temperature [K]': 'Air_temperature_K',
    'Process temperature [K]': 'Process_temperature_K',
    'Rotational speed [rpm]': 'Rotational_speed_rpm',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_wear_min',
    'Machine failure': 'Machine_failure',
}

def preprocess_data(df):
    """Basic preprocessing: drop columns, handle categorical data, rename columns."""
    logging.info("Starting preprocessing...")

    # Drop UDI and Product ID as they are unique identifiers
    df = df.drop(['UDI', 'Product ID'], axis=1)

    # Encode 'Type' (L, M, H)
    le = LabelEncoder()
    df['Type'] = le.fit_transform(df['Type'])

    # Standardize column names so every downstream script agrees on them
    df = df.rename(columns=RENAME_MAP)

    # Handle missing values if any (though this dataset is usually clean)
    df = df.dropna()

    logging.info("Preprocessing complete.")
    return df

def split_data(df, target_col='Machine_failure', test_size=0.2, random_state=42):
    """Splits data into training and testing sets."""
    # 'TWF', 'HDF', 'PWF', 'OSF', 'RNF' are individual failure-mode columns;
    # we drop them because they'd leak the answer for binary classification.
    cols_to_drop = [target_col, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(cols_to_drop, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logging.info(f"Data split: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_path = os.path.join(base_dir, "data/processed/data.csv")

    df = pd.read_csv(processed_data_path)
    df = preprocess_data(df)
    df.to_csv(os.path.join(base_dir, "data/processed/preprocessed_data.csv"), index=False)
    logging.info("Preprocessed data saved.")