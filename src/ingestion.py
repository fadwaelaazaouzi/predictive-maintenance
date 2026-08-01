import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXPECTED_COLUMNS = [
    'UDI', 'Product ID', 'Type', 'Air temperature [K]',
    'Process temperature [K]', 'Rotational speed [rpm]',
    'Torque [Nm]', 'Tool wear [min]', 'Machine failure',
    'TWF', 'HDF', 'PWF', 'OSF', 'RNF'
]


def load_data(file_path):
    """Loads the raw dataset from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    return df


def validate_data(df):
    """Validates that the raw dataset has the columns we expect."""
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")

    logging.info("Data validation successful.")
    return True


def save_processed_data(df, output_path):
    """Saves a (still-raw-schema) copy of the dataset for the next pipeline step."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(base_dir, "data/raw/ai4i2020.csv")
    processed_data_path = os.path.join(base_dir, "data/processed/data.csv")

    try:
        data = load_data(raw_data_path)
        validate_data(data)
        save_processed_data(data, processed_data_path)
    except Exception as e:
        logging.error(f"Ingestion failed: {e}")