import pandas as pd
import numpy as np
import os
import logging
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TARGET_COL = "Machine_failure"
LEAKY_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]  # individual failure modes -> would leak the label


def load_engineered_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL] + LEAKY_COLS)
    y = df[TARGET_COL]
    return X, y


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    engineered_data_path = os.path.join(base_dir, "data/processed/engineered_data.csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    X, y = load_engineered_data(engineered_data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit the scaler ONLY on training data, then apply it to both sets.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=42,
        ),
    }

    results = {}
    fitted_models = {}
    for name, model in candidates.items():
        logging.info(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        results[name] = metrics
        fitted_models[name] = model
        logging.info(f"{name} -> {metrics}")

    # Pick the best model by ROC AUC (a solid default for imbalanced failure data)
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = fitted_models[best_name]
    logging.info(f"Best model: {best_name} (ROC AUC = {results[best_name]['roc_auc']:.4f})")

    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(list(X_train.columns), os.path.join(models_dir, "feature_columns.joblib"))

    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(models_dir, "model_comparison.csv"))
    logging.info(f"Saved best model ({best_name}) and scaler to {models_dir}")


if __name__ == "__main__":
    main()