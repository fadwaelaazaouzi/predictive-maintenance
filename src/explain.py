import shap
import joblib
import pandas as pd
import os
import logging
import matplotlib
matplotlib.use("Agg")  # render to file, no GUI window needed
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_shap_explanations(model, X_data, feature_names, output_dir):
    """Generates a global SHAP summary plot and a waterfall plot for one
    instance. Assumes a tree-based model (RandomForest/XGBoost) since it
    uses shap.TreeExplainer -- if train.py picks Logistic Regression as the
    best model instead, swap this for shap.LinearExplainer."""
    logging.info("Generating SHAP explanations...")
    os.makedirs(output_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    # Some tree models return a list [class0_values, class1_values] for
    # binary classification; others return a single 2D array. Normalize
    # to "the values for the positive (failure) class".
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values_pos = shap_values
        expected_value = explainer.expected_value

    # Global feature importance
    shap.summary_plot(shap_values_pos, X_data, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
    plt.clf()
    logging.info("Saved SHAP summary plot.")

    # Explanation for a single instance (the first row)
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_pos[0],
            base_values=expected_value,
            data=X_data.iloc[0].values,
            feature_names=feature_names,
        ),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_waterfall_plot_instance_0.png"))
    plt.clf()
    logging.info("Saved SHAP waterfall plot for instance 0.")

    return shap_values_pos, expected_value


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base_dir, "models/best_model.joblib"))
    scaler = joblib.load(os.path.join(base_dir, "models/scaler.joblib"))
    feature_columns = joblib.load(os.path.join(base_dir, "models/feature_columns.joblib"))

    df = pd.read_csv(os.path.join(base_dir, "data/processed/engineered_data.csv"))
    X = df[feature_columns]

    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_columns)

    generate_shap_explanations(model, X_scaled, feature_columns, os.path.join(base_dir, "models"))
    logging.info("SHAP explanations generated and plots saved.")