from pathlib import Path

import joblib
import pandas as pd

from data_preparation import clean_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "purchase_intention_model.joblib"


def predict_from_csv(input_path: str | Path) -> pd.DataFrame:
    """Predict purchase probability for customer sessions in a CSV file."""
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    raw_df = pd.read_csv(input_path, index_col=0)
    cleaned_df = clean_data(raw_df)
    X = cleaned_df.drop(columns=["Revenue"])
    X = X.reindex(columns=feature_columns, fill_value=0)

    predictions = cleaned_df.copy()
    predictions["purchase_probability"] = model.predict_proba(X)[:, 1]
    predictions["predicted_revenue"] = model.predict(X)
    return predictions


if __name__ == "__main__":
    output = predict_from_csv(PROJECT_ROOT / "data" / "raw" / "project_data.csv")
    output.to_csv(PROJECT_ROOT / "reports" / "predictions.csv", index=False)
    print("Predictions saved to reports/predictions.csv")
