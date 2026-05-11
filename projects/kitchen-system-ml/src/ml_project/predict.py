from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml

from ml_project.preprocessing import preprocess_kitchen_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    """Load project configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_model_package(model_path: Path) -> dict:
    """Load a saved model package."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model package not found: {model_path}")

    return joblib.load(model_path)


def apply_saved_temperature_imputation(
    data: pd.DataFrame,
    monthly_temperature_medians: pd.Series,
    global_temperature_median: float,
    temperature_column: str = "temperature",
    month_column: str = "month",
) -> pd.DataFrame:
    """Apply saved month-based temperature imputation values."""
    transformed_data = data.copy()

    month_based_values = transformed_data[month_column].map(monthly_temperature_medians)
    fallback_values = month_based_values.fillna(global_temperature_median)

    transformed_data[temperature_column] = transformed_data[temperature_column].fillna(fallback_values)

    return transformed_data


def predict_orders(input_data: pd.DataFrame) -> pd.DataFrame:
    """Predict total orders for new kitchen service data."""
    config = load_config(CONFIG_PATH)

    model_path = (
        PROJECT_ROOT
        / config["training"]["model_dir"]
        / config["training"]["model_file"]
    )

    model_package = load_model_package(model_path)

    model = model_package["model"]
    model_features = model_package["model_features"]
    target_column = model_package["target_column"]
    original_feature_columns = config["features"]["original_feature_columns"]

    prediction_data = input_data.copy()

    if target_column not in prediction_data.columns:
        prediction_data[target_column] = 0

    processed_data = preprocess_kitchen_data(
        data=prediction_data,
        feature_columns=original_feature_columns,
        target_column=target_column,
    )

    processed_features = processed_data[model_features]

    processed_features = apply_saved_temperature_imputation(
        data=processed_features,
        monthly_temperature_medians=model_package["monthly_temperature_medians"],
        global_temperature_median=model_package["global_temperature_median"],
    )

    predictions = model.predict(processed_features)

    results = input_data.copy()
    results["predicted_total_orders"] = predictions.round(0).astype(int)

    return results


def main() -> None:
    """Run a small example prediction."""
    example_data = pd.DataFrame(
        [
            {
                "date": "2026-05-16",
                "day_of_week": "Saturday",
                "weekend": "yes",
                "holiday": "no",
                "service_type": "dinner",
                "weather": "sunny",
                "temperature": "18C",
                "special_event": "yes",
            }
        ]
    )

    predictions = predict_orders(example_data)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
