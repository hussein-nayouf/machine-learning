from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_project.preprocessing import (
    apply_month_temperature_imputation,
    load_excel_dataset,
    preprocess_kitchen_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    """Load project configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_train_validation_test_data(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split cleaned data into train, validation, and test sets."""
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1.")

    X = data[feature_columns]
    y = data[target_column]

    temporary_size = validation_size + test_size

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temporary_size,
        random_state=random_state,
    )

    validation_fraction_of_temp = validation_size / temporary_size

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_fraction_of_temp,
        random_state=random_state,
    )

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(
    numeric_features: list[str],
    categorical_features: list[str],
    model_parameters: dict,
) -> Pipeline:
    """Build the final tuned Random Forest pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(**model_parameters)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_regression_model(
    model: Pipeline,
    X: pd.DataFrame,
    y_true: pd.Series,
    dataset_name: str,
) -> dict:
    """Evaluate a regression model using common regression metrics."""
    predictions = model.predict(X)

    return {
        "dataset": dataset_name,
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": mean_squared_error(y_true, predictions) ** 0.5,
        "r2_score": r2_score(y_true, predictions),
    }


def print_metrics(metrics: list[dict]) -> None:
    """Print model metrics in a readable table."""
    metrics_table = pd.DataFrame(metrics)
    print(metrics_table.to_string(index=False))


def build_model_package(
    model: Pipeline,
    X_train: pd.DataFrame,
    model_features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str,
) -> dict:
    """Build a model package with the trained model and preprocessing metadata."""
    return {
        "model": model,
        "monthly_temperature_medians": X_train.groupby("month")["temperature"].median(),
        "global_temperature_median": X_train["temperature"].median(),
        "model_features": model_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": target_column,
    }


def main() -> None:
    """Train, evaluate, and save the final kitchen order prediction model."""
    config = load_config(CONFIG_PATH)

    dataset_path = (
        PROJECT_ROOT
        / config["data"]["raw_dir"]
        / config["data"]["dataset_file"]
    )
    model_dir = PROJECT_ROOT / config["training"]["model_dir"]
    model_path = model_dir / config["training"]["model_file"]

    target_column = config["data"]["target_column"]
    original_feature_columns = config["features"]["original_feature_columns"]
    numeric_features = config["features"]["numeric_features"]
    categorical_features = config["features"]["categorical_features"]
    model_features = numeric_features + categorical_features

    raw_data = load_excel_dataset(dataset_path)
    processed_data = preprocess_kitchen_data(
        data=raw_data,
        feature_columns=original_feature_columns,
        target_column=target_column,
    )

    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        create_train_validation_test_data(
            data=processed_data,
            feature_columns=model_features,
            target_column=target_column,
            validation_size=config["training"]["validation_size"],
            test_size=config["training"]["test_size"],
            random_state=config["training"]["random_state"],
        )
    )

    X_train = apply_month_temperature_imputation(
        train_data=X_train,
        data_to_transform=X_train,
    )
    X_validation = apply_month_temperature_imputation(
        train_data=X_train,
        data_to_transform=X_validation,
    )
    X_test = apply_month_temperature_imputation(
        train_data=X_train,
        data_to_transform=X_test,
    )

    model = build_model(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_parameters=config["model"]["parameters"],
    )
    model.fit(X_train, y_train)

    metrics = [
        evaluate_regression_model(model, X_train, y_train, "train"),
        evaluate_regression_model(model, X_validation, y_validation, "validation"),
        evaluate_regression_model(model, X_test, y_test, "test"),
    ]

    print_metrics(metrics)

    model_package = build_model_package(
        model=model,
        X_train=X_train,
        model_features=model_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_column,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, model_path)

    print(f"\nModel package saved to: {model_path}")


if __name__ == "__main__":
    main()
