from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_excel_dataset(file_path: Path) -> pd.DataFrame:
    """Load the kitchen dataset from an Excel file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    return pd.read_excel(file_path)


def clean_text_value(value: object) -> object:
    """Clean a single text-like value by lowercasing and stripping spaces."""
    if pd.isna(value):
        return value

    return str(value).strip().lower()


def standardize_binary_value(value: object) -> object:
    """Convert common yes/no style values into 1, 0, or missing."""
    cleaned_value = clean_text_value(value)

    if pd.isna(cleaned_value):
        return pd.NA

    positive_values = {"y", "yes", "true", "1"}
    negative_values = {"n", "no", "false", "0"}

    if cleaned_value in positive_values:
        return 1

    if cleaned_value in negative_values:
        return 0

    return pd.NA


def standardize_binary_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize selected binary columns to 1, 0, or missing."""
    cleaned_data = data.copy()

    for column in columns:
        cleaned_data[column] = cleaned_data[column].apply(standardize_binary_value).astype("Int64")

    return cleaned_data


def standardize_text_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize text columns by stripping spaces and converting to lowercase."""
    cleaned_data = data.copy()

    for column in columns:
        cleaned_data[column] = cleaned_data[column].apply(clean_text_value).astype("string")

    return cleaned_data


def normalize_weather_value(value: object) -> object:
    """Normalize weather labels into consistent categories."""
    cleaned_value = clean_text_value(value)

    if pd.isna(cleaned_value):
        return pd.NA

    weather_mapping = {
        "sun": "sunny",
        "rain": "rainy",
        "unknown": pd.NA,
    }

    return weather_mapping.get(cleaned_value, cleaned_value)


def clean_weather_column(data: pd.DataFrame, column: str = "weather") -> pd.DataFrame:
    """Clean and normalize a weather column."""
    cleaned_data = data.copy()
    cleaned_data[column] = cleaned_data[column].apply(normalize_weather_value).astype("string")

    return cleaned_data


def clean_temperature_value(value: object) -> object:
    """Convert temperature values like '23C' or 23 into numeric values."""
    if pd.isna(value):
        return pd.NA

    cleaned_value = str(value).strip().lower()
    invalid_values = {"--", "cold", "hot", ""}

    if cleaned_value in invalid_values:
        return pd.NA

    cleaned_value = cleaned_value.replace("c", "")

    try:
        return float(cleaned_value)
    except ValueError:
        return pd.NA


def clean_temperature_column(data: pd.DataFrame, column: str = "temperature") -> pd.DataFrame:
    """Clean a temperature column and convert it to numeric."""
    cleaned_data = data.copy()
    cleaned_data[column] = cleaned_data[column].apply(clean_temperature_value).astype("Float64")

    return cleaned_data


def add_time_features(
    data: pd.DataFrame,
    date_column: str = "date",
    day_column: str = "day_of_week",
) -> pd.DataFrame:
    """Create date-based and calendar-based features."""
    cleaned_data = data.copy()

    cleaned_data[date_column] = pd.to_datetime(cleaned_data[date_column], errors="coerce")

    cleaned_data["year"] = cleaned_data[date_column].dt.year
    cleaned_data["month"] = cleaned_data[date_column].dt.month
    cleaned_data["day"] = cleaned_data[date_column].dt.day
    cleaned_data["day_of_year"] = cleaned_data[date_column].dt.dayofyear

    weekend_days = {"saturday", "sunday"}
    cleaned_data["is_weekend"] = cleaned_data[day_column].isin(weekend_days).astype(int)

    return cleaned_data


def prepare_data_for_sklearn(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare cleaned data for scikit-learn by removing pandas nullable dtypes."""
    prepared_data = data.copy()

    for column in prepared_data.columns:
        prepared_data[column] = prepared_data[column].astype(object)
        prepared_data[column] = prepared_data[column].where(prepared_data[column].notna(), np.nan)

    return prepared_data


def apply_month_temperature_imputation(
    train_data: pd.DataFrame,
    data_to_transform: pd.DataFrame,
    temperature_column: str = "temperature",
    month_column: str = "month",
) -> pd.DataFrame:
    """Apply month-based temperature imputation using medians learned from training data."""
    transformed_data = data_to_transform.copy()

    monthly_medians = train_data.groupby(month_column)[temperature_column].median()
    global_median = train_data[temperature_column].median()

    month_based_values = transformed_data[month_column].map(monthly_medians)
    fallback_values = month_based_values.fillna(global_median)

    transformed_data[temperature_column] = transformed_data[temperature_column].fillna(fallback_values)

    return transformed_data


def preprocess_kitchen_data(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """Run the full cleaning and feature engineering workflow for the kitchen dataset."""
    selected_columns = feature_columns + [target_column]
    cleaned_data = data[selected_columns].copy()

    cleaned_data = standardize_binary_columns(
        data=cleaned_data,
        columns=["weekend", "holiday", "special_event"],
    )
    cleaned_data = standardize_text_columns(
        data=cleaned_data,
        columns=["day_of_week", "service_type"],
    )
    cleaned_data = clean_weather_column(cleaned_data)
    cleaned_data = clean_temperature_column(cleaned_data)
    cleaned_data = add_time_features(cleaned_data)
    cleaned_data = prepare_data_for_sklearn(cleaned_data)

    return cleaned_data
