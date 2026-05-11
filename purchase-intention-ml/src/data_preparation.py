from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "project_data.csv"
RANDOM_STATE = 42


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw customer-session dataset."""
    return pd.read_csv(path, index_col=0)


def drop_missing_target(df: pd.DataFrame, target_column: str = "Revenue") -> pd.DataFrame:
    """Remove rows where the target value is missing."""
    return df.dropna(subset=[target_column]).copy()


def keep_valid_weekend_values(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where Weekend has a valid True/False value."""
    return df[df["Weekend"].isin(["True", "False"])].copy()


def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text True/False columns to numeric 1/0 columns."""
    converted = df.copy()
    boolean_columns = ["Weekend", "Revenue"]

    converted[boolean_columns] = (
        converted[boolean_columns]
        .astype(str)
        .apply(lambda column: column.str.strip().str.lower())
        .replace({"true": 1, "false": 0})
    ).astype("int64")

    return converted


def drop_remaining_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in selected feature columns."""
    columns_with_missing_values = ["Region", "Browser", "SpecialDay"]
    return df.dropna(subset=columns_with_missing_values).copy()


def clean_month_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid month values and standardize month labels."""
    cleaned = df.copy()
    cleaned = cleaned[cleaned["Month"].astype(str).str.strip() != "Turc"].copy()
    cleaned["Month"] = cleaned["Month"].replace("Sept", "Sep")
    return cleaned


def encode_month_cyclically(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month as cyclic features so December and January stay close."""
    encoded = df.copy()
    month_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }

    encoded["Month"] = encoded["Month"].astype(str).str.strip().str[:3]
    encoded["Month_num"] = encoded["Month"].map(month_map)
    encoded = encoded.dropna(subset=["Month_num"]).copy()
    encoded["Month_sin"] = np.sin(2 * np.pi * encoded["Month_num"] / 12)
    encoded["Month_cos"] = np.cos(2 * np.pi * encoded["Month_num"] / 12)
    encoded = encoded.drop(columns=["Month"])
    return encoded


def encode_visitor_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the VisitorType categorical column."""
    return pd.get_dummies(df, columns=["VisitorType"], drop_first=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated rows from the dataset."""
    return df.drop_duplicates().copy()


def remove_invalid_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove impossible numeric values found during notebook exploration."""
    cleaned = df.copy()
    cleaned = cleaned[(cleaned["BounceRates"] >= 0) & (cleaned["BounceRates"] <= 1)]
    cleaned = cleaned[cleaned["Administrative"] >= 0]
    cleaned["Browser"] = cleaned["Browser"].astype(int)
    cleaned["Region"] = cleaned["Region"].astype(int)
    cleaned["Month_num"] = cleaned["Month_num"].astype(int)
    return cleaned.copy()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete data-cleaning and feature-engineering pipeline."""
    cleaned = drop_missing_target(df)
    cleaned = keep_valid_weekend_values(cleaned)
    cleaned = convert_boolean_columns(cleaned)
    cleaned = drop_remaining_missing_values(cleaned)
    cleaned = clean_month_values(cleaned)
    cleaned = encode_month_cyclically(cleaned)
    cleaned = encode_visitor_type(cleaned)
    cleaned = remove_duplicates(cleaned)
    cleaned = remove_invalid_numeric_values(cleaned)
    return cleaned.reset_index(drop=True)


def split_features_and_target(df: pd.DataFrame):
    """Split the cleaned data into model features and target."""
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]
    return X, y


def split_train_validation_test(X, y):
    """Create a 70/15/15 train, validation, and test split."""
    from sklearn.model_selection import train_test_split

    X_train, X_hold, y_train, y_hold = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_hold,
        y_hold,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_hold,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
