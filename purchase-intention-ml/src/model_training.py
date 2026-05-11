import json
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_preparation import (
    RANDOM_STATE,
    clean_data,
    load_data,
    split_features_and_target,
    split_train_validation_test,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "purchase_intention_model.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


def build_model() -> Pipeline:
    """Build the final model selected from the notebook experiments."""
    return Pipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "model",
                GradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=3,
                    n_estimators=100,
                    subsample=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(model, X, y) -> dict:
    """Calculate the classification metrics used in the notebook."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": round(accuracy_score(y, predictions), 4),
        "precision": round(precision_score(y, predictions), 4),
        "recall": round(recall_score(y, predictions), 4),
        "f1": round(f1_score(y, predictions), 4),
        "roc_auc": round(roc_auc_score(y, probabilities), 4),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(y, predictions),
    }


def train_final_model() -> dict:
    """Train the final model, evaluate it, and save outputs."""
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    X, y = split_features_and_target(cleaned_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(X, y)

    model = build_model()
    model.fit(X_train, y_train)

    validation_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": model, "feature_columns": list(X_train.columns)},
        MODEL_PATH,
    )

    metrics = {
        "validation": validation_metrics,
        "test": test_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    results = train_final_model()
    print(json.dumps(results["test"], indent=2))
