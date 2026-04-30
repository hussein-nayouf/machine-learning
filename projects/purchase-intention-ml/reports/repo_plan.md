# Repository Plan

This file explains how the original notebook was converted into a professional GitHub project.

## 1. Raw Data

**Folder:** `data/raw/`  
**File:** `data/raw/project_data.csv`

The raw CSV is stored here and should not be edited manually. All changes to the data should happen through Python code so the project is reproducible.

## 2. Data Preparation

**File:** `src/data_preparation.py`

This file contains the notebook code that prepares the dataset:

- Load `project_data.csv`.
- Drop rows with missing `Revenue`.
- Keep only valid `Weekend` values.
- Convert `Revenue` and `Weekend` to `1` and `0`.
- Drop remaining missing values in `Region`, `Browser`, and `SpecialDay`.
- Remove invalid `Month == "Turc"` rows.
- Replace `Sept` with `Sep`.
- Encode months using cyclic sine and cosine features.
- One-hot encode `VisitorType`.
- Remove duplicated rows.
- Remove impossible numeric values.
- Split the data into train, validation, and test sets.

## 3. Model Training

**File:** `src/model_training.py`

This file contains the final model from the notebook:

```text
SMOTE + GradientBoostingClassifier
```

SMOTE handles the imbalanced target variable. Gradient Boosting is the final classifier selected after comparing different model options in the notebook.

## 4. Prediction

**File:** `src/predict.py`

This file loads the saved model and creates predictions for a CSV file. It is useful because a real project should not only train a model; it should also be able to use the model.

## 5. Reports

**Folder:** `reports/`

This folder stores project documentation and generated results such as:

- metrics
- predictions
- plots
- explanations

## 6. Models

**Folder:** `models/`

This folder stores trained model files such as:

```text
purchase_intention_model.joblib
```

Model files can become large, so `.gitignore` prevents `.joblib` files from being committed by default.
