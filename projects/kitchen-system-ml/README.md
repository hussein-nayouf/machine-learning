# Kitchen Order Prediction ML System

Machine learning project for predicting expected kitchen order volume from pre-service restaurant information.

The model predicts `total_orders`, which helps estimate kitchen workload before service starts.


## Motivation

This project is inspired by my own experience working as a chef. In restaurant kitchens, one common challenge is planning the right amount of staff for each service. If the kitchen is understaffed, service becomes stressful and slower. If it is overstaffed, the restaurant uses more labor than needed.

I chose to build this kitchen machine learning system because I wanted to explore how data can help restaurants predict expected order volume and make better staffing and preparation decisions before service starts.



## Project Goal

The goal is to answer:

**How many orders should the kitchen expect for a given day and service type?**

This can support:

- kitchen workload planning
- food preparation planning
- ingredient and stock planning
- indirect staffing decisions

## Project Layout

```text
.
├── config/             # Project configuration
├── data/
│   ├── raw/            # Original dataset files
│   └── processed/      # Optional processed data outputs
├── models/             # Trained model artifacts
├── notebooks/          # Exploration and model development notebooks
├── reports/            # Figures, metrics, and notes
└── src/ml_project/     # Reusable Python package code
```

## Dataset

The raw dataset is stored in:

```text
data/raw/kitchen_ml_dataset.xlsx
```

The target column is:

```text
total_orders
```

The model uses pre-service features such as:

- date-derived features
- day of week
- weekend status
- holiday status
- service type
- weather
- temperature
- special event status

Columns that may create data leakage, such as revenue, customer counts, and items sold, are excluded from the model.

## Modeling Approach

The final selected model is a tuned `RandomForestRegressor`.

The workflow includes:

1. exploratory data analysis
2. data cleaning
3. feature engineering
4. baseline model comparison
5. validation error analysis
6. GridSearchCV hyperparameter tuning
7. final test evaluation

The final model uses month-based temperature imputation to make missing temperature values more realistic.

## Final Model Performance

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| Train | 23.02 | 28.77 | 0.55 |
| Validation | 26.27 | 33.75 | 0.43 |
| Test | 23.98 | 30.00 | 0.39 |

The final model predicts test order volume with an average error of about **24 orders**.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the Model

From the project root, run:

```bash
PYTHONPATH=src python -m ml_project.train
```

The trained model package is saved to:

```text
models/tuned_random_forest_model.joblib
```

The saved package includes:

- the trained model
- model feature names
- numeric and categorical feature lists
- target column name
- monthly temperature medians learned from training data
- global temperature median learned from training data

## Run an Example Prediction

After training the model, run:

```bash
PYTHONPATH=src python -m ml_project.predict
```

This loads the saved model package and predicts expected total orders for an example service.

Example output:

```text
date       day_of_week  service_type  predicted_total_orders
2026-05-16 Saturday     dinner        138
```

## Reports

A model summary is available in:

```text
reports/model_summary.md
```

## Notes

The model is a useful first version, but prediction errors are higher on weekends and unusual high-demand days.

Future improvements could include stronger demand signals such as:

- reservations
- promotions
- local events
- menu changes
- planned business campaigns
