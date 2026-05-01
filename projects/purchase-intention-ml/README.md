# Purchase Intention ML

This project predicts whether an e-commerce customer session will lead to a purchase.

The project started as a Jupyter notebook and was converted into a professional machine-learning repository.

## Problem Type

This is a binary classification project:

- `Revenue = 1`: the customer session led to a purchase
- `Revenue = 0`: the customer session did not lead to a purchase

## Project Structure

```text
purchase-intention-ml/
├── data/
│   └── raw/
│       └── project_data.csv
├── models/
├── notebooks/
│   ├── 01_purchase_intention_analysis_en.ipynb
│   └── README.md
├── reports/
│   ├── metrics.json
│   └── repo_plan.md
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── predict.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/model_training.py
```

Generate predictions:

```bash
python src/predict.py
```

## Final Model

The final model follows the best notebook result:

```text
SMOTE + GradientBoostingClassifier
```

SMOTE is used because the target variable is imbalanced: most sessions do not lead to purchases.

## Results

The final model was evaluated on the test set.

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8843 |
| Precision | 0.6033 |
| Recall | 0.7480 |
| F1-score | 0.6679 |
| ROC-AUC | 0.9268 |

Confusion matrix:

|  | Predicted No Purchase | Predicted Purchase |
| --- | ---: | ---: |
| Actual No Purchase | 1215 | 121 |
| Actual Purchase | 62 | 184 |

## Interpretation

The model finds about 74.8% of the sessions that actually lead to purchases. This is useful in a business setting where the goal is to identify likely buyers early and support marketing, personalization, or customer-support decisions.

The precision is lower than recall, which means some sessions predicted as purchases will not actually convert. This tradeoff can still be acceptable if the business prefers finding more potential buyers over missing them.

## What I Learned

This project demonstrates how to:

- clean and validate raw data
- encode categorical and cyclic features
- handle imbalanced classification with SMOTE
- train and evaluate a classification model
- save a trained model for later prediction
- document a project clearly for GitHub
