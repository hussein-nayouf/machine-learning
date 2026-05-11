# Model Summary

## Project Goal

The goal of this project is to predict expected kitchen order volume before service starts.

The selected target is:

```text
total_orders
```

The model uses pre-service information such as date features, day of week, weekend status, holiday status, service type, weather, temperature, and special event status.

## Selected Model

The final selected model is a tuned `RandomForestRegressor`.

The model was selected after comparing several approaches:

- Dummy mean baseline
- Random Forest baseline
- Random Forest with month-based temperature imputation
- Gradient Boosting
- Extra Trees
- Histogram Gradient Boosting
- Tuned Random Forest with GridSearchCV

## Best Hyperparameters

```text
n_estimators: 300
max_depth: 5
min_samples_leaf: 4
min_samples_split: 10
random_state: 42
```

## Final Performance

| Dataset | MAE | RMSE | R² |
|---|---:|---:|---:|
| Train | 23.02 | 28.77 | 0.55 |
| Validation | 26.27 | 33.75 | 0.43 |
| Test | 23.98 | 30.00 | 0.39 |

## Interpretation

The final model predicts test order volume with an average error of about 24 orders.

The model generalizes reasonably well, but it does not explain all variation in demand. Weekend demand and unusual high-demand days are harder to predict with the current features.

## Limitations

The dataset does not include some important demand drivers, such as:

- reservations
- promotions
- local events
- menu changes
- planned business campaigns

Adding these features could improve future model performance.

## Output

The trained model is saved locally as:

```text
models/tuned_random_forest_model.joblib
```
