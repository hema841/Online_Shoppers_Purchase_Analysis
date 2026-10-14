# Online Shoppers Purchasing Intention: Feature Engineering Case Study

## Project overview

This case study examines the Online Shoppers Purchasing Intention dataset and asks whether adding derived session-behavior features improves purchase prediction.

The target column is `Revenue`:

- `True`: the session resulted in a purchase.
- `False`: the session did not result in a purchase.

The original CSV was **not modified**. New features were created in the analysis code and used for model training.

## Original dataset

The supplied CSV contains **12,330 sessions and 18 columns**: 17 input features and the `Revenue` target. It has no missing values and contains 125 exact duplicate rows. The target is imbalanced: approximately 15.5% of sessions resulted in a purchase.

The features describe page counts and browsing durations, bounce and exit rates, page values, special-day closeness, month, operating system, browser, region, traffic type, visitor type, and whether a session occurred on a weekend.

## Limitations identified

- Purchase sessions are the minority class, so accuracy alone can be misleading.
- Several count and duration features are highly skewed and contain many zero values.
- `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are integer category codes, not numeric measurements.
- There are 125 exact duplicate rows. Identical feature values do not prove that the sessions are erroneous duplicates, so they were not automatically removed.
- `PageValues` may not be suitable for prediction if it is unavailable at the time a prediction must be made. It was excluded from both models for this comparison.
- The data has no session date field, so evaluation used a random stratified split rather than testing on later sessions.

## Feature engineering applied

The code added these derived features:

1. **Visited flags**: whether a session included Administrative, Informational, or ProductRelated pages.
2. **Seconds per page**: duration divided by page count for each page type.
3. **Total pages**: sum of the three page counts.
4. **Total duration**: sum of the three page durations.
5. **Cyclical month features**: `month_sin` and `month_cos`, to represent neighboring months as close together across the year boundary.

The code also one-hot encoded category codes so the model would not treat them as ordered numbers. These features were created in memory for analysis; they were not saved back into the original CSV.

## Comparison method

Two logistic-regression models were compared:

- **Raw-feature baseline:** uses the original input features.
- **Engineered-feature model:** uses the same input features plus the derived features listed above.

Both models used the same stratified 80/20 train/test split, class weighting, and preprocessing approach. `PageValues` was excluded from both models.

## Results

| Metric | Raw-feature baseline | Engineered-feature model |
|---|---:|---:|
| Purchase precision | 0.259 | 0.261 |
| Purchase recall | 0.749 | 0.746 |
| Purchase F1 | 0.385 | 0.386 |
| ROC-AUC | 0.735 | 0.741 |
| Average precision | 0.321 | 0.314 |

## Conclusion

Feature engineering produced a small increase in ROC-AUC, while average precision decreased slightly. Purchase precision, recall, and F1 changed very little. Therefore, this test split does **not** show a clear overall improvement from the engineered features.

The original CSV remains unchanged. The new features are derived columns created by the code. Results should be treated as an initial comparison; additional validation, ideally using newer sessions, would be needed before selecting a model for real use.

## Source

The dataset is the Online Shoppers Purchasing Intention dataset from the [UCI Machine Learning Repository](https://www.archive.ics.uci.edu/dataset/468/online%20shoppers%20purchasing%20intention%20dataset).

## How to run

Upload `feature_engineering.py` and `online_shoppers_intention.csv` to Google Colab, then run:

```python
!python /content/feature_engineering.py /content/online_shoppers_intention.csv
```

The script prints data-quality information and evaluation metrics.
