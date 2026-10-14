"""Feature engineering and a leakage-conscious baseline for Online Shoppers CSV.

Usage:
  python feature_engineering.py path/to/online_shoppers_intention.csv

Requires pandas and scikit-learn. PageValues is excluded by default because its
availability before the prediction moment must be established for the use case.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MONTH_NUMBER = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
)}
COUNT_DURATION_PAIRS = [
    ("Administrative", "Administrative_Duration"),
    ("Informational", "Informational_Duration"),
    ("ProductRelated", "ProductRelated_Duration"),
]
CODE_CATEGORIES = ["OperatingSystems", "Browser", "Region", "TrafficType"]


def add_behavior_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add activity summaries without learning any dataset-wide statistics."""
    x = frame.copy()
    for count_col, duration_col in COUNT_DURATION_PAIRS:
        x[f"{count_col}_visited"] = (x[count_col] > 0).astype("int8")
        x[f"{count_col}_seconds_per_page"] = np.divide(
            x[duration_col].to_numpy(dtype=float),
            x[count_col].to_numpy(dtype=float),
            out=np.zeros(len(x), dtype=float),
            where=x[count_col].to_numpy() > 0,
        )
    x["total_pages"] = x[[c for c, _ in COUNT_DURATION_PAIRS]].sum(axis=1)
    x["total_duration"] = x[[d for _, d in COUNT_DURATION_PAIRS]].sum(axis=1)
    x["has_page_value"] = (x["PageValues"] > 0).astype("int8")
    month_num = x["Month"].map(MONTH_NUMBER)
    x["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    x["month_cos"] = np.cos(2 * np.pi * month_num / 12)
    # Keep Month as a category too: cyclic coordinates do not replace month identity.
    return x


def log1p_nonnegative(values):
    """Reduce right skew while preserving valid zeros."""
    return np.log1p(np.maximum(values, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to online_shoppers_intention.csv")
    parser.add_argument("--include-page-values", action="store_true",
                        help="Include PageValues only if available before prediction time")
    parser.add_argument("--drop-exact-duplicates", action="store_true",
                        help="Remove exact duplicate rows; use only if they are invalid records")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if "Revenue" not in df:
        raise ValueError("Required target column 'Revenue' was not found.")
    print(f"Rows/columns: {df.shape}; missing cells: {int(df.isna().sum().sum())}; "
          f"exact duplicates: {int(df.duplicated().sum())}")
    if args.drop_exact_duplicates:
        before = len(df)
        df = df.drop_duplicates().copy()
        print(f"Removed {before - len(df)} exact duplicate rows by explicit request.")

    y = df.pop("Revenue").astype(int)
    X = add_behavior_features(df)
    # Conservative default: avoid potential outcome-related/post-event leakage.
    if not args.include_page_values:
        X = X.drop(columns=["PageValues", "has_page_value"])

    categorical = CODE_CATEGORIES + ["Month", "VisitorType", "Weekend"]
    numeric = [c for c in X.columns if c not in categorical]
    skewed = [c for c in numeric if c in {
        "Administrative", "Administrative_Duration", "Informational",
        "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
        "PageValues", "total_pages", "total_duration",
        "Administrative_seconds_per_page", "Informational_seconds_per_page",
        "ProductRelated_seconds_per_page",
    }]
    other_numeric = [c for c in numeric if c not in skewed]

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    skewed_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("skewed", skewed_pipe, skewed),
        ("numeric", numeric_pipe, other_numeric),
        ("category", categorical_pipe, categorical),
    ])
    model = Pipeline([
        ("features", prep),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, pred))
    print("\nClass report:")
    print(classification_report(y_test, pred, target_names=["no purchase", "purchase"],
                                zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}")
    print(f"Average precision: {average_precision_score(y_test, prob):.4f}")
    print("\nScores are from a random stratified split, not a time-based future holdout.")
    print("Compare this run with a raw-feature baseline before concluding engineering helped.")


if __name__ == "__main__":
    main()
