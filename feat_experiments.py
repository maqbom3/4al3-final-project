#!/usr/bin/env python
# coding: utf-8

import os, random, seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    make_scorer,
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier



file = Path("ObesityDataSet_raw_and_data_sinthetic.csv")

df = pd.read_csv(file)

# Drop missing values (same as before)
df = df.dropna()

# Remove BMI-related features
df = df.drop(columns=["Height", "Weight"])

# Rename target for clarity
df = df.rename(columns={"NObeyesdad": "Obesity_levels"})

print("Data shape after cleaning:", df.shape)
print("Columns:", df.columns.tolist())
print(df["Obesity_levels"].value_counts())


X = df.drop(columns=["Obesity_levels"])
y = df["Obesity_levels"]
#train split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=42,
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)



cat_cols = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

num_cols = [
    "Age",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
]
#preprocess
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        ("num", StandardScaler(with_mean=False), num_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)


# Helper to recover feature names after preprocessing
def get_feature_names(preprocessor):
    """
    Retrieve feature names from a fitted ColumnTransformer (preprocess).
    """
    output_features = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "get_feature_names_out"):
            # e.g. OneHotEncoder
            names = transformer.get_feature_names_out(cols)
            output_features.extend(names)
        else:
            # e.g. StandardScaler on numeric columns
            output_features.extend(cols)

    return output_features


#dummy baseline
scoring = {
    "f1_macro": "f1_macro",
    "balanced_accuracy": make_scorer(balanced_accuracy_score),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dummy baseline (most frequent)
dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", dummy_clf),
    ]
)

# Multinomial Logistic Regression model
logreg_clf = LogisticRegression(
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=5000,
    class_weight="balanced",
)
logreg_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", logreg_clf),
    ]
)

# Cross-validated performance
dummy_results = cross_validate(dummy_pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
logreg_results = cross_validate(logreg_pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

print("\n==== BASELINES (5-fold CV) ====")
print(
    f"Dummy (most_frequent)  | "
    f"F1-macro: {dummy_results['test_f1_macro'].mean():.3f} ± {dummy_results['test_f1_macro'].std():.3f} | "
    f"BalAcc: {dummy_results['test_balanced_accuracy'].mean():.3f} ± {dummy_results['test_balanced_accuracy'].std():.3f}"
)
print(
    f"Logistic Regression    | "
    f"F1-macro: {logreg_results['test_f1_macro'].mean():.3f} ± {logreg_results['test_f1_macro'].std():.3f} | "
    f"BalAcc: {logreg_results['test_balanced_accuracy'].mean():.3f} ± {logreg_results['test_balanced_accuracy'].std():.3f}"
)



# Fit on full X,y for importance analysis
preprocess.fit(X, y)
X_encoded = preprocess.transform(X)

feature_names = get_feature_names(preprocess)
print("\nNumber of encoded features:", len(feature_names))
print("First 10 feature names:", feature_names[:10])



# 7. Feature Importance Experiment 1: Mutual Information

mi_scores = mutual_info_classif(
    X_encoded, y, discrete_features="auto", random_state=42
)

mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)

print("\n==== TOP 15 FEATURES BY MUTUAL INFORMATION ====")
print(mi_series.head(15))

plt.figure(figsize=(8, 10))
mi_series.head(20).sort_values().plot(kind="barh")
plt.title("Top 20 Features by Mutual Information")
plt.xlabel("Mutual Information")
plt.tight_layout()
plt.show()


# 8. Feature Importance Experiment 2: L1-Regularized Logistic Regression

l1_clf = LogisticRegression(
    penalty="l1",
    solver="saga",
    multi_class="multinomial",
    class_weight="balanced",
    max_iter=5000,
)

l1_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", l1_clf),
    ]
)

l1_pipe.fit(X_train, y_train)

# Coefficients shape: (n_classes, n_features)
coefs = l1_pipe.named_steps["model"].coef_
importance_l1 = np.mean(np.abs(coefs), axis=0)  # average over classes

l1_series = pd.Series(importance_l1, index=feature_names).sort_values(ascending=False)

print("\n==== TOP 15 FEATURES BY L1-LOGISTIC COEFFICIENT MAGNITUDE ====")
print(l1_series.head(15))

plt.figure(figsize=(8, 10))
l1_series.head(20).sort_values().plot(kind="barh")
plt.title("Top 20 Features by L1 Logistic Regression (|coef|)")
plt.xlabel("Average |Coefficient| Across Classes")
plt.tight_layout()
plt.show()


# Optional: performance of L1 logistic regression vs original
l1_cv_results = cross_validate(l1_pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
print(
    "\nL1-Logistic Regression | "
    f"F1-macro: {l1_cv_results['test_f1_macro'].mean():.3f} ± {l1_cv_results['test_f1_macro'].std():.3f} | "
    f"BalAcc: {l1_cv_results['test_balanced_accuracy'].mean():.3f} ± {l1_cv_results['test_balanced_accuracy'].std():.3f}"
)


#Feature Importance Experiment 3: ExtraTreesClassifier

tree_clf = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

tree_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", tree_clf),
    ]
)

tree_pipe.fit(X_train, y_train)

tree_importances = tree_pipe.named_steps["model"].feature_importances_

tree_series = pd.Series(tree_importances, index=feature_names).sort_values(
    ascending=False
)

print("\nTOP 15 FEATURES BY EXTRATREES IMPORTANCE")
print(tree_series.head(15))

plt.figure(figsize=(8, 10))
tree_series.head(20).sort_values().plot(kind="barh")
plt.title("Top 20 Features by ExtraTrees Importance")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()


# compare features across methods

top_k = 15
top_mi = set(mi_series.head(top_k).index)
top_l1 = set(l1_series.head(top_k).index)
top_tree = set(tree_series.head(top_k).index)

print("\n==== OVERLAP OF TOP FEATURES (k = 15) ====")
print("MI ∩ L1:", top_mi & top_l1)
print("MI ∩ Tree:", top_mi & top_tree)
print("L1 ∩ Tree:", top_l1 & top_tree)
print("MI ∩ L1 ∩ Tree:", top_mi & top_l1 & top_tree)
