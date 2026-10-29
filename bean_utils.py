"""Shared constants and metric code used by both training and the Streamlit app."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "Class"
FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
    "ShapeFactor4",
]
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def clean_labels(series: pd.Series) -> pd.Series:
    """Normalise class names so 'Sira', ' sira ' and 'SIRA' all match."""
    return series.astype(str).str.strip().str.upper()


def compute_metrics(model, X: pd.DataFrame, y: pd.Series):
    """Return the six assignment metrics plus predictions and probabilities.

    Multiclass handling:
      - Precision / Recall / F1 use macro averaging (every bean variety counts equally,
        so the small BOMBAY class is not drowned out by DERMASON).
      - AUC is one-vs-rest, macro averaged, from predict_proba.
      - MCC uses sklearn's multiclass generalisation.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    labels = list(model.classes_)

    try:
        auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        # Happens if the uploaded file contains only some of the classes.
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        # No labels= here: sklearn then averages over the classes that actually appear in
        # y or y_pred, so a file holding only some varieties is not penalised for the rest.
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return {k: float(np.round(v, 4)) for k, v in metrics.items()}, y_pred, y_prob
