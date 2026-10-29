"""
Train, tune and evaluate five classifiers on the UCI Dry Bean dataset.

Run:  python train_models.py
(Cleaning and the train/test split come from preprocess.py; run eda.py first
 if you want the exploratory analysis.)

Outputs
    model/<name>.joblib   fitted Pipeline (StandardScaler + classifier)
    model/metrics.json    test-set metrics for every model
    model/best_params.json  hyperparameters chosen by cross-validation
    test_data.csv         held-out 20% test split (16 features + Class)
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from bean_utils import MODEL_FILES, compute_metrics
from preprocess import SEED, load_clean_data, save_splits, split_data

# SVC(probability=True) is deprecated from sklearn 1.9 but still works until 1.11;
# requirements.txt pins 1.9.1, so hide the notice to keep the output readable.
warnings.filterwarnings("ignore", message=".*probability.*deprecated", category=FutureWarning)

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

# Small, explainable search spaces. Every model is tuned with the same
# 5-fold stratified CV on the training split only, scored on macro F1.
SEARCH = {
    "Logistic Regression": (
        LogisticRegression(max_iter=5000),
        {"clf__C": [0.1, 1, 10, 100]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"clf__max_depth": [6, 8, 10, 14, None], "clf__min_samples_leaf": [1, 5, 10, 20]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 9, 13, 19, 25], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
    ),
    "Support Vector Machine": (
        SVC(kernel="rbf", random_state=SEED),
        {"clf__C": [1, 10, 50], "clf__gamma": ["scale", 0.1]},
    ),
}


def main():
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(X_train, X_test, y_train, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_metrics, best_params = {}, {}

    for name, (clf, grid) in SEARCH.items():
        t0 = time.time()
        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        # SVC needs probability=True for AUC; switch it on only for the final refit
        # because it makes every CV fit about 5x slower.
        if name == "Support Vector Machine":
            best.set_params(clf__probability=True)
            best.fit(X_train, y_train)

        joblib.dump(best, MODEL_DIR / MODEL_FILES[name], compress=3)
        metrics, _, _ = compute_metrics(best, X_test, y_test)
        all_metrics[name] = metrics
        best_params[name] = {
            "params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_macro_f1": round(float(search.best_score_), 4),
        }
        print(f"{name:<24} best={best_params[name]['params']}  "
              f"cvF1={search.best_score_:.4f}  ({time.time() - t0:.0f}s)")

    (MODEL_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    table = pd.DataFrame(all_metrics).T
    table.index.name = "ML Model Name"
    print("\nTest-set results\n" + table.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()