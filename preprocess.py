"""
Preprocessing for the UCI Dry Bean dataset.

Run on its own:  python preprocess.py
    - cleans the raw data and reports what changed
    - writes data/train.csv and test_data.csv (the 80/20 stratified split)

train_models.py imports load_clean_data() and split_data() from here, so the
same steps are always applied.

Why scaling is NOT done here: StandardScaler is fitted inside each model's
Pipeline in train_models.py. That way it learns its mean/std from the training
fold only (no leakage during cross-validation), and the saved model carries its
own scaler, so the Streamlit app can feed it raw test rows.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from bean_utils import CLASSES, FEATURES, TARGET, clean_labels

ROOT = Path(__file__).parent
RAW_PATH = ROOT / "data" / "Dry_Bean_Dataset.csv"
XLSX_PATH = ROOT / "data" / "Dry_Bean_Dataset.xlsx"
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "test_data.csv"
TEST_SIZE = 0.2
SEED = 7


def ensure_csv() -> None:
    """UCI ships the data as .xlsx; convert it once to CSV if only the xlsx is present."""
    if not RAW_PATH.exists() and XLSX_PATH.exists():
        pd.read_excel(XLSX_PATH).to_csv(RAW_PATH, index=False)  # needs openpyxl
        print(f"Converted {XLSX_PATH.name} -> {RAW_PATH.name}")


def load_clean_data(verbose: bool = True) -> pd.DataFrame:
    """Load the raw CSV and apply every cleaning step. Returns features + target."""
    ensure_csv()
    df = pd.read_csv(RAW_PATH)
    say = print if verbose else (lambda *a, **k: None)
    say(f"Loaded {len(df)} rows x {df.shape[1]} columns from {RAW_PATH.name}")

    # 1. Schema check: every expected column must be present.
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing columns: {missing_cols}")
    df = df[FEATURES + [TARGET]]

    # 2. Standardise class labels (strip spaces, upper-case) and check they are known.
    df[TARGET] = clean_labels(df[TARGET])
    unknown = set(df[TARGET]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unexpected class labels: {unknown}")

    # 3. Force numeric features; anything unparsable becomes NaN and is dropped.
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    n_na = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    say(f"Rows with missing/invalid values dropped: {n_na}")

    # 4. Remove exact duplicates so no bean can sit in both train and test.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    say(f"Exact duplicate rows dropped: {before - len(df)}")

    # 5. Sanity check: physical measurements must be positive.
    bad = int((df[FEATURES] <= 0).any(axis=1).sum())
    if bad:
        df = df[(df[FEATURES] > 0).all(axis=1)].reset_index(drop=True)
    say(f"Rows with non-positive measurements dropped: {bad}")

    say(f"Clean dataset: {len(df)} rows")
    return df


def split_data(df: pd.DataFrame, verbose: bool = True):
    """Stratified 80/20 split so every variety keeps its share in both sets."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    if verbose:
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test) -> None:
    train = X_train.copy()
    train[TARGET] = y_train
    train.to_csv(TRAIN_PATH, index=False)

    test = X_test.copy()
    test[TARGET] = y_test
    test.to_csv(TEST_PATH, index=False)  # this is the file uploaded to the Streamlit app


def main():
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(X_train, X_test, y_train, y_test)

    check = pd.DataFrame({
        "train_share": y_train.value_counts(normalize=True),
        "test_share": y_test.value_counts(normalize=True),
    }).reindex(CLASSES).round(4)
    print("\nClass shares after stratified split (should match):")
    print(check.to_string())
    print(f"\nSaved {TRAIN_PATH.relative_to(ROOT)} and {TEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()