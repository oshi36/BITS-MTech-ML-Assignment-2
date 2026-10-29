"""
Dry Bean Classifier — Streamlit app.

Upload the held-out test CSV, choose one of the five saved models, and see its
metrics, confusion matrix and per-class report. Models are loaded from model/
and are only used for prediction; nothing is retrained here.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, recall_score

from bean_utils import CLASSES, FEATURES, MODEL_FILES, TARGET, clean_labels, compute_metrics

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🫘", layout="wide")


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def read_upload(file) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(file)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error("The uploaded CSV is missing required columns: " + ", ".join(missing))
        st.stop()
    df[TARGET] = clean_labels(df[TARGET])
    unknown = sorted(set(df[TARGET]) - set(CLASSES))
    if unknown:
        st.error(f"Unknown class labels in '{TARGET}': {unknown}. Expected one of {CLASSES}.")
        st.stop()
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        st.error("Some feature values are empty or not numeric. Please clean the file and re-upload.")
        st.stop()
    return df, X, df[TARGET]


def plot_confusion(y, y_pred, labels, normalise: bool):
    cm = confusion_matrix(y, y_pred, labels=labels, normalize="true" if normalise else None)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalise else "d", cmap="YlOrBr", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax, annot_kws={"size": 9})
    ax.set_xlabel("Predicted variety")
    ax.set_ylabel("Actual variety")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    return fig


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("🫘 Setup")
    uploaded = st.file_uploader("1. Upload test data (CSV)", type="csv")
    sample = ROOT / "test_data.csv"
    if sample.exists():
        st.download_button("No file? Download sample test_data.csv", sample.read_bytes(),
                           "test_data.csv", "text/csv", use_container_width=True)
    model_name = st.selectbox("2. Pick a classifier", list(MODEL_FILES))
    normalise = st.toggle("Show confusion matrix as row %", value=False)
    st.divider()
    st.caption("Models were tuned with 5-fold CV and trained offline in `train_models.py`. "
               "This app only loads them and predicts.")

# ---------------- Header ----------------
st.title("Dry Bean Variety Classifier")
st.markdown(
    "Seven Turkish dry bean varieties, told apart using **16 shape measurements** extracted from "
    "photos of 13,611 beans (UCI Dry Bean dataset). Precision, Recall and F1 are **macro-averaged** "
    "and AUC is **one-vs-rest**, because this is a 7-class problem."
)

if uploaded is None:
    st.info("⬅️ Upload a CSV with the 16 feature columns and the `Class` column to start. "
            "The sidebar has a sample file you can download and re-upload.")
    ref_path = MODEL_DIR / "metrics.json"
    if ref_path.exists():
        st.subheader("Reference results on the held-out test split")
        ref = pd.DataFrame(json.loads(ref_path.read_text())).T
        ref.index.name = "ML Model Name"
        st.dataframe(ref.style.format("{:.4f}").highlight_max(axis=0, color="#f6d7a7"),
                     use_container_width=True)
    st.stop()

raw, X, y = read_upload(uploaded)
tab1, tab2, tab3 = st.tabs(["🔍 Model results", "⚖️ Compare all five", "📄 Uploaded data"])

# ---------------- Tab 1: selected model ----------------
with tab1:
    model = load_model(model_name)
    labels = list(model.classes_)
    metrics, y_pred, y_prob = compute_metrics(model, X, y)

    st.subheader(f"{model_name} on {len(y):,} beans")
    for col, (k, v) in zip(st.columns(6), metrics.items()):
        col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")
    if pd.isna(metrics["AUC"]):
        st.caption("AUC needs every class to be present in the uploaded file, so it is shown as n/a.")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion(y, y_pred, labels, normalise))
    with right:
        st.markdown("**Recall per variety** (share of each variety identified correctly)")
        per_class = pd.Series(
            recall_score(y, y_pred, labels=labels, average=None, zero_division=0), index=labels
        ).rename("Recall").reset_index().rename(columns={"index": "Variety"})
        st.bar_chart(per_class, x="Variety", y="Recall", horizontal=True, sort="Recall",
                     color="#8d5524")
        worst = per_class.loc[per_class["Recall"].idxmin()]
        st.caption(f"Hardest variety for this model: **{worst['Variety']}** "
                   f"({worst['Recall']:.1%} recalled).")

    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)

    out = raw.copy()
    out["Predicted"] = y_pred
    out["Confidence"] = y_prob.max(axis=1).round(4)
    st.download_button("Download predictions as CSV", out.to_csv(index=False),
                       f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv", "text/csv")

# ---------------- Tab 2: compare ----------------
with tab2:
    st.subheader("Every model on the uploaded file")
    scores, recalls = {}, {}
    for name in MODEL_FILES:
        m = load_model(name)
        s, p, _ = compute_metrics(m, X, y)
        scores[name] = s
        recalls[name] = recall_score(y, p, labels=CLASSES, average=None, zero_division=0)

    comp = pd.DataFrame(scores).T
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#f6d7a7"),
                 use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        pick = st.radio("Metric to chart", list(comp.columns), index=5, horizontal=True)
        chart = comp[[pick]].reset_index()
        st.bar_chart(chart, x="ML Model Name", y=pick, horizontal=True, sort=f"-{pick}",
                     color="#8d5524")
    with c2:
        st.markdown("**Recall per variety, per model**")
        heat = pd.DataFrame(recalls, index=CLASSES).T
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrBr", vmin=0.75, vmax=1, cbar=False,
                    ax=ax, annot_kws={"size": 8})
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    best = comp["MCC"].idxmax()
    st.success(f"Best model on this file by MCC: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f})")

# ---------------- Tab 3: data ----------------
with tab3:
    st.write(f"**{raw.shape[0]:,} rows × {raw.shape[1]} columns**")
    dist = y.value_counts().rename("Beans").reset_index().rename(columns={TARGET: "Variety"})
    st.bar_chart(dist, x="Variety", y="Beans", sort="-Beans", color="#8d5524")
    st.dataframe(raw.head(100), use_container_width=True)
