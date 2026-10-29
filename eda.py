"""
Exploratory data analysis for the UCI Dry Bean dataset.

Run:  python eda.py

Prints a summary to the terminal and saves figures plus a text report to eda_outputs/:
    01_class_distribution.png
    02_feature_distributions.png
    03_correlation_heatmap.png
    04_boxplots_by_class.png
    05_major_vs_minor_axis.png
    06_shape_factors_by_class.png
    eda_summary.txt
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from bean_utils import FEATURES, TARGET

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "Dry_Bean_Dataset.csv"
OUT_DIR = ROOT / "eda_outputs"
CLASS_ORDER = ["DERMASON", "SIRA", "SEKER", "HOROZ", "CALI", "BARBUNYA", "BOMBAY"]

sns.set_theme(style="whitegrid")
report_lines: list[str] = []


def log(text: str = "") -> None:
    """Print to terminal and keep a copy for eda_summary.txt."""
    print(text)
    report_lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=120)
    plt.close(fig)
    log(f"[saved] eda_outputs/{name}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()

    # 1. Basic structure ------------------------------------------------------
    section("1. Shape and data types")
    log(f"Rows: {df.shape[0]}, Columns: {df.shape[1]} ({len(FEATURES)} features + target '{TARGET}')")
    log(df.dtypes.value_counts().to_string())
    log("\nFirst 5 rows:")
    log(df.head().to_string())

    # 2. Data quality ---------------------------------------------------------
    section("2. Missing values and duplicates")
    missing = df.isna().sum()
    log(f"Total missing values: {int(missing.sum())}")
    if missing.sum():
        log(missing[missing > 0].to_string())
    dups = df.duplicated().sum()
    log(f"Exact duplicate rows: {dups} ({dups / len(df):.2%} of data)")
    log("Duplicate rows will be dropped in preprocess.py so the same bean "
        "cannot appear in both train and test.")
    dup_by_class = df[df.duplicated(keep=False)][TARGET].value_counts()
    if not dup_by_class.empty:
        log("Rows involved in duplication, by class:")
        log(dup_by_class.to_string())

    # 3. Target distribution --------------------------------------------------
    section("3. Class distribution")
    counts = df[TARGET].value_counts().reindex(CLASS_ORDER)
    dist = pd.DataFrame({"count": counts, "share": (counts / len(df)).round(4)})
    log(dist.to_string())
    log(f"Imbalance ratio (largest / smallest class): {counts.max() / counts.min():.1f}x")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=counts.index, y=counts.values, color="#8d5524", ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 40, str(v), ha="center", fontsize=9)
    ax.set(title="Beans per variety", xlabel="Variety", ylabel="Count")
    save(fig, "01_class_distribution.png")

    # 4. Feature statistics ---------------------------------------------------
    section("4. Summary statistics")
    desc = df[FEATURES].describe().T
    desc["skew"] = df[FEATURES].skew()
    log(desc[["mean", "std", "min", "max", "skew"]].round(4).to_string())
    ratio = df["Area"].max() / df["ShapeFactor2"].max()
    log(f"\nFeature scales differ hugely (max Area / max ShapeFactor2 = {ratio:,.0f}x), "
        "so distance- and margin-based models (KNN, SVM, LR) need standardisation.")
    skewed = desc["skew"][desc["skew"].abs() > 1].sort_values(key=abs, ascending=False)
    log(f"Strongly skewed features (|skew| > 1): {', '.join(skewed.index)}")

    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.histplot(df[col], bins=40, ax=ax, color="#8d5524")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Feature distributions", fontsize=14)
    save(fig, "02_feature_distributions.png")

    # 5. Correlations ---------------------------------------------------------
    section("5. Feature correlations")
    corr = df[FEATURES].corr()
    upper = corr.where(~pd.DataFrame(
        [[i >= j for j in range(len(FEATURES))] for i in range(len(FEATURES))],
        index=FEATURES, columns=FEATURES))
    top_pairs = upper.stack().abs().sort_values(ascending=False).head(10)
    log("Top 10 most correlated feature pairs (|r|):")
    for (a, b), r in top_pairs.items():
        log(f"  {a:<16} vs {b:<16} {corr.loc[a, b]:+.4f}")
    log("Near-perfect correlations mean several features carry the same information. "
        "This breaks Naive Bayes' independence assumption.")

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlation matrix of the 16 features")
    save(fig, "03_correlation_heatmap.png")

    # 6. Class separation -----------------------------------------------------
    section("6. How the varieties differ")
    means = df.groupby(TARGET)[["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
                                "AspectRation", "roundness", "Compactness"]].mean()
    log("Per-class means of key features (sorted by Area):")
    log(means.sort_values("Area").round(3).to_string())
    log("BOMBAY is far larger than every other variety, so it should be easy to classify. "
        "DERMASON/SIRA and BARBUNYA/CALI have similar sizes and are likely to be confused.")

    key = ["Area", "Perimeter", "AspectRation", "roundness", "Compactness", "ShapeFactor1"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), key):
        sns.boxplot(data=df, x=TARGET, y=col, order=CLASS_ORDER, ax=ax,
                    color="#f3e9dc", fliersize=1)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
    fig.suptitle("Key features by variety", fontsize=14)
    save(fig, "04_boxplots_by_class.png")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.scatterplot(data=df, x="MajorAxisLength", y="MinorAxisLength", hue=TARGET,
                    hue_order=CLASS_ORDER, s=6, alpha=0.6, ax=ax)
    ax.set_title("Major vs minor axis length: overlapping clusters show hard pairs")
    ax.legend(markerscale=3, fontsize=8)
    save(fig, "05_major_vs_minor_axis.png")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, col in zip(axes, ["ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]):
        sns.violinplot(data=df, x=TARGET, y=col, order=CLASS_ORDER, ax=ax,
                       color="#f6d7a7", inner=None, cut=0)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    save(fig, "06_shape_factors_by_class.png")

    # 7. Takeaways ------------------------------------------------------------
    section("7. Key takeaways for modelling")
    log(f"- {dups} duplicate rows -> drop before splitting.")
    log("- No missing values -> no imputation needed.")
    log(f"- Moderate imbalance ({counts.max() / counts.min():.1f}x) -> stratified split, "
        "macro-averaged metrics, MCC.")
    log("- Very different feature scales -> StandardScaler inside each model pipeline.")
    log("- Heavy multicollinearity -> expect Naive Bayes to suffer; tree and SVM less affected.")
    log("- BOMBAY clearly separable; DERMASON vs SIRA expected to be the main source of error.")

    (OUT_DIR / "eda_summary.txt").write_text("\n".join(report_lines))
    print(f"\nFull report written to {OUT_DIR / 'eda_summary.txt'}")


if __name__ == "__main__":
    main()