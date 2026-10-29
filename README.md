# 🫘 Dry Bean Variety Classifier - ML Assignment 2

Five classifiers trained on the UCI Dry Bean dataset to identify one of seven dry bean varieties from shape measurements, compared on six metrics and served through an interactive Streamlit app.

- **Live Streamlit app:** https://2026aj05108-ml-assignment-2.streamlit.app/ 
- **GitHub repository:** https://github.com/oshi36/BITS-MTech-ML-Assignment-2

---

## a. Problem statement

Dry beans are among the most widely grown pulses, and seed lots of different varieties look very similar. Sorting them by hand is slow and inconsistent. If each bean is photographed and its shape measured automatically, a classifier can identify its variety.

The task is to predict a bean's **variety (one of 7 classes)** from **16 numerical shape features** extracted from its image. This is a **multi-class classification** problem. Five models are compared on the same train/test split: Logistic Regression, Decision Tree, KNN, Naive Bayes and SVM.

## b. Dataset description

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository — *Dry Bean Dataset* (id 602), Koklu & Ozkan (2020) |
| Dataset link | https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| DOI | https://doi.org/10.24432/C50S4B |
| How it was made | 13,611 beans photographed with a high-resolution camera; each bean segmented from the image and 16 shape features computed |
| Instances | 13,611 raw → **13,543 after removing 68 exact duplicate rows** |
| Features | 16, all continuous numeric |
| Target | `Class` — 7 varieties |
| Missing values | None |

**Features**

| Group | Features |
|---|---|
| Size (12 dimensional features) | `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRation`*, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `roundness`, `Compactness` |
| Shape forms (4) | `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4` |


**Class distribution (raw data)**

| Variety | Beans | Share |
|---|---|---|
| DERMASON | 3,546 | 26.1% |
| SIRA | 2,636 | 19.4% |
| SEKER | 2,027 | 14.9% |
| HOROZ | 1,928 | 14.2% |
| CALI | 1,630 | 12.0% |
| BARBUNYA | 1,322 | 9.7% |
| BOMBAY | 522 | 3.8% |

The classes are moderately imbalanced: DERMASON is about 7 times as common as BOMBAY.

**Notable properties of the data**
- **Strong multicollinearity.** `Area`, `ConvexArea`, `Perimeter,` and `EquivDiameter` are almost the same signal: the correlation between `Area` and `ConvexArea` is 1.000, and the others are 0.97–0.99. `Compactness` and `ShapeFactor3` correlate at 0.999.
- **BOMBAY is far larger than the rest.** Its mean area is about 173k pixels, against 32k–76k for the other six varieties, so it is easy to separate.
- **DERMASON and SIRA overlap heavily**, as do BARBUNYA and CALI. These pairs cause most misclassifications.

**Exploratory analysis (`eda.py`).** Reports the class balance, missing values, duplicates, skewness, the top correlated feature pairs and per-class feature means. It saves six figures to `eda_outputs/`: class distribution, feature histograms, correlation heatmap, box plots by variety, a major-vs-minor-axis scatter, and shape-factor violin plots. The scatter plot shows BOMBAY as an isolated cluster, while DERMASON/SIRA and BARBUNYA/CALI overlap.

**Preprocessing and protocol (`preprocess.py`)**
1. Checked the schema, standardised the class labels, coerced all features to numeric, and checked for missing or non-positive values (none were found).
2. Dropped 68 exact duplicate rows, so no identical bean can appear in both train and test.
3. Made a stratified 80/20 train/test split (`random_state=7`): 10,834 training rows (`data/train.csv`) and **2,709 test rows** (`test_data.csv`).
4. Applied `StandardScaler` to all features inside each model's `Pipeline` in `train_models.py`, not in `preprocess.py`. The scaler then learns only from training data (no leakage), and each saved model carries its own scaler for the app. Scaling matters for KNN, SVM and Logistic Regression, because `Area` (tens of thousands) would otherwise swamp `ShapeFactor2` (around 0.002).
5. Tuned hyperparameters with `GridSearchCV`, using 5-fold stratified CV on the **training split only**, scored on macro F1. The test split was used exactly once, for the final scores.

**Metric definitions for a 7-class problem.** Precision, Recall and F1 are **macro-averaged**, so each variety counts equally and the small BOMBAY class carries the same weight as DERMASON. AUC is **one-vs-rest, macro-averaged**, computed from `predict_proba`. MCC uses sklearn's multiclass generalisation.

## c. GitHub repository link

`https://github.com/oshi36/BITS-MTech-ML-Assignment-2`

```
drybean-classifier/
├── app.py                    # Streamlit app — loads saved models, never retrains
├── bean_utils.py             # Shared feature list, class list and metric function
├── eda.py                    # Exploratory data analysis → eda_outputs/
├── preprocess.py             # Cleaning, dedup, stratified split → data/train.csv, test_data.csv
├── train_models.py           # CV tuning, training, evaluation, model export
├── requirements.txt
├── README.md
├── test_data.csv             # Held-out 20% test split (16 features + Class)
├── .streamlit/config.toml    # App theme
├── data/
│   ├── Dry_Bean_Dataset.csv  # Original UCI data (converted from .xlsx)
│   └── train.csv             # 80% training split
├── eda_outputs/              # EDA figures + eda_summary.txt
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── metrics.json          # Test-set metrics
    └── best_params.json      # Hyperparameters chosen by CV
```

**Run locally**
```bash
pip install -r requirements.txt
python eda.py                 # 1. exploratory analysis, figures saved to eda_outputs/
python preprocess.py          # 2. clean + split, writes data/train.csv and test_data.csv
python train_models.py        # 3. tune, train, evaluate, save models to model/
streamlit run app.py          # 4. launch the app
```

## d. Models used

| Model | Search space (5-fold CV) | Selected | CV macro-F1 |
|---|---|---|---|
| Logistic Regression | `C` ∈ {0.1, 1, 10, 100} | `C=100` | 0.9352 |
| Decision Tree | `max_depth` ∈ {6, 8, 10, 14, None}, `min_samples_leaf` ∈ {1, 5, 10, 20} | `max_depth=8`, `min_samples_leaf=1` | 0.9223 |
| KNN | `n_neighbors` ∈ {5, 9, 13, 19, 25}, `weights` ∈ {uniform, distance} | `k=9`, `distance` | 0.9363 |
| Naive Bayes (Gaussian) | `var_smoothing` ∈ {1e-9, 1e-7, 1e-5, 1e-3} | `1e-3` | 0.9049 |
| SVM (RBF kernel) | `C` ∈ {1, 10, 50}, `gamma` ∈ {scale, 0.1} | `C=10`, `gamma=scale` | 0.9426 |

### Comparison table (held-out test set, 2,709 beans)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9247 | 0.9946 | 0.9368 | 0.9360 | 0.9364 | 0.9089 |
| Decision Tree | 0.9158 | 0.9791 | 0.9298 | 0.9244 | 0.9270 | 0.8981 |
| KNN | 0.9258 | 0.9886 | 0.9392 | 0.9366 | 0.9378 | 0.9102 |
| Naive Bayes | 0.9003 | 0.9913 | 0.9105 | 0.9103 | 0.9098 | 0.8798 |
| Support Vector Machine | **0.9324** | **0.9953** | **0.9445** | **0.9433** | **0.9439** | **0.9183** |

**Recall per variety on the test set**

| Model | BARBUNYA | BOMBAY | CALI | DERMASON | HOROZ | SEKER | SIRA |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.925 | 1.000 | 0.933 | 0.920 | 0.949 | 0.956 | 0.871 |
| Decision Tree | 0.894 | 0.990 | 0.926 | 0.929 | 0.935 | 0.941 | 0.854 |
| KNN | 0.913 | 1.000 | 0.945 | 0.920 | 0.949 | 0.953 | 0.877 |
| Naive Bayes | 0.804 | 1.000 | 0.914 | 0.883 | 0.957 | 0.943 | 0.871 |
| SVM | 0.943 | 1.000 | 0.945 | 0.928 | 0.946 | 0.958 | 0.882 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model: accuracy 0.925, MCC 0.909, and the second-best AUC (0.995). Its ranking of classes is excellent, which suggests most varieties are close to linearly separable once the features are scaled. CV picked the weakest regularisation (`C=100`), so the model wanted all the detail in the correlated size features. Most of its errors are DERMASON ↔ SIRA (87 of its 204 errors), where the two varieties' shapes overlap. |
| Decision Tree | The weakest of the four strong models (accuracy 0.916, MCC 0.898) and the lowest AUC (0.979). A tree's probabilities come from leaf proportions, so they are coarse, which lowers AUC. Axis-aligned splits approximate the smooth DERMASON/SIRA boundary poorly: SIRA recall is only 0.854, the lowest of any model for that class. Its strength is interpretability: `ShapeFactor1`, `ShapeFactor3` and `MajorAxisLength` carry about 77% of the importance. The tuned tree has a depth of 8 and 163 leaves. |
| KNN | Nearly tied with Logistic Regression and second overall by MCC (0.910). KNN does well here because the 16 features were standardised and the beans of each variety form compact clusters in feature space. CV chose `k=9` with distance weighting, which smooths noise without blurring class boundaries. It makes the second-fewest errors (201), but its AUC (0.989) is below LR and SVM because neighbour-vote probabilities are coarse. |
| Naive Bayes | The weakest model (accuracy 0.900, MCC 0.880). That is still a respectable result considering its assumption that features are independent of each other is badly broken: `Area`, `ConvexArea`, `Perimeter` and `EquivDiameter` are almost perfectly correlated, so the model effectively counts the same size evidence four times. This hurts most on BARBUNYA (recall 0.804, with 38 beans misclassified as CALI) and DERMASON (0.883). Its AUC is still high (0.991): it ranks classes well, but its over-confident probabilities give poorly placed decision boundaries. |
| Support Vector Machine | **Best on every metric**: accuracy 0.932, AUC 0.995, macro-F1 0.944, MCC 0.918, and the fewest errors (183 of 2,709). The RBF kernel draws curved boundaries, which helps most where classes overlap: it has the best BARBUNYA (0.943) and SIRA (0.882) recall of any model. The result matches the original paper, where an SVM was also the best model at about 93% accuracy. |
| **Overall winner for this dataset** | **Support Vector Machine (RBF, C=10).** It leads on all six metrics, and on the hardest classes. The gap is small (MCC 0.918 vs 0.910 for KNN), so KNN or Logistic Regression would be reasonable lighter-weight alternatives. Every model recognises BOMBAY perfectly, and every model's main weakness is the DERMASON ↔ SIRA pair. That suggests the limit is the shape features themselves, not the choice of algorithm. |

## Streamlit app

1. **CSV upload** of test data, validated for required columns, numeric values and known class labels. The sidebar has a download button for the sample `test_data.csv`.
2. **Model selection dropdown** for the five saved models.
3. **Evaluation metrics**: Accuracy, AUC, Precision, Recall, F1 and MCC as metric tiles.
4. **Confusion matrix** (7×7, toggle between counts and row percentages), **classification report**, and recall per variety.
5. **Comparison tab** that scores all five models on the uploaded file, with a metric bar chart and a recall heatmap of model × variety.
6. **Download of predictions** with a confidence score for each bean.

The app only loads the saved `.joblib` pipelines and calls `predict` / `predict_proba`. Nothing is retrained on uploaded data.

## Reference

- **Dataset:** Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/602/dry+bean+dataset - DOI: https://doi.org/10.24432/C50S4B
