# Comprehensive Technical Report: Clinical Feature Engineering, 18-Experiment Benchmarking, and Visual Analysis for Intradialytic Hypotension (IDH) Prediction

## 1. Introduction & Clinical Problem Statement
Intradialytic Hypotension (IDH) is one of the most frequent and dangerous complications during hemodialysis sessions, characterized by a sharp drop in blood pressure that can lead to myocardial ischemia, arrhythmia, and organ hypoperfusion. 

Our clinical AI system aims to predict IDH crash events prior to occurrence. Analysis of the Mendeley hemodialysis benchmark dataset (`dataset_10.csv`, 13,637 sessions) reveals a severe class imbalance: **79.61% of sessions are clinically stable (`0`)**, while **20.39% result in an IDH crash (`1`)**. In medical informatics, standard models trained on imbalanced data suffer from the "accuracy trap," yielding high accuracy by simply guessing the majority class while missing minority crash events (0% to 16% Recall). 

To overcome this, our Week 3 update establishes an end-to-end machine learning pipeline encompassing automated data ingestion, domain-informed clinical feature engineering, a 90-fit cross-validation model tournament, and visual validation artifacts.

---

## 2. Phase 1: Data Ingestion, Hygiene, & Clinical Feature Engineering

### 2.1 Pipeline Architecture (`src/ingest_data.py`)
The ingestion pipeline standardizes raw Mendeley records through a sequence of transformations:
1. **Semicolon Delimiter Parsing:** Loads raw semicolon-separated session data (`dataset_10.csv`).
2. **Column Translation:** Maps Spanish clinical headers (e.g., `D_Hipotension`, `D_TAS`, `D_TAD`, `D_Pulso`) to standardized English terms.
3. **Target Binarization:** Standardizes raw target labels into binary classification integers (`0` = Stable, `1` = IDH Crash), resolving class distributions to 79.61% stable vs. 20.39% crash across 13,637 valid sessions.
4. **Leakage & Variance Auditing:** Drops post-treatment leakable variables (`Post_Weight`, `Kt`, `Replacement_Volume`, `Ultrafiltration`) and constant zero-variance features (`Technique`, `Bicarbonate_Conductivity`).
5. **Missing Value Imputation & Outlier Capping:** Imputes missing numerical values using feature medians and applies 1st–99th percentile Winsorization to clamp extreme physiological anomalies.

### 2.2 Domain-Specific Feature Engineering (`src/feature_engineering.py`)
To provide machine learning models with advanced physiological context beyond raw vital signs, 5 hemodynamic and fluid strain interaction features were engineered prior to `MinMaxScaler` normalization:

1. **Pulse Pressure ($PP$):**

   $$PP = \text{Systolic} - \text{Diastolic}$$
   
   *Clinical Rationale:* Measures arterial stiffness and vascular compliance. Narrow pulse pressure combined with rapid fluid removal often precedes circulatory collapse.

2. **Mean Arterial Pressure ($MAP$):**

   $$MAP = \text{Diastolic} + \frac{1}{3}(\text{Systolic} - \text{Diastolic})$$

   *Clinical Rationale:* Provides a robust summary metric of true organ perfusion pressure across the cardiac cycle.

3. **Shock Index ($SI$):**

   $$SI = \frac{\text{Pulse}}{\text{Systolic} + 10^{-5}}$$

   *Clinical Rationale:* A well-established emergency triage metric. A rising heart rate relative to dropping systolic blood pressure signals early compensatory shock.

4. **Relative Weight Gain ($RWG$):**

   $$RWG = \frac{\text{PreWeight} - \text{DryWeight}}{\text{DryWeight} + 10^{-5}}$$

   *Clinical Rationale:* Normalizes interdialytic fluid accumulation relative to patient baseline dry weight, accounting for body size differences.

5. **Ultrafiltration Strain Proxy ($UFR$):**

   $$UFR = \frac{\text{WeightGain}}{\text{BloodFlow} + 10^{-5}}$$
   
   *Clinical Rationale:* Ratios total fluid removal requirements against extracorporeal blood flow rates to gauge vascular refilling stress, stabilized by $\epsilon = 10^{-5}$ to prevent division-by-zero errors.

---

## 3. Phase 2: The 18-Experiment Model Benchmark Matrix (`src/train_baseline.py`)

To rigorously test model behavior under severe class imbalance, we implemented an automated cross-validation engine executing **6 classification algorithms** across **3 class-imbalance strategies** ($6 \times 3 = 18$ distinct experimental configurations) validated via **5-Fold Stratified Cross-Validation** ($18 \times 5 = 90$ total model trainings):

* **Algorithms Evaluated:** Dummy Classifier (Zero-R baseline), Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and XGBoost.
* **Imbalance Strategies:**
  1. *Unadjusted:* Raw 80/20 training distribution.
  2. *Class-Weighted:* Cost-sensitive penalization (~3.90x penalty multiplier for misclassifying minority crash events).
  3. *SMOTE:* Synthetic Minority Over-sampling Technique applied **strictly within training folds** (`smote.fit_resample(X_train, y_train)`) to prevent validation data leakage.

### Complete 18-Experiment Benchmark Leaderboard (5-Fold CV Mean, Sorted by F1-Score)
| Rank | Model | Strategy | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Random Forest** | **SMOTE** | 0.8096 | 0.5389 | 0.4595 | **0.4957** | **0.7915** | 0.5267 |
| **2** | **XGBoost** | **Class-Weighted** | 0.7882 | 0.4818 | **0.5067** | **0.4938** | 0.7676 | 0.5123 |
| **3** | **Random Forest** | **Class-Weighted** | 0.8148 | 0.5585 | 0.4398 | **0.4915** | 0.7905 | **0.5320** |
| **4** | **XGBoost** | **SMOTE** | 0.8169 | 0.5765 | 0.3851 | 0.4612 | 0.7695 | 0.5154 |
| **5** | **Logistic Regression** | **Class-Weighted** | 0.6848 | 0.3530 | **0.6552** | 0.4588 | 0.7344 | 0.4527 |
| **6** | **Logistic Regression** | **SMOTE** | 0.6826 | 0.3516 | **0.6595** | 0.4587 | 0.7341 | 0.4517 |
| **7** | **Gaussian Naive Bayes** | **Class-Weighted** | 0.6715 | 0.3368 | 0.6307 | 0.4391 | 0.7119 | 0.4256 |
| **8** | **Gaussian Naive Bayes** | **SMOTE** | 0.6504 | 0.3223 | 0.6476 | 0.4304 | 0.7063 | 0.4287 |
| **9** | **XGBoost** | **Unadjusted** | 0.8163 | 0.5871 | 0.3413 | 0.4300 | 0.7719 | 0.5151 |
| **10** | **Gaussian Naive Bayes** | **Unadjusted** | 0.7626 | 0.4188 | 0.4243 | 0.4214 | 0.7119 | 0.4256 |
| **11** | **Decision Tree** | **SMOTE** | 0.7188 | 0.3549 | 0.4628 | 0.4015 | 0.6236 | 0.2739 |
| **12** | **Decision Tree** | **Class-Weighted** | 0.7484 | 0.3855 | 0.3930 | 0.3892 | 0.6162 | 0.2753 |
| **13** | **Random Forest** | **Unadjusted** | **0.8252** | **0.6805** | 0.2690 | 0.3850 | 0.7883 | **0.5360** |
| **14** | **Decision Tree** | **Unadjusted** | 0.7422 | 0.3722 | 0.3837 | 0.3778 | 0.6089 | 0.2686 |
| **15** | **Logistic Regression** | **Unadjusted** | 0.8103 | 0.6385 | 0.1636 | 0.2600 | 0.7332 | 0.4587 |
| **16** | **Dummy Classifier** | **Class-Weighted** | 0.4407 | 0.1223 | 0.6000 | 0.2032 | 0.5000 | 0.2039 |
| **17** | **Dummy Classifier** | **Unadjusted** | 0.7961 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2039 |
| **18** | **Dummy Classifier** | **SMOTE** | 0.7961 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2039 |

---

## 4. Phase 3: Notebook Visual Artifacts & Cell-by-Cell Breakdown (`notebooks/baseline_modeling.ipynb`)

To inspect and present findings, an analytical Jupyter Notebook was constructed containing four specialized execution cells:

### Cell 1: Benchmark Leaderboard & Sorting Logic
* **What it does:** Loads `data/baseline_benchmark_results.csv` and renders a clean 18-row tabular leaderboard.
* **Faculty Explanation:** Sorting by **F1-Score** rather than Accuracy prevents models that achieve high accuracy by ignoring minority crashes (e.g., Unadjusted Random Forest at 82.52% accuracy but only 26.90% recall) from falsely dominating the leaderboard. **Random Forest (SMOTE)** ranks #1 overall (`F1 = 0.4957`, `ROC-AUC = 0.7915`), followed closely by **XGBoost (Class-Weighted)** (`F1 = 0.4938`, `Recall = 0.5067`) and **Random Forest (Class-Weighted)** (`F1 = 0.4915`, `PR-AUC = 0.5320`).

### Cell 2: Strategy Impact on Recall & F1-Score (Grouped Bar Charts)
* **What it does:** Plots side-by-side grouped bar charts comparing Clinical Recall (left) and F1-Score (right) across all 6 models and 3 imbalance strategies.
* **Visual Insight:** Proves that unadjusted models fail clinically (Recall < 0.35 for tree models and 0.16 for linear models), whereas Class-Weighting and SMOTE elevate crash detection sensitivity into the 44%–66% range while lifting F1-Scores across all non-trivial classifiers.

### Cell 3: Overlay ROC Curves & Precision-Recall Curves
* **What it does:** Evaluates models on an 80/20 stratified holdout test split (2,728 sessions: 2,172 Stable, 556 Crash) and plots continuous probability thresholds across ROC space and Precision-Recall space.
* **Visual Insight:** **Random Forest (Weighted)** achieves an **ROC-AUC of 0.811** and **PR-AUC of 0.576**, **Random Forest (SMOTE)** achieves an **ROC-AUC of 0.805** and **PR-AUC of 0.548**, and **XGBoost (Weighted)** achieves an **ROC-AUC of 0.793** and **PR-AUC of 0.537**—nearly **triple** the random-chance baseline prevalence (`0.204`).

### Cell 4: Confusion Matrices & Top 12 Feature Importances
* **What it does:** Renders side-by-side $2 \times 2$ Confusion Matrices for our Top 3 Leaderboard models (`Random Forest (SMOTE)`, `XGBoost (Weighted)`, and `Random Forest (Weighted)`) and plots the Top 12 relative feature importances for `Random Forest (SMOTE)` and `XGBoost (Weighted)`.
* **Visual Insight (Confusion Matrices):** On the 2,728 holdout test sessions, **Random Forest (Weighted)** is the most conservative (catching `265` crashes with only `169` false alarms), **Random Forest (SMOTE)** increases sensitivity to catch **`276` crashes** (`217` false alarms), and **XGBoost (Weighted)** achieves the highest ensemble crash capture rate with **`302` caught crashes** (`292` false alarms).
* **Visual Insight (Feature Importance):** Confirms that **4 of the top 9 most decisive predictors** in `Random Forest (SMOTE)` are our newly engineered clinical variables (`Pulse_Pressure` at #2, `Mean_Arterial_Pressure` at #5, `Relative_Weight_Gain` at #6, and `Shock_Index` at #9), validating our Phase 1 physiological feature engineering.

---

## 5. Conclusion & Next Steps
Week 3 successfully transformed exploratory data analysis into a production-grade machine learning benchmark pipeline. Tree ensembles (**Random Forest** with SMOTE/Class-Weighting and **XGBoost** with Class-Weighting) deliver the strongest predictive discrimination and Precision-Recall balance.

*Planned for Week 4:* Hyperparameter optimization via GridSearchCV/RandomizedSearchCV, decision threshold tuning to maximize clinical recall, and SHAP (Shapley Additive exPlanations) model explainability.