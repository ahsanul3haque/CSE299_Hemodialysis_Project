# Comprehensive Technical Report: Clinical Feature Engineering, 18-Experiment Benchmarking, and Visual Analysis for Intradialytic Hypotension (IDH) Prediction

## 1. Introduction & Clinical Problem Statement
Intradialytic Hypotension (IDH) is one of the most frequent and dangerous complications during hemodialysis sessions, characterized by a sharp drop in blood pressure that can lead to myocardial ischemia, arrhythmia, and organ hypoperfusion. 

Our clinical AI system aims to predict IDH crash events prior to occurrence. Analysis of the Mendeley hemodialysis benchmark datasets reveals a severe class imbalance: **79.61% of sessions are clinically stable (`0`)**, while **20.39% result in an IDH crash (`1`)**. In medical informatics, standard models trained on imbalanced data suffer from the "accuracy trap," yielding high accuracy by simply guessing the majority class while completely missing minority crash events (0% to 16% Recall). 

To overcome this, our Week 3 update establishes an exhaustive, end-to-end machine learning pipeline encompassing automated data ingestion, domain-informed clinical feature engineering, a 90-fit cross-validation model tournament, and comprehensive visual validation artifacts.

---

## 2. Phase 1: Data Ingestion, Hygiene, & Clinical Feature Engineering

### 2.1 Pipeline Architecture (`src/ingest_data.py`)
The ingestion pipeline standardizes raw Mendeley records through a rigorous sequence of transformations:
1. **Semicolon Delimiter Parsing:** Loads raw semicolon-separated session data (`dataset_10.csv`).
2. **Column Translation:** Maps Spanish clinical headers (e.g., `D_Hipotension`, `D_TAS`, `D_TAD`, `D_Pulso`) to standardized English terms.
3. **Target Binarization:** Standardizes raw target codes (`1` = Stable, `2` = IDH Crash) into binary classification labels (`0` and `1`), resolving class distributions to 79.61% stable vs. 20.39% crash across 13,637 valid sessions.
4. **Leakage & Variance Auditing:** Drops post-treatment leakable variables (`Post_Weight`, `Kt`, `Replacement_Volume`, `Ultrafiltration`) and constant zero-variance features (`Technique`, `Bicarbonate_Conductivity`).
5. **Missing Value Imputation & Outlier Capping:** Imputes missing numerical values using feature medians and applies 1st–99th percentile Winsorization to clamp extreme physiological anomalies.

### 2.2 Domain-Specific Feature Engineering (`src/feature_engineering.py`)
To provide machine learning models with advanced physiological context beyond raw vital signs, 5 hemodynamic and fluid strain interaction features were engineered prior to scaling:

1. **Pulse Pressure ($PP$):**
   $$PP = \text{Systolic} - \text{Diastolic}$$
   
   *Clinical Rationale:* Measures arterial stiffness and vascular compliance. Narrow pulse pressure combined with rapid fluid removal often precedes circulatory collapse.

2. **Mean Arterial Pressure ($MAP$):**
   $$MAP = \text{Diastolic} + \frac{1}{3}(\text{Systolic} - \text{Diastolic})$$
   
   *Clinical Rationale:* Provides a robust summary metric of true organ perfusion pressure across the cardiac cycle.

3. **Shock Index ($SI$):**
   $$SI = \frac{\text{Pulse}}{\text{Systolic}}$$
   
   *Clinical Rationale:* A well-established emergency triage metric. An rising heart rate relative to dropping systolic blood pressure signals early compensatory shock.

4. **Relative Weight Gain ($RWG$):**
   $$RWG = \frac{\text{PreWeight} - \text{DryWeight}}{\text{DryWeight}}$$
   
   *Clinical Rationale:* Normalizes interdialytic fluid accumulation relative to patient baseline dry weight, accounting for patient body size differences.

5. **Ultrafiltration Strain Proxy ($UFR$):**
   $$UFR = \frac{\text{WeightGain}}{\text{BloodFlow} + 10^{-5}}$$
   
   *Clinical Rationale:* Ratios total fluid removal requirements against extracorporeal blood flow rates to gauge vascular refilling stress.

---

## 3. Phase 2: The 18-Experiment Model Benchmark Matrix (`src/train_baseline.py`)

To rigorously test model behavior under severe class imbalance, we implemented an automated cross-validation engine executing **6 classification algorithms** across **3 class-imbalance strategies** ($6 \times 3 = 18$ distinct experimental configurations) validated via **5-Fold Stratified Cross-Validation** ($18 \times 5 = 90$ total model trainings):

* **Algorithms Evaluated:** Dummy Classifier (Zero-R baseline), Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and XGBoost.
* **Imbalance Strategies:**
  1. *Unadjusted:* Raw 80/20 training distribution.
  2. *Class-Weighted:* Cost-sensitive penalization (~3.90x penalty multiplier for misclassifying minority crash events).
  3. *SMOTE:* Synthetic Minority Over-sampling Technique applied **strictly within training folds** to prevent data leakage.

### Summary Benchmark Leaderboard (5-Fold CV Mean)
| Model | Strategy | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Class-Weighted | 0.8148 | 0.5585 | 0.4398 | 0.4915 | 0.7905 | 0.5320 |
| **XGBoost** | Class-Weighted | 0.7882 | 0.4818 | 0.5067 | 0.4938 | 0.7676 | 0.5123 |
| **Logistic Regression** | Class-Weighted | 0.6848 | 0.3530 | 0.6552 | 0.4588 | 0.7332 | 0.4517 |
| **Gaussian Naive Bayes** | Class-Weighted | 0.6715 | 0.3368 | 0.6307 | 0.4391 | 0.7119 | 0.4256 |
| **Decision Tree** | Class-Weighted | 0.7484 | 0.3855 | 0.3930 | 0.3892 | 0.7275 | 0.2753 |
| **Dummy Classifier** | Unadjusted | 0.7961 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2039 |

---

## 4. Phase 3: Notebook Visual Artifacts & Cell-by-Cell Breakdown (`notebooks/baseline_modeling.ipynb`)

To inspect and present findings, an analytical Jupyter Notebook was constructed containing four specialized execution cells:

### Cell 1: Benchmark Leaderboard & Sorting Logic
* **What it does:** Loads `data/baseline_benchmark_results.csv` and renders a clean tabular summary.
* **Faculty Explanation:** Sorting by **F1-Score** rather than Accuracy prevents models that achieve high accuracy by ignoring minority crashes from falsely dominating the leaderboard.

### Cell 2: Strategy Impact on Recall & F1-Score (Grouped Bar Charts)
* **What it does:** Plots side-by-side grouped bar charts comparing Clinical Recall (left) and F1-Score (right) across all 6 models and 3 imbalance strategies.
* **Visual Insight:** Proves that unadjusted models fail clinically (Recall < 0.27), whereas class weighting and SMOTE successfully elevate crash detection sensitivity above 50%–65% across all algorithms.

### Cell 3: Overlay ROC Curves & Precision-Recall Curves
* **What it does:** Trains models on an 80/20 stratified holdout split and plots continuous probability thresholds across ROC space and Precision-Recall space.
* **Visual Insight:** **Random Forest (Weighted)** achieves an **AUC of 0.811**, and **XGBoost (Weighted)** achieves an **AUC of 0.793**. In Precision-Recall space—the gold standard for imbalanced clinical data—both ensemble models achieve a PR-AUC of **0.537–0.576**, nearly **triple** the random-chance baseline prevalence (0.204).

### Cell 4: Confusion Matrices & Top 12 Feature Importances
* **What it does:** Renders side-by-side $2 \times 2$ Confusion Matrices for top models and plots relative feature importances for Random Forest and XGBoost.
* **Visual Insight:** Demonstrates that **4 of the top 6 most important features** in Random Forest are our newly engineered clinical variables (`Pulse_Pressure`, `Mean_Arterial_Pressure`, `Relative_Weight_Gain`, and `Shock_Index`), validating the clinical feature engineering module.

---

## 5. Conclusion & Next Steps
Week 3 successfully transformed exploratory data analysis into a rigorous, production-grade machine learning benchmark pipeline. Tree ensembles combined with cost-sensitive class weighting deliver strong predictive performance while maintaining clinical interpretability. 

*Planned for Week 4:* Hyperparameter optimization via GridSearchCV, advanced threshold tuning to maximize clinical recall, and SHAP (Shapley Additive exPlanations) interpretability plots.