# CSE299 Weekly Updates

## Week 1
**Date:** October 4, 2026

**Status:** Complete

**Update Summary:**
For Week 1, our primary goal was establishing a medically defensible foundation and our software architecture. We successfully sourced a 2021 peer-reviewed clinical dataset from Mendeley Data containing over 97,000 real hemodialysis sessions. Rather than using synthetic data or guessing, we documented the exact clinical blood-pressure thresholds the medical researchers used to define intradialytic hypotension, ensuring our ground truth is medically sound.

On the engineering side, we initialized our GitHub repository, configured our data privacy settings, and wrote a Python ingestion pipeline using Pandas. Running this script validated the schema and gave us three critical insights for next week: our target variable has a moderate 67-to-33 class imbalance, the clinical features are labeled in Spanish, and the temperature column is 100% empty. We are now perfectly positioned to build our data cleaning and preprocessing pipeline for Week 2.

---

## Week 2
**Date:** October 11, 2026

**Status:** Complete

**Update Summary:**
For Week 2, our primary objective was performing complete Exploratory Data Analysis (EDA) on `dataset_10.csv` (13,637 total sessions), refining our automated data ingestion pipeline, and conducting a strict data leakage audit ahead of Week 3 model training.

On the data engineering side, we updated `src/ingest_data.py` to handle target binarization properly by converting Spanish string labels (`NO` / `SI`) to binary integers (`0` for Stable, `1` for IDH Crash) with string sanitization. We expanded the pipeline feature-dropping logic to eliminate post-session leakable attributes (hours 1–5 vats/volumes, post-treatment weight) as well as constant, zero-variance features (`Technique`, `Bicarbonate_Conductivity`), resolving division-by-zero `NaN` correlation artifacts.

On the analysis side, we developed `exploratory_data_analysis.ipynb` across four structured cells:
1. **Environment Setup & Data Loading:** Confirmed 13,637 session rows with zero missing values and `MinMaxScaler` normalization within `[0.0, 1.0]`.
2. **Bivariate EDA & Class Imbalance:** Analyzed the target class imbalance (~79.6% Stable vs. ~20.4% IDH Crash) and observed higher pre-dialysis Systolic BP in crash sessions.
3. **Multivariate EDA & Pairplots:** Verified Winsorization outlier capping and identified significant multidimensional spatial overlap between classes, justifying non-linear ML models (e.g., Random Forest, XGBoost).
4. **Correlation Heatmap & Leakage Audit:** Audited all feature correlations, confirming a 100% leak-proof pipeline with a maximum target correlation of `+0.27` (pre-dialysis Systolic BP).

Finally, all findings were documented in `docs/eda_report.md`, establishing a clean foundation for baseline model building in Week 3.

---

## Week 3
**Date:** October 18, 2026

**Status:** Complete

**Update Summary:**
For Week 3, our primary objective was maximizing experimental work volume and moving from exploratory analysis to robust, modular machine learning modeling. Building upon our clean preprocessing pipeline from Week 2, we established a professional two-layer architecture separating heavy model training from visual analysis. 

On the engineering and feature design side, we developed `src/feature_engineering.py` to inject clinical domain knowledge directly into the pipeline. Rather than relying solely on raw vitals, we mathematically engineered 5 hemodynamically critical interaction features before MinMax scaling: Pulse Pressure ($PP$), Mean Arterial Pressure ($MAP$), Shock Index ($SI$), Relative Weight Gain ($RWG$), and Ultrafiltration Strain Proxy ($UFR$). Feature importance analysis confirmed that these engineered variables rank among the top 10 most decisive predictors for tree ensembles.

To tackle the inherent ~80/20 class imbalance (79.61% stable sessions vs. 20.39% IDH crash sessions), we built `src/train_baseline.py`—an automated cross-validation benchmarking engine. We executed a rigorous **$6 \times 3$ model benchmark matrix (18 distinct configurations)** evaluated across **5-Fold Stratified Cross-Validation (90 total model fits)**. We tested 6 algorithms (Dummy, Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and XGBoost) against 3 imbalance handling strategies (Unadjusted, Class-Weighted cost-sensitive penalization, and in-fold SMOTE). 

On the analytical side, we constructed `notebooks/baseline_modeling.ipynb` containing four advanced visual execution cells:
1. **Benchmark Leaderboard:** Ranks all 18 configurations by F1-Score to prioritize crash detection without falling into the accuracy trap.
2. **Strategy Impact Bar Charts:** Grouped bar charts visually proving that unadjusted models fail with <27% Recall, whereas cost weighting and SMOTE successfully surge crash detection sensitivity above 50%–65%.
3. **Overlay ROC & Precision-Recall Curves:** Evaluated models on an 80/20 holdout split, showing that Random Forest (Weighted) and XGBoost (Weighted) achieve an ROC-AUC of 0.79–0.81 and a PR-AUC of 0.53–0.57—nearly triple the random-chance baseline prevalence.
4. **Confusion Matrices & Feature Importances:** Rendered side-by-side confusion matrices and relative feature importance plots demonstrating that 4 of the top 6 predictive features in Random Forest are our newly engineered clinical interaction variables.

Finally, all methodology, metrics, and findings were formally documented in `docs/baseline_report.md`, establishing an airtight foundation for hyperparameter tuning and model explainability in Week 4.