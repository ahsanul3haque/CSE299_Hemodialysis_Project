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
