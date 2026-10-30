# CSE299 Weekly Updates

## Week 1
**Date:** October 5, 2026

**Status:** Complete

**Update Summary:**
For Week 1, our primary goal was establishing a medically defensible foundation and our software architecture. We successfully sourced a 2021 peer-reviewed clinical dataset from Mendeley Data containing over 97,000 real hemodialysis sessions. Rather than using synthetic data or guessing, we documented the exact clinical blood-pressure thresholds the medical researchers used to define intradialytic hypotension, ensuring our ground truth is medically sound.

On the engineering side, we initialized our GitHub repository, configured our data privacy settings, and wrote a Python ingestion pipeline using Pandas. Running this script validated the schema and gave us three critical insights for next week: our target variable has a moderate 67-to-33 class imbalance, the clinical features are labeled in Spanish, and the temperature column is 100% empty. We are now perfectly positioned to build our data cleaning and preprocessing pipeline for Week 2.

---

## Week 2
**Date:** October 12, 2026

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
**Date:** October 26, 2026

**Status:** Complete

**Update Summary:**
For Week 3, our primary objective was maximizing experimental work volume and transitioning from exploratory analysis to modular machine learning modeling. Building upon our clean preprocessing pipeline from Week 2, we established a two-layer architecture separating heavy cross-validation model training from notebook visual diagnostics.

On the feature engineering side, we developed `src/feature_engineering.py` to inject clinical domain knowledge into the pipeline. Prior to `MinMaxScaler` normalization, we mathematically engineered 5 hemodynamic and fluid strain interaction features: Pulse Pressure ($PP$), Mean Arterial Pressure ($MAP$), Shock Index ($SI$), Relative Weight Gain ($RWG$), and Ultrafiltration Strain Proxy ($UFR$). Feature importance analysis later confirmed that 4 of these engineered variables rank among the top 9 most decisive predictors for Random Forest.

To address the ~80/20 class imbalance (79.61% stable sessions vs. 20.39% IDH crash sessions), we built `src/train_baseline.py`—an automated benchmarking engine executing a **$6 \times 3$ model matrix (18 distinct configurations)** across **5-Fold Stratified Cross-Validation (90 total model fits)**. We evaluated 6 algorithms (Dummy, Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, and XGBoost) across 3 imbalance strategies (Unadjusted, Class-Weighted cost-sensitive penalization, and strictly in-fold SMOTE). **Random Forest (SMOTE)** achieved the #1 overall F1-Score (`0.4957`) and ROC-AUC (`0.7915`), **XGBoost (Class-Weighted)** ranked #2 (`0.4938` F1-Score, `50.67%` Recall), and **Random Forest (Class-Weighted)** ranked #3 (`0.4915` F1-Score, `0.5320` PR-AUC).

On the analytical side, we constructed `notebooks/baseline_modeling.ipynb` across four visual execution cells:
1. **Benchmark Leaderboard:** Ranks all 18 configurations by F1-Score to prioritize crash detection over misleading majority-class accuracy.
2. **Strategy Impact Bar Charts:** Grouped bar charts proving that unadjusted models fail with low sensitivity (16%–34% Recall), whereas Class-Weighting and SMOTE elevate crash recall into the 44%–66% range.
3. **Overlay ROC & Precision-Recall Curves:** Evaluated models on an 80/20 stratified holdout split (2,728 sessions), demonstrating that Random Forest (Weighted), Random Forest (SMOTE), and XGBoost (Weighted) achieve ROC-AUCs of 0.811, 0.805, and 0.793, alongside PR-AUCs of 0.576, 0.548, and 0.537—nearly triple the 0.204 baseline prevalence.
4. **Confusion Matrices & Feature Importances:** Compared exact patient classification counts across our top 3 ensemble models (`265` crashes caught by Weighted RF, `276` by SMOTE RF, and `302` by Weighted XGBoost) and verified that engineered hemodynamic ratios directly drive ensemble split decisions.

All methodology, equations, and full 18-experiment tables were documented in `docs/baseline_report.md`, setting up our top tree ensembles for hyperparameter tuning and SHAP explainability in Week 4.

---

---

## Week 4
**Date:** November 2, 2026

**Status:** Complete

**Update Summary:**
In Week 4, our primary objective was transitioning our pipeline from untuned baseline benchmarking to systematically optimized, clinically calibrated production models and selecting our single Champion Architecture for downstream Explainable AI (XAI) and Counterfactual deployment.

First, fulfilling our CSE299 project proposal roadmap, we expanded our gradient boosting framework by integrating **LightGBM (`LGBMClassifier`)** alongside our top Week 3 contenders: **Random Forest (Class-Weighted)**, **Random Forest (SMOTE)**, and **XGBoost (Class-Weighted)**. This allowed us to benchmark leaf-wise histogram decision tree splitting against depth-wise gradient boosting and bootstrap aggregation.

Second, we engineered `src/tune_models.py` to execute **5-Fold Stratified `RandomizedSearchCV` (`15 candidate iterations × 5 folds = 75 fits per model`, totaling 300 cross-validation fits)** optimizing specifically for F1-Score on the 80% training split (10,909 sessions). To guarantee zero validation data leakage during hyperparameter tuning, `Random Forest (SMOTE)` was wrapped inside an `imblearn.pipeline.Pipeline` so synthetic minority oversampling occurred strictly inside the 4 training folds of every split. Hyperparameter optimization produced major structural improvements across all architectures:
- **Random Forest Regularization:** Capping previously unconstrained tree depth (`max_depth=16` for Weighted, `15` for SMOTE) and enforcing minimum leaf populations (`min_samples_leaf=4` and `2`) eliminated majority-class overfitting, surging default-threshold holdout Recall from `48.74%` to `60.61%` (+11.87%) and pushing ROC-AUC to a project-high `0.8158`.
- **Gradient Boosting Shrinkage & Subsampling:** Reducing `learning_rate` from default `0.30` down to `0.03` and restricting feature subsampling (`colsample_bytree=0.8`) prevented `Systolic_BP` from monopolizing early splits, elevating **XGBoost PR-AUC from `0.5369` to `0.5981` (+6.12%)** and propelling **Tuned LightGBM (Weighted)** to the highest overall **PR-AUC (`0.6008`)** and **F1-Score (`0.5717`)** in the study.

Third, recognizing that the default `0.50` probability cutoff is suboptimal for asymmetric medical risk, we performed continuous probability threshold sweeps (`t = 0.20` to `0.80`, step `0.01`) to calibrate two distinct clinical operating modes for each tuned model:
1. **Max-F1 Operating Point:** Maximizes balanced Precision and Recall (where Tuned LightGBM at `t = 0.57` caught `327/556` crashes with only `261` false alarms—achieving our project's highest F1-Score and cutting 110 false alarms compared to untuned baseline).
2. **Clinical Safety Operating Point ($\ge 65\%$ Recall):** Prioritizes patient safety by minimizing missed hypotensive collapses, intentionally accepting a ~1:1 false-to-true alarm ratio (`46.6%–49.5%` Precision) to rescue high-risk patients.

Fourth, we constructed a 4-cell visual diagnostic notebook (`notebooks/hyperparameter_tuning.ipynb`) featuring a 16-stage progression leaderboard, grouped stage bar charts, continuous Precision-Recall-F1 threshold sweep curves, and holdout confusion matrices (evaluated on 2,728 unseen test sessions: 2,172 Stable, 556 IDH Crash) incorporating a 3-way LightGBM progression breakdown (Untuned Baseline @ 0.50 vs. Max-F1 @ 0.57 vs. Clinical Safety @ 0.51). The confusion matrices demonstrated that Week 4 optimization increased captured holdout crashes in Random Forest (Weighted) from `265` to `387` (**+122 additional crashes rescued**, `69.60%` Recall at `t = 0.44`), Random Forest (SMOTE) from `276` to `366` (**+90 rescued**, `65.83%` Recall at `t = 0.44`), and XGBoost/LightGBM to `363` (**+61 rescued**, `65.29%` Recall at `t = 0.51`).

Finally, based on its #1 ranking in both PR-AUC (`0.6008`) and F1-Score (`0.5717`) alongside its dual-threshold flexibility, we selected **Tuned Class-Weighted LightGBM** as our primary Champion Model (with Tuned Random Forest retained as our secondary benchmark), serializing all tuned estimators, probability arrays, and threshold metadata into `models/tuned_models_bundle.pkl` and documenting all findings in `docs/tuning_report.md`.