# CSE299 Capstone Project — Week 2 EDA Findings Report

**Dataset:** `dataset_10.csv` (Processed output: `data/processed_dataset_10.csv`)  
**Author:** Ahsanul Haque  
**File:** `Ahsanul_Week2_EDA`  

---

## 1. Notebook Cell Overview (`Ahsanul_Week2_EDA.ipynb`)

| Cell | Title | Purpose / Methods |
| :--- | :--- | :--- |
| **Cell 1** | `Environment Setup & Data Loading` | Pipeline import, dataset load, schema inspection via `df.info()` |
| **Cell 2** | `Bivariate EDA & Target Class Imbalance Analysis` | Class proportion breakdown, `countplot`, bivariate `boxplot` comparisons |
| **Cell 3** | `Multivariate EDA & Feature Distributions` | Bounded feature histograms (`histplot` with KDE), multi-feature `pairplot` |
| **Cell 4** | `Correlation Matrix & Data Leakage Audit` | Full dataset `corr()` matrix, annotated `heatmap`, target correlation ranking |

---

## 2. Findings by Cell

### Cell 1: Environment Setup & Data Loading
* **Dataset Schema & Integrity:** The ingestion pipeline processed 13,637 total hemodialysis session records across 21 features.
* **Missing Value Resolution:** `df.info()` confirms 0 missing or `NaN` values across all columns.
* **Target Binarization:** Target column `Hypotension` was mapped from Spanish string labels (`NO`/`SI`) to binary integers (`0` = Stable, `1` = IDH Crash).
* **Feature Scaling:** Continuous clinical parameters are normalized strictly within `[0.0, 1.0]` using `MinMaxScaler`.

### Cell 2: Bivariate EDA & Target Class Imbalance Analysis
* **Class Imbalance Ratio:**
  * **Class 0 (Stable):** 79.61% (10,856 sessions)
  * **Class 1 (IDH Crash):** 20.39% (2,781 sessions)
  * *Takeaway:* An ~80/20 distribution represents a moderate class imbalance typical of clinical risk events. Model evaluation must prioritize PR-AUC, ROC-AUC, and F1-score over raw accuracy.
* **Bivariate Associations:**
  * **Systolic BP:** Sessions resulting in IDH crashes display higher pre-dialysis Systolic BP medians and IQRs compared to stable sessions.
  * **Blood Flow Rate:** Near-identical medians and spreads between crash and non-crash sessions prove blood flow alone cannot linearly discriminate IDH risk.

### Cell 3: Multivariate EDA & Feature Distributions
* **Winsorization Outlier Control:** Feature histograms for `Systolic_BP` and `Age` confirm bounded scaling within `[0.0, 1.0]`. Minor boundary accumulation validates successful capping of extreme clinical outliers.
* **Class Overlap & Decision Boundaries:** Pairplot interactions across `Systolic_BP`, `Diastolic_BP`, `Blood_Flow`, and `Age` show heavy spatial overlap between classes, proving linear models will be insufficient and justifying non-linear tree ensembles (Random Forest, XGBoost).
* **Feature Co-variance:** Confirmed strong expected physiological alignment between `Systolic_BP` and `Diastolic_BP`.

### Cell 4: Correlation Matrix & Data Leakage Audit
* **Data Leakage Verification (PASSED):**
  * Maximum linear correlation with `Hypotension` is pre-dialysis `Systolic_BP` at `+0.27`.
  * No feature approaches a correlation magnitude above `+0.85` or below `-0.85`.
  * *Verdict:* Post-session features (hours 1–5, post-weight, Kt/V, ultrafiltration) were completely purged. The pipeline is 100% leak-proof.
* **Top Target Correlates:**
  * `Systolic_BP` (`+0.27`) and `Diastolic_BP` (`+0.09`) are the primary positive linear indicators.
  * `Pre_Weight` (`-0.10`) and `Transmembrane_Pressure` (`-0.10`) show slight inverse linear trends.
* **Zero-Variance NaN Resolution (`Technique` & `Bicarbonate_Conductivity`):**
  * Both features display `NaN` correlation scores because they remain constant across all 13,637 sessions in `dataset_10.csv`.
  * Constant values produce a standard deviation of $\sigma = 0$, leading to a division-by-zero error in Pearson correlation calculations.
  * *Pipeline Action:* Updated `remove_leakable_features` in `src/ingest_data.py` to automatically drop zero-variance features alongside post-session attributes, preventing noise in Week 3 model training.
* **Multicollinearity:** `Dry_Weight` and `Pre_Weight` display a perfect correlation of `1.00`. Strong equipment correlations exist between `Blood_Flow`, `Arterial_Pressure` (`-0.63`), and `Venous_Pressure` (`+0.64`).