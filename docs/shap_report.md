# Week 5 Comprehensive Technical Report: Explainable AI (XAI) via Global & Local SHAP Frameworks

## 1. Executive Summary & Clinical Motivation
In Week 4, **Tuned Class-Weighted LightGBM (`LGBMClassifier`)** emerged as our primary Champion Model (`PR-AUC = 0.6008`, `F1 = 0.5717`), calibrated with dual operational thresholds: a **Max-F1 Cutoff (`t = 0.57`)** for balanced alarm precision and a **Clinical Safety Cutoff (`t = 0.51`)** guaranteeing $\ge 65\%$ crash sensitivity. However, in high-acuity nephrology care, a black-box risk score alone cannot guide bedside intervention; clinicians must understand *which* physiological parameters are destabilizing a specific patient before adjusting ultrafiltration or dialysate settings.

In Week 5, we implemented a complete **SHAP (SHapley Additive exPlanations)** pipeline (`src/explainability.py`, `notebooks/shap_explainability.ipynb`, and `models/shap_bundle.pkl`) across all **2,728 unseen holdout hemodialysis sessions** (`23` clinical features)[cite: 5]:
1. **Exact TreeExplainer Attribution:** Computed exact Shapley values in polynomial time in log-odds margin space ($E[f(X)] = -0.6145$), verifying exact mathematical additivity ($\text{base\_value} + \sum_{j=1}^{23} \phi_{i,j} = f(x_i)$) across every holdout session.
2. **Global Risk Driver Discovery:** Ranked all 23 features by $\text{mean}(|\phi_j|)$, confirming that both primary blood pressure vitals and our Phase 1 domain-engineered hemodynamic ratios dominate model decisions[cite: 4].
3. **Local Clinical Archetype Auditing:** Generated individualized SHAP Waterfall explanations for four representative holdout patients—including a real crash patient rescued specifically by our `0.51` Clinical Safety Threshold[cite: 6, 7, 8, 9].
4. **Non-Linear Interaction & Actionability Mapping:** Identified physiological tipping points across top predictors and categorized features into **Modifiable Clinical Levers** vs. **Immutable Patient Baselines** for Week 6 Counterfactual Intervention generation[cite: 3].

---

## 2. Global Feature Importance & Physiological Validation

Across the 2,728 unseen test sessions, the model's expected base log-odds is **$E[f(X)] = -0.6145$** (corresponding to a baseline probability of $35.11\%$ under class-weighted scaling)[cite: 6, 7, 8, 9].

### 2.1 Top 12 Global Clinical Predictors (Ranked by Mean |SHAP|)

| Rank | Feature Name | Feature Type | Mean \|SHAP\| | Relative Contribution (%) | Clinical & Hemodynamic Interpretation |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `Systolic_BP` | Raw Vital | **0.3464** | **18.69%** | Primary arterial pressure driver; extreme shifts trigger steep log-odds risk spikes[cite: 4, 5]. |
| **2** | `Relative_Weight_Gain` | **Engineered** | **0.1914** | **10.33%** | Interdialytic fluid overload relative to dry weight; dictates required fluid removal strain[cite: 4, 5]. |
| **3** | `Pulse_Pressure` | **Engineered** | **0.1825** | **9.85%** | Difference between systolic and diastolic pressure; reflects arterial stiffness & stroke volume[cite: 4, 5]. |
| **4** | `Age` | Demographic | **0.1368** | **7.38%** | Older patients exhibit impaired baroreceptor reflex compensation during rapid fluid removal[cite: 4, 5]. |
| **5** | `Dry_Weight` | Baseline Metric | **0.1296** | **6.99%** | Target post-dialysis euvolemic mass; anchors total intravascular volume reserve[cite: 4, 5]. |
| **6** | `Transmembrane_Pressure` | Machine Parameter | **0.1187** | **6.41%** | Pressure gradient across the dialyzer membrane driving ultrafiltration rate[cite: 4, 5]. |
| **7** | `Pre_Weight` | Session Vital | **0.1021** | **5.51%** | Pre-dialysis patient mass; interacts with dry weight to determine net fluid removal volume[cite: 4, 5]. |
| **8** | `Dialysate_Bath` | Prescription | **0.0980** | **5.29%** | Electrolyte/calcium concentration profile influencing myocardial contractility & vascular tone[cite: 4, 5]. |
| **9** | `Mean_Arterial_Pressure` | **Engineered** | **0.0734** | **3.96%** | Average organ perfusion pressure across the cardiac cycle[cite: 4, 5]. |
| **10** | `Dialysate_Flow` | Machine Parameter | **0.0718** | **3.87%** | Solute clearance and thermal exchange rate across the dialyzer[cite: 4, 5]. |
| **11** | `Pulse` | Raw Vital | **0.0635** | **3.43%** | Heart rate response; reflects compensatory tachycardia vs. autonomic blunting[cite: 4, 5]. |
| **12** | `Dialyzer` | Equipment | **0.0579** | **3.13%** | Membrane surface area and flux characteristics governing clearance dynamics[cite: 4, 5]. |

### 2.2 Validation of Phase 1 Domain Feature Engineering
A critical finding of Week 5 is that three domain-engineered hemodynamic features—**`Relative_Weight_Gain` (`10.33%`)**, **`Pulse_Pressure` (`9.85%`)**, and **`Mean_Arterial_Pressure` (`3.96%`)**—jointly contribute **24.14%** of the Champion LightGBM model's total explanatory power[cite: 4]. Rather than relying purely on isolated raw vitals, leaf-wise gradient boosting actively leverages physiological ratios capturing fluid overload strain and vascular compliance[cite: 3, 5].

---

## 3. Local Patient Waterfall Diagnostics (4 Clinical Archetypes)

To audit bedside decision transparency, we extracted individualized SHAP Waterfall explanations (`visualizations/shap_waterfall_*.png`) across four distinct operational outcomes in `X_test`[cite: 6, 7, 8, 9]:

| Clinical Archetype | Holdout Row | Session ID | Actual Outcome | Predicted Prob | Final Log-Odds $f(x)$ | Primary Driving Attributions |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. High-Confidence Caught Crash (`TP`)** | `#1666` | `6780` | `1` (Crash) | **81.33%** | `+1.471` | `Pulse_Pressure` (`+0.48`), `Dry_Weight` (`+0.29`), `Systolic_BP` (`+0.25`), `Transmembrane_Pressure` (`+0.19`), `Relative_Weight_Gain` (`+0.14`)[cite: 6] |
| **2. Clinical Safety Rescued Crash (`TP`)** | `#1692` | `11163` | `1` (Crash) | **53.63%** | `+0.145` | `Pulse_Pressure` (`+0.50`), `Systolic_BP` (`+0.16`), `Age` (`+0.12`), `Patient_Temperature` (`+0.12`) offset by `Relative_Weight_Gain` (`-0.31`)[cite: 7] |
| **3. False Alarm Session (`FP`)** | `#2035` | `10999` | `0` (Stable) | **62.58%** | `+0.514` | `Transmembrane_Pressure` (`+0.19`), `Pulse_Pressure` (`+0.19`), `Dialysate_Bath` (`+0.14`), `Pre_Weight` (`+0.14`)[cite: 8] |
| **4. Hemodynamically Stable (`TN`)** | `#830` | `924` | `0` (Stable) | **16.01%** | `-1.658` | Protective `Pulse_Pressure` (`-0.29`), `Age` (`-0.26`), `Systolic_BP` (`-0.26`), `Transmembrane_Pressure` (`-0.19`)[cite: 9] |

### Key Case-Study Insights:
* **Why Dual Thresholds Matter (`Session ID 11163`):** Patient `#1692` (Session ID `11163`) experienced an actual IDH crash (`Actual = 1`)[cite: 7]. Because their fluid gain was relatively controlled (`Relative_Weight_Gain` contributed `-0.31` log-odds), their overall risk probability settled at **`53.63%`**[cite: 7]. Under the balanced Max-F1 threshold (`0.57`), this patient would have been classified as safe (a False Negative). Under our **Clinical Safety Threshold (`0.51`)**, the alarm triggered correctly, demonstrating concrete clinical value[cite: 7].
* **Auditing False Alarms (`Session ID 10999`):** Patient `#2035` exhibited high `Transmembrane_Pressure` (`+0.19`) and `Pulse_Pressure` (`+0.19`)[cite: 8]—genuine physiological precursors to hypotension—yielding a `62.58%` risk alert[cite: 8] despite completing the session without crashing (likely stabilized via routine bedside nursing adjustment).

---

## 4. Non-Linear Tipping Points & Week 6 Counterfactual Readiness

Inspecting the 2x2 SHAP Dependence & Interaction plots (`visualizations/shap_dependence_interactions.png`) reveals sharp non-linear regime shifts across normalized feature scales (`[0, 1]`)[cite: 3]:
1. **`Systolic_BP` × `Relative_Weight_Gain`:** Risk remains strongly negative below normalized `0.50`, crosses zero around `0.55`, and exhibits a sharp upward step discontinuity at `0.62–0.65`, amplified when `Relative_Weight_Gain` is elevated[cite: 3].
2. **`Relative_Weight_Gain` × `Pre_Weight`:** Crosses from protective to positive crash risk at normalized `0.45`, surging steeply above `0.60`[cite: 3].
3. **`Pulse_Pressure` × `Dry_Weight`:** Displays a U-shaped hemodynamic curve—lowest risk in the `0.15–0.35` range, followed by an abrupt jump above `0.55`[cite: 3].

### Feature Actionability Contract for Week 6 (Counterfactual Engine)
To prepare for **Week 6 Counterfactual "What-If" Intervention Optimization**, SHAP-ranked features are partitioned into actionable clinical levers vs. fixed constraints:
* **Actionable Machine & Prescription Levers (Modifiable during/before dialysis):** `Transmembrane_Pressure` (Rank #6), `Dialysate_Bath` (Rank #8), `Dialysate_Flow` (Rank #10), `Dialysate_Temperature` (Rank #14), and ultrafiltration rate targets governing `Relative_Weight_Gain` strain[cite: 4].
* **Immutable Patient & Baseline Variables (Constrained/Fixed in Counterfactual search):** `Age` (Rank #4), `Dry_Weight` (Rank #5), `Pre_Weight` (Rank #7), `Sex`, and pre-existing baseline vitals[cite: 4].