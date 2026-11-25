# Clinical Decision Support System (CDSS) Technical Report: Prototype v1

## 1. Executive Summary & Clinical Objective
To bridge the gap between offline machine learning pipelines and chairside nephrology workflows, we engineered **Prototype v1** of the **Intradialytic Hypotension (IDH) Clinical Decision Support System** (`app.py` and `requirements.txt`) using **Streamlit**. The application integrates our Champion **Tuned Class-Weighted LightGBM** model (`PR-AUC = 0.6008`) with our **Constrained Counterfactual Intervention Engine**, enabling clinicians to inspect patient crash probabilities, apply minimal AI-recommended dialysis machine adjustments with one click, and interactively simulate custom "What-If" dial modifications in real time.

---

## 2. System Architecture & Artifact Integration
Rather than retraining models or recomputing cohort optimizations during user interaction, `app.py` follows a decoupled, low-latency inference architecture:

* **Cached Resource Loading (`@st.cache_resource`):** All three pre-computed serialized artifacts are loaded once per server session and held in memory:
  1. `models/tuned_models_bundle.pkl`: Supplies the Champion `LightGBM (Weighted)` estimator, the `2,728` unseen holdout sessions (`X_test`, `y_test`), baseline probabilities, and calibrated operational thresholds (`0.51` and `0.57`).
  2. `models/shap_bundle.pkl`: Supplies the clinical patient archetype indices (`Session 11163`, `Session 6780`, `Session 10999`, and `Session 924`) for rapid case-study navigation.
  3. `models/counterfactual_bundle.pkl`: Supplies pre-computed bedside prescriptions for all `734` high-risk sessions, empirical 5th–95th percentile safe operating ranges, and human-readable lever metadata.
* **Live Inference Loop (`st.session_state`):** Every slider movement constructs an updated 23-feature patient vector (preserving all 17 immutable patient baselines while updating only the 6 actionable machine levers) and invokes `champion_model.predict_proba()` instantaneously to refresh risk metrics, alert banners, and the comparison chart.

---

## 3. Bedside Interface & Functional Capabilities

### A. Sidebar — Bedside Control Panel
1. **Alarm Sensitivity Mode Toggle:** Allows clinicians to switch dynamically between:
   * **Clinical Safety Mode (`0.51` cutoff):** Prioritizes early crash detection ($\ge 65\%$ recall).
   * **Balanced Max-F1 Mode (`0.57` cutoff):** Optimizes the precision–recall trade-off to limit alarm fatigue.
2. **Session Selector:** Supports one-click switching across curated **Clinical Case-Study Presets** as well as full search across all `2,728` holdout sessions (with an optional filter for the `734` high-risk sessions where `Prob >= 0.51`).
3. **Immutable Patient Baselines Card:** Displays key locked patient traits (`Systolic_BP`, `Diastolic_BP`, `Pulse_Pressure`, `Mean_Arterial_Pressure`, `Relative_Weight_Gain`, `Pre_Weight`, `Dry_Weight`, `Age`, `Pulse`, `Patient_Temperature`) in a read-only table so clinicians have full physiological context without risking accidental edits to fixed patient attributes.

### B. Main Workspace — Risk Monitor & Live 6-Dial Simulator
1. **Three-Tier Clinical Alert Banner:**
   * **Green (`HEMODYNAMICALLY STABLE`):** Triggered when simulated crash probability sits strictly below the active threshold.
   * **Red (`HIGH-RISK IDH CRASH ALERT`):** Triggered when simulated risk meets or exceeds the active cutoff and machine-level stabilization is feasible.
   * **Amber (`SEVERE HIGH-RISK ALERT — MEDICAL ESCALATION ADVISED`):** Triggered when a high-risk patient cannot be brought below `0.51` even after 3 simultaneous machine dial adjustments (`Rescued_Below_051 = False`), signaling the need for direct nephrologist intervention.
2. **Real-Time KPI Metrics Row:** Tracks **Baseline Crash Risk**, **Simulated What-If Risk** (with color-inverted percentage-point delta $\Delta P$), **Active Safety Cutoff**, and **Machine Dials Modified (`k / 6`)**.
3. **AI Counterfactual Prescription Card:** Displays the exact minimal dial recipe, projected risk drop, and two synchronization buttons—**Auto-Apply AI Prescription** (snaps sliders to the optimal counterfactual settings) and **Reset to Original Dials**.
4. **Live Risk Comparison Chart & 6-Dial Control Grid:** Pairs a dynamic horizontal bar chart (Baseline vs. Simulated probability against the dashed operational cutoff) with six interactive sliders (`Transmembrane_Pressure`, `Dialysate_Bath`, `Dialysate_Flow`, `Dialysate_Temperature`, `Blood_Flow`, and `Dialysate_Conductivity`).

---

## 4. Clinical Safety & Usability Guardrails
* **Strict Actionability Enforcement:** The UI exposes controls *exclusively* for the 6 modifiable dialysis machine levers; all 17 immutable patient traits are structurally isolated from user mutation.
* **Discrete vs. Continuous Input Snapping:** Discrete prescription profiles (such as `Dialysate_Bath`) are rendered as constrained `select_slider` controls restricted to exact clinically observed levels, while continuous machine parameters use fine-grained sliders (`step=0.001`) bounded strictly to observed operating minimums and maximums.
* **Empirical Operating Tooltips:** Every dial control displays the dataset's **5th–95th percentile operating range** directly in its tooltip to discourage out-of-distribution manual testing.

---

## 5. Verified Bedside Demonstration Scenarios (`0.51` Safety Cutoff)

| Clinical Archetype | Session ID (Row) | Baseline Risk | Post-Prescription Risk | Alert Transition | Bedside Action Demonstrated |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Rescued Crash (`TP`)** | `11163` (`#1692`) | `53.63%` | **`43.10%`** (`-10.52%`) | Red $\rightarrow$ **Green** | Single-dial stabilization (`Dialysate_Temperature: 0.333 -> 0.667`). |
| **High-Risk Caught Crash (`TP`)** | `6780` (`#1666`) | `81.33%` | **`66.58%`** (`-14.74%`) | **Amber Escalation** | 3-dial mitigation reduces risk but remains `>= 0.51`; triggers physician escalation alert. |
| **False Alarm Audit (`FP`)** | `10999` (`#2035`) | `62.58%` | **`45.29%`** (`-17.29%`) | Red $\rightarrow$ **Green** | Single-dial adjustment (`Dialysate_Temperature: 0.333 -> 1.000`) clears alarm threshold. |

---

## 6. Local Execution & Planned Roadmap
Prototype v1 runs locally via:
```bash
pip install -r requirements.txt
streamlit run app.py
