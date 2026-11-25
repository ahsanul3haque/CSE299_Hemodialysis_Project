# Prescriptive AI Technical Report: Constrained Counterfactual Intervention Optimization

## 1. Executive Summary & Clinical Objective
Predictive risk alerts and SHAP explanations identify *when* an Intradialytic Hypotension (IDH) crash is imminent and *why* it is occurring, but bedside clinicians ultimately require actionable guidance on *how to intervene*. In this phase, we engineered a **Constrained Counterfactual "What-If" Intervention Engine** (`src/counterfactuals.py` and `notebooks/counterfactual_analysis.ipynb`) built directly atop our Champion **Tuned Class-Weighted LightGBM** estimator (`PR-AUC = 0.6008`, `Clinical Safety Threshold = 0.51`).

Across all **734 high-risk holdout dialysis sessions** flagged by the Clinical Safety Threshold (`Predicted Crash Prob >= 0.51`), the optimizer achieved:
* **51.91% Complete Rescue Rate (`381 / 734` sessions):** Over half of all predicted crashes were brought safely below the `0.51` danger threshold solely via plausible machine adjustments.
* **11.72% Average Absolute Risk Reduction:** Mean cohort crash probability dropped from **`67.67%` to `55.95%`**.
* **High Bedside Sparsity (`1.41` dials/patient):** Rescued patients required adjusting an average of only **1.41 machine dials**, minimizing nursing workload and intervention complexity.

---

## 2. Clinical Actionability & Plausibility Constraints

Unconstrained counterfactual algorithms frequently generate medically impossible recommendations (e.g., reducing patient age or pre-dialysis weight mid-session). To guarantee real-world bedside feasibility, the 23 features were strictly partitioned:

1. **Actionable Bedside Levers (6 Modifiable Parameters):**
   * `Dialysate_Temperature` (Thermal regulation / vasoconstriction control)
   * `Transmembrane_Pressure` (Ultrafiltration driving gradient)
   * `Dialysate_Bath` (Electrolyte/calcium profile)
   * `Dialysate_Flow` (Dialysate circulation rate)
   * `Blood_Flow` (Extracorporeal blood pump rate $Q_b$)
   * `Dialysate_Conductivity` (Sodium modeling / plasma osmolality balance)
2. **Immutable Patient & Baseline Variables (17 Locked Features):**
   * `Age`, `Sex`, `Dry_Weight`, `Pre_Weight`, `Relative_Weight_Gain`, `Dialyzer`, and baseline pre-dialysis hemodynamics (`Systolic_BP`, `Diastolic_BP`, `Pulse_Pressure`, `Mean_Arterial_Pressure`, `Pulse`, `Patient_Temperature`, etc.).
3. **Empirical Plausibility Bounds:** Candidate machine adjustments were restricted to discrete observed prescription levels (`Dialysate_Bath`) or 19 fine-grained quantiles bounded strictly within the **5th to 95th percentile** operating range of the clinical dataset.
4. **Hierarchical Sparsity Optimization:** The solver evaluated **1-lever** changes first, escalating to **2-lever** and **3-lever** combinations only when single-dial adjustments could not cross below `0.51` (targeting a `<= 0.45` safety buffer with minimal $L_1$ perturbation).

---

## 3. Bedside Dialysis Lever Effectiveness Ranking

Evaluating all 734 high-risk sessions revealed which machine parameters most reliably stabilize hemodynamic risk:

| Rank | Dialysis Machine Lever | Times Prescribed | Usage Rate (%) | Avg Risk Reduction ($\Delta P$) | Mean Setting Shift | Clinical Mechanism |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | `Dialysate_Temperature` | **420** | **57.22%** | **-13.66%** | `+0.3781` | Modulates thermal balance and peripheral vascular tone to prevent vasodilation. |
| **2** | `Transmembrane_Pressure` | **336** | **45.78%** | **-11.42%** | `+0.2631` | Adjusts transmembrane filtration dynamics to prevent rapid intravascular volume depletion. |
| **3** | `Dialysate_Bath` | **210** | **28.61%** | **-14.60%** | `-0.1167` | Optimizes electrolyte/calcium bath composition to support myocardial contractility. |
| **4** | `Dialysate_Flow` | **201** | **27.38%** | **-10.15%** | `+0.1201` | Tunes solute and thermal clearance rate across the dialyzer membrane. |
| **5** | `Blood_Flow` | **182** | **24.80%** | **-9.17%** | `+0.2321` | Balances extracorporeal circuit flow against cardiac output tolerance. |
| **6** | `Dialysate_Conductivity` | **147** | **20.03%** | **-10.19%** | `+0.1156` | Stabilizes plasma sodium gradient to preserve intravascular osmotic refilling. |

---

## 4. Personalized Bedside Prescriptions (Patient Archetype Case Studies)

Applying the optimizer to our high-risk clinical archetypes illustrates how counterfactual prescriptions support both **automated machine tuning** and **clinical triage escalation**:

| Clinical Archetype | Session ID (Row) | Before Risk | After Risk | Rescued (`< 0.51`) | Dials Changed | Recommended Bedside Prescription |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. High-Confidence Caught Crash (`TP`)** | `6780` (`#1666`) | **81.33%** | **66.58%** | `False` | `3` | `Transmembrane_Pressure`: `0.027 -> 0.703` (`↑0.676`) \| `Dialysate_Temperature`: `0.333 -> 1.000` (`↑0.667`) \| `Blood_Flow`: `0.655 -> 0.828` (`↑0.172`) |
| **2. Clinical Safety Rescued Crash (`TP`)** | `11163` (`#1692`) | **53.63%** | **43.10%** | **`True`** | **`1`** | `Dialysate_Temperature`: `0.333 -> 0.667` (`↑0.333`) |
| **3. False Alarm Session (`FP`)** | `10999` (`#2035`) | **62.58%** | **45.29%** | **`True`** | **`1`** | `Dialysate_Temperature`: `0.333 -> 1.000` (`↑0.667`) |

### Key Clinical Takeaways:
1. **Single-Dial Stabilization (`Session 11163`):** Caught earlier by our `0.51` Clinical Safety Threshold at `53.63%` risk, adjusting a single dial (`Dialysate_Temperature` from `0.333` to `0.667`) reduces predicted crash risk by `10.53` percentage points down to **`43.10%`**, safely clearing the danger zone.
2. **Recognizing Non-Rescuable Frailty (`Session 6780`):** Starting at `81.33%` risk due to severe immutable baseline factors (`Pulse_Pressure = 0.60`, `Dry_Weight = 0.709`, `Systolic_BP = 0.632`), adjusting three machine settings reduces risk by `14.75%` (to `66.58%`) but cannot pull the patient below `0.51`. In clinical practice, this provides an unambiguous escalation signal: machine dial changes alone are insufficient, and physician intervention (ultrafiltration hold, saline bolus, or pharmacological vasopressor support) is warranted.
