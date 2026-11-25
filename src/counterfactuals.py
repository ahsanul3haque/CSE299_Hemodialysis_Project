# ==============================================================================
# PHASE: Prescriptive AI via Constrained Counterfactual Optimization
# ==============================================================================
# OBJECTIVE:
# 1. Load the Champion 'LightGBM (Weighted)' estimator and holdout test partition
#    (2,728 unseen sessions) from 'models/tuned_models_bundle.pkl', plus the
#    patient archetypes from 'models/shap_bundle.pkl'.
# 2. Enforce strict clinical actionability constraints:
#    - ACTIONABLE LEVERS (6 Modifiable Bedside Settings):
#      Transmembrane_Pressure, Dialysate_Bath, Dialysate_Flow,
#      Dialysate_Temperature, Blood_Flow, Dialysate_Conductivity
#    - IMMUTABLE BASELINES (17 Locked Patient & Vital Features):
#      Age, Sex, Dry_Weight, Pre_Weight, Systolic_BP, Pulse_Pressure, etc.
# 3. For every high-risk holdout session flagged by our Clinical Safety Threshold
#    (Predicted Crash Prob >= 0.51), compute a sparse, plausible counterfactual
#    intervention (1 to 3 machine lever adjustments) that pulls predicted risk
#    below 0.51 (targeting <= 0.45 safety margin) with minimal perturbation.
# 4. Export prescriptive CSV tables and serialize 'models/counterfactual_bundle.pkl'
#    so 'notebooks/counterfactual_analysis.ipynb' can render all visualizations
#    and the Streamlit app can serve live "What-If" prescriptions.
# ==============================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


# ==============================================================================
# STEP 1: DEFINE CLINICAL ACTIONABILITY CONTRACT (MODIFIABLE VS. IMMUTABLE)
# ==============================================================================
# These 6 parameters represent real-time dialysis machine & prescription settings
# that a nephrologist or dialysis nurse can adjust before or during treatment.
ACTIONABLE_LEVERS = [
    "Transmembrane_Pressure",
    "Dialysate_Bath",
    "Dialysate_Flow",
    "Dialysate_Temperature",
    "Blood_Flow",
    "Dialysate_Conductivity",
]

# Human-readable labels saved in the bundle for clean notebook charts and reports
LEVER_DISPLAY_NAMES = {
    "Transmembrane_Pressure": "Transmembrane Pressure (TMP)",
    "Dialysate_Bath": "Dialysate Bath Profile",
    "Dialysate_Flow": "Dialysate Flow Rate",
    "Dialysate_Temperature": "Dialysate Temperature",
    "Blood_Flow": "Blood Flow Rate (Qb)",
    "Dialysate_Conductivity": "Dialysate Conductivity",
}


def load_project_bundles(
    tuned_path="models/tuned_models_bundle.pkl",
    shap_path="models/shap_bundle.pkl",
):
    """
    Loads the serialized tuned model bundle and SHAP archetype bundle.
    """
    if not os.path.exists(tuned_path):
        raise FileNotFoundError(
            f"Missing '{tuned_path}'. Run 'python src/tune_models.py' first."
        )
    if not os.path.exists(shap_path):
        raise FileNotFoundError(
            f"Missing '{shap_path}'. Run 'python src/explainability.py' first."
        )

    tuned_bundle = joblib.load(tuned_path)
    shap_bundle = joblib.load(shap_path)

    X_test = tuned_bundle["X_test"].copy()
    y_test = tuned_bundle["y_test"].copy()

    champ_data = tuned_bundle["models"]["LightGBM (Weighted)"]
    champion_model = champ_data["best_estimator"]
    y_prob_test = champ_data["y_prob_test"]
    opt_f1_thresh = champ_data["opt_f1_threshold"]       # 0.57
    clinical_thresh = champ_data["clinical_threshold"]   # 0.51

    archetypes = shap_bundle["archetypes"]

    # Partition columns into modifiable levers vs. locked patient traits
    valid_levers = [col for col in ACTIONABLE_LEVERS if col in X_test.columns]
    immutable_features = [
        col for col in X_test.columns if col not in valid_levers]

    return (
        champion_model,
        X_test,
        y_test,
        y_prob_test,
        opt_f1_thresh,
        clinical_thresh,
        archetypes,
        valid_levers,
        immutable_features,
    )


# ==============================================================================
# STEP 2: BUILD REALISTIC HOSPITAL OPERATING GRIDS FOR EACH MACHINE LEVER
# ==============================================================================
def build_plausible_lever_grids(X_test, valid_levers):
    """
    Constructs realistic candidate setting values for each actionable lever based
    strictly on empirical clinical operating levels observed in X_test.
    - Discrete settings (<= 15 unique values, e.g., Dialysate_Bath): exact levels.
    - Continuous dials: 19 evenly spaced quantiles from the 5th to 95th percentile.
    """
    lever_grids = {}
    for feat in valid_levers:
        unique_vals = np.sort(X_test[feat].dropna().unique())
        if len(unique_vals) <= 15:
            lever_grids[feat] = unique_vals
        else:
            quantiles = np.linspace(0.05, 0.95, 19)
            grid_vals = np.unique(
                np.round(np.quantile(X_test[feat], quantiles), 4))
            lever_grids[feat] = grid_vals
    return lever_grids


# ==============================================================================
# STEP 3: SPARSE CONSTRAINED COUNTERFACTUAL SEARCH FOR A SINGLE PATIENT
# ==============================================================================
def optimize_single_patient_cf(
    model,
    patient_row,
    orig_prob,
    feature_names,
    valid_levers,
    lever_grids,
    clinical_thresh=0.51,
    target_safe_prob=0.45,
):
    """
    Finds the minimal, clinically realistic machine adjustment that reduces a
    high-risk patient's crash probability below 'clinical_thresh' (0.51).

    Hierarchical Sparsity Search:
      - Stage 1 (1-Lever Search): Test single machine dial adjustments first.
      - Stage 2 (2-Lever Search): Test pairs of machine dials if 1 isn't enough.
      - Stage 3 (3-Lever Search): Test 3-lever combinations for severe high-risk cases.
    """
    base_array = patient_row.values.astype(float)
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    def score_candidates(cand_matrix, l1_dists):
        """
        Scores candidate interventions through Champion LightGBM and selects the
        valid option (P < 0.51) with minimal L1 perturbation distance.
        """
        probs = model.predict_proba(cand_matrix)[:, 1]
        valid_mask = probs < clinical_thresh
        if not np.any(valid_mask):
            return None

        valid_indices = np.where(valid_mask)[0]
        # Prefer candidates reaching <= 0.45 safety margin; fallback to any < 0.51
        safe_margin_indices = valid_indices[probs[valid_indices]
                                            <= target_safe_prob]
        pool = safe_margin_indices if len(
            safe_margin_indices) > 0 else valid_indices

        costs = l1_dists[pool] + 0.25 * \
            np.maximum(0.0, probs[pool] - target_safe_prob)
        best_idx = pool[int(np.argmin(costs))]
        return cand_matrix[best_idx], float(probs[best_idx]), float(l1_dists[best_idx])

    # --------------------------------------------------------------------------
    # STAGE 1: 1-LEVER INTERVENTION SEARCH (Simplest Bedside Action)
    # --------------------------------------------------------------------------
    candidates_1 = []
    l1_1 = []
    single_moves = []

    for lever in valid_levers:
        col_idx = feat_to_idx[lever]
        orig_val = base_array[col_idx]

        for val in lever_grids[lever]:
            if np.isclose(val, orig_val, atol=1e-4):
                continue
            cand = base_array.copy()
            cand[col_idx] = val
            candidates_1.append(cand)
            l1_1.append(abs(val - orig_val))

        # Record this lever's strongest protective setting to seed Stage 2/3
        if len(lever_grids[lever]) > 0:
            grid_batch = np.repeat(
                base_array[np.newaxis, :], len(lever_grids[lever]), axis=0)
            grid_batch[:, col_idx] = lever_grids[lever]
            grid_probs = model.predict_proba(grid_batch)[:, 1]
            min_i = int(np.argmin(grid_probs))
            if grid_probs[min_i] < orig_prob:
                single_moves.append(
                    (grid_probs[min_i], lever, float(lever_grids[lever][min_i])))

    if candidates_1:
        res_1 = score_candidates(np.array(candidates_1), np.array(l1_1))
        if res_1 is not None:
            return res_1[0], res_1[1], 1, res_1[2], True

    single_moves.sort(key=lambda x: x[0])

    # --------------------------------------------------------------------------
    # STAGE 2: 2-LEVER COMBINATION SEARCH
    # --------------------------------------------------------------------------
    candidates_2 = []
    l1_2 = []
    for i in range(len(single_moves)):
        _, lev_a, _ = single_moves[i]
        idx_a = feat_to_idx[lev_a]
        orig_a = base_array[idx_a]

        for j in range(i + 1, len(single_moves)):
            _, lev_b, _ = single_moves[j]
            idx_b = feat_to_idx[lev_b]
            orig_b = base_array[idx_b]

            grid_a = lever_grids[lev_a][:: max(
                1, len(lever_grids[lev_a]) // 8)]
            grid_b = lever_grids[lev_b][:: max(
                1, len(lever_grids[lev_b]) // 8)]

            for va in grid_a:
                if np.isclose(va, orig_a, atol=1e-4):
                    continue
                for vb in grid_b:
                    if np.isclose(vb, orig_b, atol=1e-4):
                        continue
                    cand = base_array.copy()
                    cand[idx_a] = va
                    cand[idx_b] = vb
                    candidates_2.append(cand)
                    l1_2.append(abs(va - orig_a) + abs(vb - orig_b))

    if candidates_2:
        res_2 = score_candidates(np.array(candidates_2), np.array(l1_2))
        if res_2 is not None:
            return res_2[0], res_2[1], 2, res_2[2], True

    # --------------------------------------------------------------------------
    # STAGE 3: 3-LEVER COMBINATION SEARCH (Severe High-Risk Sessions)
    # --------------------------------------------------------------------------
    candidates_3 = []
    l1_3 = []
    top_moves = single_moves[:5]
    for i in range(len(top_moves)):
        _, lev_a, val_a = top_moves[i]
        idx_a = feat_to_idx[lev_a]
        for j in range(i + 1, len(top_moves)):
            _, lev_b, val_b = top_moves[j]
            idx_b = feat_to_idx[lev_b]
            for k in range(j + 1, len(top_moves)):
                _, lev_c, val_c = top_moves[k]
                idx_c = feat_to_idx[lev_c]

                cand = base_array.copy()
                cand[idx_a] = val_a
                cand[idx_b] = val_b
                cand[idx_c] = val_c
                dist = (
                    abs(val_a - base_array[idx_a])
                    + abs(val_b - base_array[idx_b])
                    + abs(val_c - base_array[idx_c])
                )
                candidates_3.append(cand)
                l1_3.append(dist)

    if candidates_3:
        cand_mat_3 = np.array(candidates_3)
        l1_arr_3 = np.array(l1_3)
        res_3 = score_candidates(cand_mat_3, l1_arr_3)
        if res_3 is not None:
            return res_3[0], res_3[1], 3, res_3[2], True

        # Fallback: return the strongest 3-lever risk reduction even if P >= 0.51
        all_probs = model.predict_proba(cand_mat_3)[:, 1]
        best_mit_idx = int(np.argmin(all_probs))
        return (
            cand_mat_3[best_mit_idx],
            float(all_probs[best_mit_idx]),
            3,
            float(l1_arr_3[best_mit_idx]),
            False,
        )

    return base_array, float(orig_prob), 0, 0.0, False


# ==============================================================================
# STEP 4: RUN COHORT-WIDE COUNTERFACTUAL OPTIMIZATION & EXTRACT RECIPES
# ==============================================================================
def generate_cohort_counterfactuals(
    model,
    X_test,
    y_test,
    y_prob_test,
    clinical_thresh,
    valid_levers,
    lever_grids,
):
    """
    Runs constrained counterfactual optimization across EVERY high-risk session
    in the holdout test set (Predicted Prob >= 0.51) and compiles patient-level
    prescriptions + lever frequency analytics.
    """
    high_risk_pos = np.where(y_prob_test >= clinical_thresh)[0]
    feature_names = list(X_test.columns)
    y_true_arr = np.asarray(y_test)

    print(
        f"[2/3] Optimizing counterfactual interventions for {len(high_risk_pos)} "
        f"high-risk holdout sessions (Prob >= {clinical_thresh:.2f})..."
    )

    cf_records = []
    cf_rows_dict = {}
    lever_stats = {
        lev: {"times_prescribed": 0, "total_risk_drop": 0.0, "deltas": []}
        for lev in valid_levers
    }

    for pos_idx in high_risk_pos:
        orig_row = X_test.iloc[pos_idx]
        orig_prob = float(y_prob_test[pos_idx])
        actual_outcome = int(y_true_arr[pos_idx])
        df_id = int(X_test.index[pos_idx])

        cf_arr, cf_prob, num_changed, l1_dist, rescued = optimize_single_patient_cf(
            model=model,
            patient_row=orig_row,
            orig_prob=orig_prob,
            feature_names=feature_names,
            valid_levers=valid_levers,
            lever_grids=lever_grids,
            clinical_thresh=clinical_thresh,
            target_safe_prob=0.45,
        )

        risk_drop = orig_prob - cf_prob
        cf_series = pd.Series(cf_arr, index=feature_names)
        cf_rows_dict[pos_idx] = cf_series

        changes_desc = []
        for lev in valid_levers:
            orig_v = float(orig_row[lev])
            new_v = float(cf_series[lev])
            if not np.isclose(orig_v, new_v, atol=1e-4):
                delta = new_v - orig_v
                direction = "↑" if delta > 0 else "↓"
                changes_desc.append(
                    f"{lev}: {orig_v:.3f} -> {new_v:.3f} ({direction}{abs(delta):.3f})")
                lever_stats[lev]["times_prescribed"] += 1
                lever_stats[lev]["total_risk_drop"] += risk_drop
                lever_stats[lev]["deltas"].append(delta)

        cf_records.append({
            "Holdout_Row": int(pos_idx),
            "Session_ID": df_id,
            "Actual_Crash": actual_outcome,
            "Original_Prob": orig_prob,
            "Counterfactual_Prob": cf_prob,
            "Risk_Reduction": risk_drop,
            "Rescued_Below_051": bool(rescued),
            "Levers_Changed": int(num_changed),
            "L1_Perturbation": l1_dist,
            "Clinical_Prescription": " | ".join(changes_desc) if changes_desc else "No feasible reduction",
        })

    cf_df = pd.DataFrame(cf_records)

    lever_summary_rows = []
    total_high_risk = len(cf_df)
    for lev, st in lever_stats.items():
        count = st["times_prescribed"]
        avg_drop = (st["total_risk_drop"] / count) if count > 0 else 0.0
        avg_delta = float(np.mean(st["deltas"])) if count > 0 else 0.0
        lever_summary_rows.append({
            "Lever_Feature": lev,
            "Display_Name": LEVER_DISPLAY_NAMES.get(lev, lev),
            "Times_Prescribed": count,
            "Usage_Rate_Pct": (count / total_high_risk) * 100.0 if total_high_risk > 0 else 0.0,
            "Avg_Risk_Reduction": avg_drop,
            "Mean_Setting_Shift": avg_delta,
        })

    lever_summary_df = (
        pd.DataFrame(lever_summary_rows)
        .sort_values(by="Times_Prescribed", ascending=False)
        .reset_index(drop=True)
    )

    return cf_df, lever_summary_df, cf_rows_dict


# ==============================================================================
# STEP 5: MAIN PIPELINE EXECUTION & SERIALIZATION
# ==============================================================================
def run_counterfactual_pipeline():
    print("=====================================================================")
    print("PRESCRIPTIVE COUNTERFACTUAL 'WHAT-IF' INTERVENTION ENGINE")
    print("=====================================================================\n")

    # 1. Load Champion LightGBM & SHAP archetypes
    (
        champion_model,
        X_test,
        y_test,
        y_prob_test,
        opt_f1_thresh,
        clinical_thresh,
        archetypes,
        valid_levers,
        immutable_features,
    ) = load_project_bundles()

    print(f"[1/3] Actionability Partition Verified:")
    print(
        f"      -> Modifiable Bedside Levers ({len(valid_levers)}): {valid_levers}")
    print(
        f"      -> Locked Patient Baselines  ({len(immutable_features)}): Immutable during session\n")

    # 2. Build realistic hospital operating grids for the 6 actionable levers
    lever_grids = build_plausible_lever_grids(X_test, valid_levers)

    # 3. Compute sparse counterfactual interventions across all high-risk test patients
    cf_df, lever_summary_df, cf_rows_dict = generate_cohort_counterfactuals(
        model=champion_model,
        X_test=X_test,
        y_test=y_test,
        y_prob_test=y_prob_test,
        clinical_thresh=clinical_thresh,
        valid_levers=valid_levers,
        lever_grids=lever_grids,
    )

    # 4. Export CSVs and serialized Counterfactual Bundle for Notebook & Streamlit App
    print("[3/3] Exporting Counterfactual Tables & Serialized Bundle...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    cf_df.to_csv("data/counterfactual_prescriptions.csv", index=False)
    lever_summary_df.to_csv(
        "data/counterfactual_lever_summary.csv", index=False)

    cf_bundle = {
        "actionable_levers": valid_levers,
        "immutable_features": immutable_features,
        "lever_grids": lever_grids,
        "lever_display_names": LEVER_DISPLAY_NAMES,
        "cf_df": cf_df,
        "lever_summary_df": lever_summary_df,
        "cf_rows_dict": cf_rows_dict,
        "clinical_threshold": clinical_thresh,
        "opt_f1_threshold": opt_f1_thresh,
    }
    joblib.dump(cf_bundle, "models/counterfactual_bundle.pkl", compress=3)

    # 5. Print comprehensive summary report to terminal
    rescue_rate = cf_df["Rescued_Below_051"].mean() * 100.0
    mean_orig = cf_df["Original_Prob"].mean()
    mean_cf = cf_df["Counterfactual_Prob"].mean()
    mean_drop = cf_df["Risk_Reduction"].mean()
    avg_levers = cf_df.loc[cf_df["Rescued_Below_051"], "Levers_Changed"].mean()

    print("\n=====================================================================")
    print("COHORT-WIDE COUNTERFACTUAL INTERVENTION SUMMARY")
    print("=====================================================================")
    print(
        f"Total High-Risk Holdout Sessions (Prob >= {clinical_thresh:.2f}) : {len(cf_df)}")
    print(
        f"Successfully Rescued Below {clinical_thresh:.2f} Threshold      : {cf_df['Rescued_Below_051'].sum()}/{len(cf_df)} ({rescue_rate:.2f}%)")
    print(
        f"Mean Crash Risk Before Intervention               : {mean_orig:.2%}")
    print(f"Mean Crash Risk After Intervention                : {mean_cf:.2%}")
    print(
        f"Average Absolute Risk Reduction                   : -{mean_drop:.2%}")
    print(
        f"Average Machine Dials Adjusted (Sparsity)         : {avg_levers:.2f} levers/patient")

    print("\n=====================================================================")
    print("BEDSIDE DIALYSIS LEVER EFFECTIVENESS RANKING")
    print("=====================================================================")
    print(
        lever_summary_df[
            ["Lever_Feature", "Times_Prescribed", "Usage_Rate_Pct",
                "Avg_Risk_Reduction", "Mean_Setting_Shift"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n=====================================================================")
    print("HIGH-RISK PATIENT ARCHETYPES — PERSONALIZED BEDSIDE PRESCRIPTIONS")
    print("=====================================================================")
    for key in ["tp_caught", "tp_rescued", "fp_alarm"]:
        if key in archetypes:
            pos_idx = archetypes[key]["pos_idx"]
            match = cf_df[cf_df["Holdout_Row"] == pos_idx]
            if len(match) > 0:
                r = match.iloc[0]
                print(
                    f"- {archetypes[key]['label']} (Session ID {int(r['Session_ID'])}, Row #{pos_idx}):\n"
                    f"  Risk: {r['Original_Prob']:.2%} -> {r['Counterfactual_Prob']:.2%} "
                    f"(Rescued: {r['Rescued_Below_051']} | Dials Changed: {int(r['Levers_Changed'])})\n"
                    f"  Prescription: {r['Clinical_Prescription']}"
                )

    print("\n[SUCCESS] Exported:")
    print("  -> data/counterfactual_prescriptions.csv")
    print("  -> data/counterfactual_lever_summary.csv")
    print("  -> models/counterfactual_bundle.pkl")


if __name__ == "__main__":
    run_counterfactual_pipeline()
