# ==============================================================================
# Explainable AI (XAI) via Global & Local SHAP Frameworks
# ==============================================================================
# OBJECTIVE:
# 1. Load the serialized Week 4 artifact bundle ('models/tuned_models_bundle.pkl').
# 2. Extract our Primary Champion Model: 'LightGBM (Weighted)' along with its
#    calibrated thresholds (Max-F1 = 0.57, Clinical Safety = 0.51) and holdout
#    test partition (2,728 unseen sessions across 23 clinical features).
# 3. Compute exact Shapley Additive exPlanations using shap.TreeExplainer with
#    mathematical additivity verification (check_additivity=True).
# 4. Generate Global Explainability artifacts:
#    - Global Feature Importance CSV ('data/shap_feature_importance.csv')
#    - Global Bar Plot ('visualizations/shap_global_bar.png')
#    - Global Beeswarm Summary Plot ('visualizations/shap_global_beeswarm.png')
# 5. Isolate 4 Clinical Patient Archetypes from the holdout set and generate
#    individualized Local Waterfall Plots ('visualizations/shap_waterfall_*.png'):
#    - Archetype 1: High-Confidence Caught Crash (TP >= 0.57)
#    - Archetype 2: Clinical Safety Rescued Crash (0.51 <= TP < 0.57)
#    - Archetype 3: False Alarm Case (FP >= 0.51)
#    - Archetype 4: Hemodynamically Stable Baseline (TN < 0.51)
# 6. Generate a 2x2 Feature Dependence & Interaction grid for the top 4 global
#    predictors ('visualizations/shap_dependence_interactions.png').
# 7. Serialize the SHAP Explanation object and archetype metadata into
#    'models/shap_bundle.pkl' for Week 5 Notebook, Week 6 Counterfactuals, and
#    Week 7 Streamlit Dashboard deployment.
# ==============================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap

# Use non-interactive 'Agg' backend if running purely via CLI without display,
# ensuring clean high-DPI PNG rendering without GUI window popups.
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def load_champion_bundle(bundle_path="models/tuned_models_bundle.pkl"):
    """
    Loads the Week 4 model bundle and extracts the Champion LightGBM model,
    unseen holdout test data, predicted probabilities, and dual thresholds.
    """
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Error: '{bundle_path}' not found. Run 'python src/tune_models.py' first."
        )

    bundle = joblib.load(bundle_path)
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    champion_name = "LightGBM (Weighted)"
    if champion_name not in bundle["models"]:
        raise KeyError(
            f"Champion model '{champion_name}' missing from bundle.")

    champ_data = bundle["models"][champion_name]
    champion_model = champ_data["best_estimator"]
    # 0.57 (Balanced Max-F1)
    opt_f1_thresh = champ_data["opt_f1_threshold"]
    clinical_thresh = champ_data["clinical_threshold"]   # 0.51 (Recall >= 65%)
    y_prob_test = champ_data["y_prob_test"]

    return (
        champion_name,
        champion_model,
        X_test,
        y_test,
        y_prob_test,
        opt_f1_thresh,
        clinical_thresh,
    )


def compute_shap_explanation(model, X_test):
    """
    Computes exact Shapley values for all 2,728 holdout sessions using TreeExplainer.

    How TreeExplainer Works for LightGBM:
    - Exploits the exact tree structure of LightGBM's leaf-wise decision trees to
      compute exact Shapley values in polynomial time O(TLD^2) rather than exponential time.
    - Operates in raw margin (log-odds) space:
      base_value + sum(SHAP_values_i) = log_odds(P_i), where sigmoid(log_odds) = P_i.
    - check_additivity=True verifies that feature attributions sum exactly to the
      model's raw margin prediction for every single patient session.
    """
    print("[1/5] Initializing shap.TreeExplainer on Champion LightGBM...")
    explainer = shap.TreeExplainer(model)

    print(
        f"[2/5] Computing exact SHAP values across {len(X_test)} unseen sessions...")
    shap_exp = explainer(X_test, check_additivity=True)

    # Compatibility safeguard across SHAP versions:
    # Some SHAP/LightGBM combinations return a 3D Explanation (N_samples, N_features, 2_classes)
    # instead of 2D (N_samples, N_features). If 3D, slice index [:, :, 1] for Class 1 (IDH Crash).
    if len(shap_exp.shape) == 3:
        shap_exp = shap_exp[:, :, 1]

    # Extract scalar expected base value (average log-odds across the training background)
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
    else:
        base_val = float(base_val)

    return explainer, shap_exp, base_val


def summarize_global_importance(shap_exp, feature_names):
    """
    Computes mean(|SHAP|) across all 2,728 test sessions to rank global clinical drivers,
    and also tracks mean directional SHAP value during true crash predictions.
    """
    mean_abs_shap = np.abs(shap_exp.values).mean(axis=0)
    total_importance = mean_abs_shap.sum()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Abs_SHAP": mean_abs_shap,
        "Relative_Contribution_Pct": (mean_abs_shap / total_importance) * 100.0,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    importance_df["Rank"] = importance_df.index + 1
    return importance_df[["Rank", "Feature", "Mean_Abs_SHAP", "Relative_Contribution_Pct"]]


def select_clinical_archetypes(X_test, y_test, y_prob_test, opt_f1_thresh, clinical_thresh):
    """
    Identifies 4 representative holdout patients using positional integer indices (0..2727)
    for local waterfall diagnostics and clinical case studies:
      1. High-Confidence Caught Crash (TP): Actual=1, Prob >= 0.57 (Max-F1 cutoff)
      2. Clinical Safety Rescued Crash (TP): Actual=1, 0.51 <= Prob < 0.57
         -> Demonstrates real patients saved specifically by our dual-threshold calibration!
      3. False Alarm Case (FP): Actual=0, Prob >= 0.51
         -> Helps audit why stable patients trigger clinical alerts.
      4. Hemodynamically Stable Baseline (TN): Actual=0, Prob < 0.51
    """
    y_true_arr = np.asarray(y_test)
    prob_arr = np.asarray(y_prob_test)

    # 1. High-Confidence TP: Pick a strong, representative crash above Max-F1 (e.g. 75th percentile of high TPs)
    tp_high_indices = np.where(
        (y_true_arr == 1) & (prob_arr >= opt_f1_thresh))[0]
    tp_high_sorted = tp_high_indices[np.argsort(prob_arr[tp_high_indices])]
    idx_tp_caught = int(tp_high_sorted[int(len(tp_high_sorted) * 0.75)])

    # 2. Rescued TP: Crashes caught ONLY because Clinical Safety Threshold (0.51) < Max-F1 (0.57)
    tp_rescued_indices = np.where(
        (y_true_arr == 1) & (prob_arr >= clinical_thresh) & (
            prob_arr < opt_f1_thresh)
    )[0]
    tp_rescued_sorted = tp_rescued_indices[np.argsort(
        prob_arr[tp_rescued_indices])]
    idx_tp_rescued = int(tp_rescued_sorted[len(tp_rescued_sorted) // 2])

    # 3. False Positive (FP): Stable session flagged by Clinical Safety Threshold (>= 0.51)
    fp_indices = np.where((y_true_arr == 0) & (prob_arr >= clinical_thresh))[0]
    fp_sorted = fp_indices[np.argsort(prob_arr[fp_indices])]
    idx_fp_alarm = int(fp_sorted[len(fp_sorted) // 2])

    # 4. True Negative (TN): Clearly stable session well below threshold
    tn_indices = np.where((y_true_arr == 0) & (prob_arr < clinical_thresh))[0]
    tn_sorted = tn_indices[np.argsort(prob_arr[tn_indices])]
    idx_tn_stable = int(tn_sorted[int(len(tn_sorted) * 0.25)])

    archetypes = {
        "tp_caught": {
            "pos_idx": idx_tp_caught,
            "df_index": int(X_test.index[idx_tp_caught]),
            "label": "1. High-Confidence Caught Crash (True Positive)",
            "actual": int(y_true_arr[idx_tp_caught]),
            "pred_prob": float(prob_arr[idx_tp_caught]),
            "filename": "visualizations/shap_waterfall_1_tp_caught.png",
        },
        "tp_rescued": {
            "pos_idx": idx_tp_rescued,
            "df_index": int(X_test.index[idx_tp_rescued]),
            "label": "2. Clinical Safety Rescued Crash (0.51 <= P < 0.57)",
            "actual": int(y_true_arr[idx_tp_rescued]),
            "pred_prob": float(prob_arr[idx_tp_rescued]),
            "filename": "visualizations/shap_waterfall_2_tp_rescued.png",
        },
        "fp_alarm": {
            "pos_idx": idx_fp_alarm,
            "df_index": int(X_test.index[idx_fp_alarm]),
            "label": "3. False Alarm Session (False Positive)",
            "actual": int(y_true_arr[idx_fp_alarm]),
            "pred_prob": float(prob_arr[idx_fp_alarm]),
            "filename": "visualizations/shap_waterfall_3_fp_alarm.png",
        },
        "tn_stable": {
            "pos_idx": idx_tn_stable,
            "df_index": int(X_test.index[idx_tn_stable]),
            "label": "4. Hemodynamically Stable Session (True Negative)",
            "actual": int(y_true_arr[idx_tn_stable]),
            "pred_prob": float(prob_arr[idx_tn_stable]),
            "filename": "visualizations/shap_waterfall_4_tn_stable.png",
        },
    }
    return archetypes


def generate_all_visualizations(shap_exp, importance_df, archetypes):
    """
    Renders and saves all Week 5 publication-grade SHAP diagnostics:
      1. Global Feature Importance Bar Chart
      2. Global Beeswarm Summary Plot
      3. Four Individual Patient Waterfall Plots
      4. 2x2 Non-Linear Feature Dependence & Interaction Grid
    """
    os.makedirs("visualizations", exist_ok=True)

    # --------------------------------------------------------------------------
    # PLOT 1: Global Mean(|SHAP|) Feature Importance Bar Chart
    # --------------------------------------------------------------------------
    print("[3/5] Generating Global SHAP Bar & Beeswarm Summary Plots...")
    plt.figure(figsize=(11, 7))
    shap.plots.bar(shap_exp, max_display=15, show=False)
    plt.title(
        "Global Clinical Feature Importance — Champion LightGBM (Mean |SHAP|)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.xlabel(
        "Mean |SHAP Value| (Average Impact on Log-Odds Crash Risk)", fontsize=10)
    plt.tight_layout()
    plt.savefig("visualizations/shap_global_bar.png",
                dpi=300, bbox_inches="tight")
    plt.close("all")

    # --------------------------------------------------------------------------
    # PLOT 2: Global Beeswarm Summary Plot (Magnitude + Directionality)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_exp, max_display=15, show=False)
    plt.title(
        "Global SHAP Beeswarm Summary — Directional Impact Across 2,728 Holdout Sessions",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.xlabel("SHAP Value (Impact on Log-Odds of IDH Crash)", fontsize=10)
    plt.tight_layout()
    plt.savefig("visualizations/shap_global_beeswarm.png",
                dpi=300, bbox_inches="tight")
    plt.close("all")

    # --------------------------------------------------------------------------
    # PLOT 3: Four Individualized Patient Waterfall Plots (Local Explainability)
    # --------------------------------------------------------------------------
    print("[4/5] Generating Local Patient Waterfall Plots for 4 Clinical Archetypes...")
    for key, info in archetypes.items():
        pos_idx = info["pos_idx"]
        plt.figure(figsize=(11, 6.5))
        shap.plots.waterfall(shap_exp[pos_idx], max_display=12, show=False)
        plt.title(
            f"{info['label']}\n"
            f"Holdout Row #{pos_idx} (Session ID {info['df_index']}) | "
            f"Actual={info['actual']} | Predicted Crash Prob={info['pred_prob']:.2%}",
            fontsize=11.5,
            fontweight="bold",
            pad=16,
        )
        plt.tight_layout()
        plt.savefig(info["filename"], dpi=300, bbox_inches="tight")
        plt.close("all")

    # --------------------------------------------------------------------------
    # PLOT 4: 2x2 Feature Dependence & Interaction Grid (Top 4 Global Features)
    # --------------------------------------------------------------------------
    print("[5/5] Generating 2x2 Feature Dependence & Interaction Scatter Plots...")
    top_4_features = importance_df["Feature"].head(4).tolist()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for idx, feat in enumerate(top_4_features):
        ax = axes[idx]
        # Passing color=shap_exp automatically colors dots by the strongest interacting feature
        shap.plots.scatter(
            shap_exp[:, feat],
            color=shap_exp,
            ax=ax,
            show=False,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(
            f"SHAP Dependence & Interaction: {feat}",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.grid(alpha=0.25)

    plt.suptitle(
        "Non-Linear Hemodynamic Tipping Points & Feature Interactions (Top 4 Predictors)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(
        "visualizations/shap_dependence_interactions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close("all")


def run_explainability_pipeline():
    print("=====================================================================")
    print("EXPLAINABLE AI (SHAP) GLOBAL & LOCAL ATTRIBUTION PIPELINE")
    print("=====================================================================\n")

    # 1. Load Champion LightGBM and holdout test set
    (
        champion_name,
        champion_model,
        X_test,
        y_test,
        y_prob_test,
        opt_f1_thresh,
        clinical_thresh,
    ) = load_champion_bundle()

    print(f"Loaded Champion Model : {champion_name}")
    print(
        f"Holdout Test Matrix   : {X_test.shape[0]} sessions x {X_test.shape[1]} features")
    print(
        f"Calibrated Thresholds : Max-F1 = {opt_f1_thresh:.2f} | Clinical Safety = {clinical_thresh:.2f}\n")

    # 2. Compute exact SHAP values with additivity verification
    explainer, shap_exp, base_val = compute_shap_explanation(
        champion_model, X_test)
    print(f"Expected Base Value (Log-Odds): {base_val:.4f}\n")

    # 3. Rank global features and export CSV
    importance_df = summarize_global_importance(shap_exp, list(X_test.columns))
    os.makedirs("data", exist_ok=True)
    importance_df.to_csv("data/shap_feature_importance.csv", index=False)

    # 4. Select 4 clinical patient archetypes
    archetypes = select_clinical_archetypes(
        X_test, y_test, y_prob_test, opt_f1_thresh, clinical_thresh
    )

    # 5. Generate and export all visualizations
    generate_all_visualizations(shap_exp, importance_df, archetypes)

    # 6. Serialize SHAP artifacts for Week 5 Notebook, Week 6 Counterfactuals & Week 7 Streamlit
    os.makedirs("models", exist_ok=True)
    shap_bundle = {
        "champion_name": champion_name,
        "shap_explanation": shap_exp,
        "expected_value": base_val,
        "feature_names": list(X_test.columns),
        "importance_df": importance_df,
        "archetypes": archetypes,
        "opt_f1_threshold": opt_f1_thresh,
        "clinical_threshold": clinical_thresh,
    }
    joblib.dump(shap_bundle, "models/shap_bundle.pkl", compress=3)

    # 7. Print summary diagnostics to terminal
    print("\n=====================================================================")
    print("TOP 12 GLOBAL CLINICAL PREDICTORS (RANKED BY MEAN |SHAP|)")
    print("=====================================================================")
    print(importance_df.head(12).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=====================================================================")
    print("SELECTED LOCAL PATIENT ARCHETYPES (FOR WATERFALL DIAGNOSTICS)")
    print("=====================================================================")
    for key, info in archetypes.items():
        print(
            f"- {info['label']}\n"
            f"  Pos Index: {info['pos_idx']} | Actual: {info['actual']} | "
            f"Pred Prob: {info['pred_prob']:.4f} | Saved: {info['filename']}"
        )

    print("\n[SUCCESS] Exported:")
    print("  -> data/shap_feature_importance.csv")
    print("  -> models/shap_bundle.pkl")
    print("  -> visualizations/shap_global_bar.png")
    print("  -> visualizations/shap_global_beeswarm.png")
    print("  -> visualizations/shap_waterfall_1_tp_caught.png")
    print("  -> visualizations/shap_waterfall_2_tp_rescued.png")
    print("  -> visualizations/shap_waterfall_3_fp_alarm.png")
    print("  -> visualizations/shap_waterfall_4_tn_stable.png")
    print("  -> visualizations/shap_dependence_interactions.png")


if __name__ == "__main__":
    run_explainability_pipeline()
