# ==============================================================================
# APPLICATION: app.py (Prototype v1)
# PROJECT: Intradialytic Hypotension (IDH) Clinical Decision Support System
# STACK: Streamlit + Tuned LightGBM + Constrained Counterfactual Engine
# ==============================================================================
# LAUNCH COMMAND:
#   streamlit run app.py
# ==============================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

# ==============================================================================
# PAGE CONFIGURATION & CLINICAL ALERT STYLING
# ==============================================================================
st.set_page_config(
    page_title="IDH Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .alert-danger {
        padding: 14px 18px;
        border-radius: 8px;
        background-color: rgba(220, 38, 38, 0.14);
        border-left: 6px solid #dc2626;
        margin-bottom: 14px;
    }
    .alert-safe {
        padding: 14px 18px;
        border-radius: 8px;
        background-color: rgba(16, 185, 129, 0.14);
        border-left: 6px solid #10b981;
        margin-bottom: 14px;
    }
    .alert-warning {
        padding: 14px 18px;
        border-radius: 8px;
        background-color: rgba(245, 158, 11, 0.14);
        border-left: 6px solid #f59e0b;
        margin-bottom: 14px;
    }
    .rx-box {
        padding: 14px 18px;
        border-radius: 8px;
        background-color: rgba(59, 130, 246, 0.10);
        border-left: 5px solid #2563eb;
        margin-top: 8px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# CACHED ARTIFACT LOADER
# ==============================================================================
@st.cache_resource
def load_clinical_artifacts():
    """
    Loads the serialized Champion LightGBM model, patient archetypes,
    and constrained counterfactual prescriptions from 'models/'.
    """
    required_files = [
        "models/tuned_models_bundle.pkl",
        "models/shap_bundle.pkl",
        "models/counterfactual_bundle.pkl",
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing required model artifacts: {missing}. "
            "Run src/tune_models.py, src/explainability.py, and src/counterfactuals.py first."
        )

    tuned_bundle = joblib.load("models/tuned_models_bundle.pkl")
    shap_bundle = joblib.load("models/shap_bundle.pkl")
    cf_bundle = joblib.load("models/counterfactual_bundle.pkl")

    return tuned_bundle, shap_bundle, cf_bundle


try:
    tuned_bundle, shap_bundle, cf_bundle = load_clinical_artifacts()
except Exception as exc:
    st.error(f"Failed to load clinical artifacts: {exc}")
    st.stop()

# Unpack core objects
X_test = tuned_bundle["X_test"]
y_test = tuned_bundle["y_test"]
champ_data = tuned_bundle["models"]["LightGBM (Weighted)"]
champion_model = champ_data["best_estimator"]
y_prob_test = champ_data["y_prob_test"]
opt_f1_thresh = float(champ_data["opt_f1_threshold"])       # 0.57
clinical_thresh = float(champ_data["clinical_threshold"])   # 0.51

archetypes = shap_bundle["archetypes"]

actionable_levers = cf_bundle["actionable_levers"]
lever_grids = cf_bundle["lever_grids"]
lever_display_names = cf_bundle["lever_display_names"]
cf_df = cf_bundle["cf_df"]
cf_rows_dict = cf_bundle["cf_rows_dict"]

feature_names = list(X_test.columns)


# ==============================================================================
# HELPER FUNCTIONS FOR SAFE SLIDER BOUNDS & STATE SYNCHRONIZATION
# ==============================================================================
def get_lever_config(lever_name):
    """
    Returns slider bounds for a given actionable dialysis machine lever.
    """
    col_vals = X_test[lever_name].dropna()
    unique_vals = sorted([float(v) for v in col_vals.unique()])
    grid = lever_grids[lever_name]
    safe_min, safe_max = float(np.min(grid)), float(np.max(grid))

    if len(unique_vals) <= 15:
        return {
            "is_discrete": True,
            "options": unique_vals,
            "safe_min": safe_min,
            "safe_max": safe_max,
        }
    return {
        "is_discrete": False,
        "min_val": float(col_vals.min()),
        "max_val": float(col_vals.max()),
        "safe_min": safe_min,
        "safe_max": safe_max,
    }


def snap_value_to_config(val, cfg):
    """Snaps a value cleanly to the closest discrete option or [min, max] range."""
    val = float(val)
    if cfg["is_discrete"]:
        return min(cfg["options"], key=lambda x: abs(x - val))
    return float(np.clip(val, cfg["min_val"], cfg["max_val"]))


def set_sliders_from_series(row_series):
    """Populates all 6 actionable lever sliders in st.session_state."""
    for lev in actionable_levers:
        cfg = get_lever_config(lev)
        st.session_state[f"slider_{lev}"] = snap_value_to_config(
            row_series[lev], cfg)


# ==============================================================================
# SIDEBAR: THRESHOLD TOGGLE, PATIENT SELECTOR & LOCKED VITALS
# ==============================================================================
st.sidebar.title("🩺 Bedside Control Panel")
st.sidebar.caption("Champion Model: **LightGBM (Weighted)** | Prototype v1")

# 1. Clinical Threshold Mode Toggle
st.sidebar.subheader("1. Alarm Sensitivity Mode")
threshold_mode = st.sidebar.radio(
    "Select Operational Cutoff",
    options=[
        f"Clinical Safety Mode ({clinical_thresh:.2f} | >=65% Recall)",
        f"Balanced Max-F1 Mode ({opt_f1_thresh:.2f} | Max F1)",
    ],
    index=0,
    help="Clinical Safety Mode (0.51) prioritizes early crash detection; Max-F1 Mode (0.57) balances precision and recall.",
)
active_threshold = clinical_thresh if "Clinical Safety" in threshold_mode else opt_f1_thresh

st.sidebar.divider()

# 2. Patient Selection (Quick Case-Study Presets or Any Holdout Session)
st.sidebar.subheader("2. Select Dialysis Session")
selection_mode = st.sidebar.radio(
    "Selection Method",
    ["Clinical Case-Study Presets", "Search Any Holdout Session"],
    index=0,
)

if selection_mode == "Clinical Case-Study Presets":
    preset_map = {
        f"Rescued Crash — Session {archetypes['tp_rescued']['df_index']} ({archetypes['tp_rescued']['pred_prob']:.1%})": archetypes["tp_rescued"]["pos_idx"],
        f"High-Risk Caught Crash — Session {archetypes['tp_caught']['df_index']} ({archetypes['tp_caught']['pred_prob']:.1%})": archetypes["tp_caught"]["pos_idx"],
        f"False Alarm Audit — Session {archetypes['fp_alarm']['df_index']} ({archetypes['fp_alarm']['pred_prob']:.1%})": archetypes["fp_alarm"]["pos_idx"],
        f"Stable Baseline — Session {archetypes['tn_stable']['df_index']} ({archetypes['tn_stable']['pred_prob']:.1%})": archetypes["tn_stable"]["pos_idx"],
    }
    chosen_preset = st.sidebar.selectbox(
        "Clinical Archetype", list(preset_map.keys()))
    pos_idx = int(preset_map[chosen_preset])
else:
    high_risk_only = st.sidebar.checkbox(
        "Filter to High-Risk Sessions Only (Prob >= 0.51)", value=True)
    if high_risk_only:
        candidate_positions = np.where(y_prob_test >= clinical_thresh)[0]
    else:
        candidate_positions = np.arange(len(X_test))

    session_options = [
        f"Session {int(X_test.index[p])} (Row #{p} | Orig Risk: {y_prob_test[p]:.1%})"
        for p in candidate_positions
    ]
    chosen_str = st.sidebar.selectbox("Holdout Session", session_options)
    chosen_idx_in_list = session_options.index(chosen_str)
    pos_idx = int(candidate_positions[chosen_idx_in_list])

# Extract current patient baseline data
orig_row = X_test.iloc[pos_idx]
session_id = int(X_test.index[pos_idx])
actual_outcome = int(y_test.iloc[pos_idx])
orig_prob = float(y_prob_test[pos_idx])

# Reset sliders automatically whenever a different patient is selected
if st.session_state.get("active_pos_idx") != pos_idx:
    st.session_state["active_pos_idx"] = pos_idx
    set_sliders_from_series(orig_row)

# 3. Display Locked Immutable Patient Baselines
st.sidebar.divider()
st.sidebar.subheader("3. Immutable Patient Baselines")
st.sidebar.caption(
    "Locked during treatment (17 features — cannot be modified mid-session).")

key_vitals = [
    "Systolic_BP",
    "Diastolic_BP",
    "Pulse_Pressure",
    "Mean_Arterial_Pressure",
    "Relative_Weight_Gain",
    "Pre_Weight",
    "Dry_Weight",
    "Age",
    "Pulse",
    "Patient_Temperature",
]
vital_rows = [
    {"Clinical Feature": f, "Normalized Value": f"{float(orig_row[f]):.3f}"}
    for f in key_vitals
    if f in orig_row.index
]
st.sidebar.dataframe(
    pd.DataFrame(vital_rows),
    hide_index=True,
    width="stretch",
    height=320,
)


# ==============================================================================
# MAIN WORKSPACE: BEDSIDE RISK MONITOR & LIVE WHAT-IF DIAL SIMULATOR
# ==============================================================================
st.title("🩺 Intradialytic Hypotension (IDH) Bedside Risk & What-If Simulator")
st.markdown(
    f"**Active Patient:** `Session ID {session_id}` (Holdout Row `#{pos_idx}`) &nbsp;|&nbsp; "
    f"**Actual Clinical Outcome:** `{'IDH Crash (1)' if actual_outcome == 1 else 'Stable Session (0)'}` &nbsp;|&nbsp; "
    f"**Active Alert Cutoff:** `{active_threshold:.2f}`"
)

# Check if a pre-computed counterfactual prescription exists for this patient
cf_match = cf_df[cf_df["Holdout_Row"] == pos_idx]
has_cf = len(cf_match) > 0 and pos_idx in cf_rows_dict

# Build simulated feature vector from live slider positions
sim_row = orig_row.copy()
for lev in actionable_levers:
    sim_row[lev] = float(st.session_state[f"slider_{lev}"])

# Run live LightGBM inference on simulated machine settings
sim_df = pd.DataFrame([sim_row.values], columns=feature_names)
sim_prob = float(champion_model.predict_proba(sim_df)[0, 1])
delta_prob = sim_prob - orig_prob

# Count how many dials currently differ from baseline
modified_dials = [
    lev for lev in actionable_levers
    if not np.isclose(float(sim_row[lev]), float(orig_row[lev]), atol=1e-4)
]

# Dynamic Clinical Alert Banner
if sim_prob >= active_threshold:
    if has_cf and not bool(cf_match.iloc[0]["Rescued_Below_051"]):
        st.markdown(
            f"""
            <div class="alert-warning">
                <strong>⚠️ SEVERE HIGH-RISK ALERT ({sim_prob:.2%} &ge; {active_threshold:.2f}) — MEDICAL ESCALATION ADVISED</strong><br>
                Baseline frailty keeps this session above the safety cutoff even after 3 machine dial adjustments.
                Consider direct nephrologist intervention (ultrafiltration hold, saline bolus, or vasopressor support).
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-danger">
                <strong>🚨 HIGH-RISK IDH CRASH ALERT ({sim_prob:.2%} &ge; {active_threshold:.2f})</strong><br>
                Immediate preventive review recommended. Click <strong>Auto-Apply AI Prescription</strong> or adjust the bedside machine dials below.
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        f"""
        <div class="alert-safe">
            <strong>✅ HEMODYNAMICALLY STABLE ({sim_prob:.2%} &lt; {active_threshold:.2f})</strong><br>
            Current dialysis machine configuration sits safely below the active clinical alert threshold.
        </div>
        """,
        unsafe_allow_html=True,
    )

# Top KPI Metrics Row
k1, k2, k3, k4 = st.columns(4)
k1.metric("Baseline Crash Risk", f"{orig_prob:.2%}")
k2.metric(
    "Simulated What-If Risk",
    f"{sim_prob:.2%}",
    delta=f"{delta_prob:+.2%}",
    delta_color="inverse",
)
k3.metric("Active Safety Cutoff", f"{active_threshold:.2%}")
k4.metric("Machine Dials Modified",
          f"{len(modified_dials)} / {len(actionable_levers)}")

st.divider()

# Two-Column Layout: AI Prescription Card (Left) + Live Sliders (Right)
col_left, col_right = st.columns([1.05, 1.25], gap="large")

with col_left:
    st.subheader("🤖 AI Counterfactual Prescription")

    if has_cf:
        r = cf_match.iloc[0]
        rescued_flag = bool(r["Rescued_Below_051"])
        status_badge = "RESCUED BELOW 0.51" if rescued_flag else "PARTIAL MITIGATION (ESCALATE)"

        st.markdown(
            f"""
            <div class="rx-box">
                <strong>Recommended Bedside Adjustment ({status_badge})</strong><br>
                • <strong>Projected Risk Shift:</strong> {r['Original_Prob']:.2%} &rarr; {r['Counterfactual_Prob']:.2%} (<strong>-{r['Risk_Reduction']:.2%}</strong>)<br>
                • <strong>Dials Changed:</strong> {int(r['Levers_Changed'])} setting(s)<br>
                • <strong>Exact Recipe:</strong> <code>{r['Clinical_Prescription']}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("✨ Auto-Apply AI Prescription", type="primary", width="stretch"):
                set_sliders_from_series(cf_rows_dict[pos_idx])
                st.rerun()
        with btn_col2:
            if st.button("↩️ Reset to Original Dials", width="stretch"):
                set_sliders_from_series(orig_row)
                st.rerun()
    else:
        st.info(
            "This session already starts safely below the 0.51 Clinical Safety Threshold. "
            "No preventive machine adjustment is required, though you can still test manual 'What-If' changes on the right."
        )
        if st.button("↩️ Reset to Original Dials", width="stretch"):
            set_sliders_from_series(orig_row)
            st.rerun()

    # Live Before vs. Simulated Horizontal Gauge Bar
    st.markdown("#### Live Risk Comparison")
    fig_bar, ax_bar = plt.subplots(figsize=(6.2, 2.8))
    bar_labels = ["Baseline Risk", "Simulated Risk"]
    bar_vals = [orig_prob, sim_prob]
    bar_colors = [
        "#dc2626" if orig_prob >= active_threshold else "#10b981",
        "#dc2626" if sim_prob >= active_threshold else "#10b981",
    ]
    bars = ax_bar.barh(bar_labels, bar_vals, color=bar_colors,
                       edgecolor="black", height=0.5)
    ax_bar.axvline(
        active_threshold,
        color="#111827",
        linestyle="--",
        linewidth=2.0,
        label=f"Cutoff ({active_threshold:.2f})",
    )
    for b in bars:
        w = b.get_width()
        ax_bar.text(w + 0.02, b.get_y() + b.get_height() / 2.0,
                    f"{w:.1%}", va="center", fontweight="bold")
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel("Predicted IDH Crash Probability")
    ax_bar.legend(loc="lower right", fontsize=8.5)
    ax_bar.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig_bar, clear_figure=True)

with col_right:
    st.subheader("🎛️ Actionable Dialysis Machine Levers")
    st.caption(
        "Adjust any of the 6 bedside machine parameters below to simulate real-time hemodynamic risk changes. "
        "Safe 5th–95th percentile hospital operating limits are shown in each tooltip."
    )

    s_cols = st.columns(2)
    for idx, lev in enumerate(actionable_levers):
        cfg = get_lever_config(lev)
        disp_name = lever_display_names.get(lev, lev)
        orig_v = float(orig_row[lev])

        with s_cols[idx % 2]:
            label_txt = f"{disp_name} (Orig: {orig_v:.3f})"
            help_txt = f"Empirical 5th–95th percentile operating range: [{cfg['safe_min']:.3f}, {cfg['safe_max']:.3f}]"

            if cfg["is_discrete"]:
                st.select_slider(
                    label=label_txt,
                    options=cfg["options"],
                    key=f"slider_{lev}",
                    format_func=lambda x: f"{x:.3f}",
                    help=help_txt,
                )
            else:
                st.slider(
                    label=label_txt,
                    min_value=cfg["min_val"],
                    max_value=cfg["max_val"],
                    step=0.001,
                    key=f"slider_{lev}",
                    format="%.3f",
                    help=help_txt,
                )
