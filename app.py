# ==============================================================================
# APPLICATION: app.py (CDSS v2 — Unified Inference Engine + Counterfactuals)
# PROJECT: Intradialytic Hypotension (IDH) Clinical Decision Support System
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
from pydantic import ValidationError

from src.models.inference import InferenceEngine

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
# CACHED BACKEND & ARTIFACT LOADERS
# ==============================================================================
@st.cache_resource
def load_inference_engine() -> InferenceEngine:
    return InferenceEngine("models/champion_inference_pipeline.joblib")


@st.cache_resource
def load_clinical_artifacts():
    required_files = [
        "models/tuned_models_bundle.pkl",
        "models/shap_bundle.pkl",
        "models/counterfactual_bundle.pkl",
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing required model artifacts: {missing}. "
            "Run run_week_6_pipeline.py, src/explainability.py, and src/counterfactuals.py first."
        )

    tuned_bundle = joblib.load("models/tuned_models_bundle.pkl")
    shap_bundle = joblib.load("models/shap_bundle.pkl")
    cf_bundle = joblib.load("models/counterfactual_bundle.pkl")
    return tuned_bundle, shap_bundle, cf_bundle


try:
    engine = load_inference_engine()
    tuned_bundle, shap_bundle, cf_bundle = load_clinical_artifacts()
except Exception as exc:
    st.error(f"Failed to initialize CDSS backend: {exc}")
    st.stop()

# Unpack synced thresholds & holdout artifacts
X_test = tuned_bundle["X_test"]
y_test = tuned_bundle["y_test"]
champ_data = tuned_bundle["models"]["LightGBM (Weighted)"]
champion_model = champ_data["best_estimator"]
y_prob_test = champ_data["y_prob_test"]
clinical_thresh = float(engine.clinical_threshold)
max_f1_thresh = float(engine.max_f1_threshold)

archetypes = shap_bundle["archetypes"]
actionable_levers = cf_bundle["actionable_levers"]
lever_grids = cf_bundle["lever_grids"]
lever_display_names = cf_bundle["lever_display_names"]
cf_df = cf_bundle["cf_df"]
cf_rows_dict = cf_bundle["cf_rows_dict"]
feature_names = list(X_test.columns)


# ==============================================================================
# HELPER FUNCTIONS (TAB 3 SLIDER SYNC)
# ==============================================================================
def get_lever_config(lever_name):
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
    val = float(val)
    if cfg["is_discrete"]:
        return min(cfg["options"], key=lambda x: abs(x - val))
    return float(np.clip(val, cfg["min_val"], cfg["max_val"]))


def set_sliders_from_series(row_series):
    for lev in actionable_levers:
        cfg = get_lever_config(lev)
        st.session_state[f"slider_{lev}"] = snap_value_to_config(row_series[lev], cfg)


def render_three_tier_gauge(prob: float, clin_t: float, f1_t: float):
    fig, ax = plt.subplots(figsize=(7.0, 1.9))
    ax.axvspan(0.0, clin_t, color="#10b981", alpha=0.22, label=f"Low Risk (< {clin_t:.2f})")
    ax.axvspan(clin_t, f1_t, color="#f59e0b", alpha=0.28, label=f"Clinical Safety ({clin_t:.2f}–{f1_t:.2f})")
    ax.axvspan(f1_t, 1.0, color="#dc2626", alpha=0.22, label=f"High Risk (>= {f1_t:.2f})")

    bar_color = "#dc2626" if prob >= f1_t else ("#f59e0b" if prob >= clin_t else "#10b981")
    ax.barh(["Crash Risk"], [prob], color=bar_color, edgecolor="black", height=0.45)
    ax.axvline(clin_t, color="#d97706", linestyle="--", linewidth=1.8)
    ax.axvline(f1_t, color="#b91c1c", linestyle="-.", linewidth=1.8)
    ax.text(min(prob + 0.02, 0.92), 0, f"{prob:.1%}", va="center", fontweight="bold", fontsize=11)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Predicted IDH Crash Probability")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.38), ncol=3, frameon=False, fontsize=8.5)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

# ==============================================================================
# LIVE RAW-UNIT COUNTERFACTUAL ENGINE (TAB 1 — POWERED BY engine.predict_batch)
# ==============================================================================
RAW_LEVER_CONFIG = {
    "Dialysate_Temperature": {"grid": [35.0, 35.5, 36.0, 36.5, 37.0, 37.5], "unit": "°C", "label": "Dialysate Temperature"},
    "Blood_Flow": {"grid": [200.0, 250.0, 300.0, 350.0, 400.0, 450.0], "unit": "mL/min", "label": "Blood Flow Rate (Qb)"},
    "Dialysate_Flow": {"grid": [300.0, 400.0, 500.0, 600.0, 700.0, 800.0], "unit": "mL/min", "label": "Dialysate Flow Rate"},
    "Dialysate_Conductivity": {"grid": [13.5, 13.8, 14.0, 14.2, 14.5, 14.8, 15.0], "unit": "mS/cm", "label": "Dialysate Conductivity"},
    "Transmembrane_Pressure": {"grid": [40.0, 60.0, 80.0, 100.0, 120.0, 150.0, 180.0, 220.0], "unit": "mmHg", "label": "Transmembrane Pressure (TMP)"},
    "Dialysate_Bath": {"grid": [0.0, 1.0, 2.0, 3.0], "unit": "ID", "label": "Dialysate Bath Profile"},
}


def generate_live_prescription(engine: InferenceEngine, base_payload: dict, orig_prob: float) -> dict:
    """
    Runs a 1->2->3 lever hierarchical counterfactual search in raw clinical units
    using engine.predict_batch(), minimizing normalized dial perturbation.
    """
    thresh = float(engine.clinical_threshold)
    levers = list(RAW_LEVER_CONFIG.keys())

    def norm_dist(changes: dict) -> float:
        total = 0.0
        for k, new_v in changes.items():
            g = RAW_LEVER_CONFIG[k]["grid"]
            span = max(g) - min(g) if max(g) > min(g) else 1.0
            total += abs(new_v - float(base_payload[k])) / span
        return total

    def evaluate_pool(change_dicts: list):
        if not change_dicts:
            return None, []
        candidates = [{**base_payload, **ch} for ch in change_dicts]
        preds = engine.predict_batch(candidates)
        probs = [p.crash_probability for p in preds]

        valid = [i for i, p in enumerate(probs) if p < thresh]
        if not valid:
            return None, list(zip(probs, change_dicts))

        best_i = min(valid, key=lambda i: norm_dist(change_dicts[i]) + 0.1 * probs[i])
        return {
            "rescued": True,
            "new_prob": probs[best_i],
            "risk_drop": orig_prob - probs[best_i],
            "changes": change_dicts[best_i],
        }, list(zip(probs, change_dicts))

    # Stage 1: 1-Lever Search
    stage1_changes = []
    for lev in levers:
        orig_v = float(base_payload[lev])
        for val in RAW_LEVER_CONFIG[lev]["grid"]:
            if not np.isclose(val, orig_v, atol=1e-4):
                stage1_changes.append({lev: val})

    res1, scored1 = evaluate_pool(stage1_changes)
    if res1:
        return res1

    # Track the single most protective setting per lever for Stage 2 & 3
    best_per_lever = {}
    for prob, ch in scored1:
        lev, val = next(iter(ch.items()))
        if prob < orig_prob and (lev not in best_per_lever or prob < best_per_lever[lev][0]):
            best_per_lever[lev] = (prob, val)

    top_levers = sorted(best_per_lever.items(), key=lambda x: x[1][0])

    # Stage 2: 2-Lever Search
    stage2_changes = []
    for i in range(len(top_levers)):
        lev_a, (_, val_a) = top_levers[i]
        for j in range(i + 1, len(top_levers)):
            lev_b, (_, val_b) = top_levers[j]
            stage2_changes.append({lev_a: val_a, lev_b: val_b})

    res2, scored2 = evaluate_pool(stage2_changes)
    if res2:
        return res2

    # Stage 3: 3-Lever Search (Severe High-Risk Fallback)
    stage3_changes = []
    top5 = top_levers[:5]
    for i in range(len(top5)):
        lev_a, (_, val_a) = top5[i]
        for j in range(i + 1, len(top5)):
            lev_b, (_, val_b) = top5[j]
            for k in range(j + 1, len(top5)):
                lev_c, (_, val_c) = top5[k]
                stage3_changes.append({lev_a: val_a, lev_b: val_b, lev_c: val_c})

    res3, scored3 = evaluate_pool(stage3_changes)
    if res3:
        return res3

    # Fallback: return the strongest risk reduction found across all stages
    all_scored = scored1 + scored2 + scored3
    if all_scored:
        best_prob, best_ch = min(all_scored, key=lambda x: x[0])
        if best_prob < orig_prob:
            return {
                "rescued": False,
                "new_prob": best_prob,
                "risk_drop": orig_prob - best_prob,
                "changes": best_ch,
            }

    return {"rescued": False, "new_prob": orig_prob, "risk_drop": 0.0, "changes": {}}    


# ==============================================================================
# GLOBAL SIDEBAR: SYSTEM STATUS & OPERATING THRESHOLDS
# ==============================================================================
st.sidebar.title("🩺 IDH CDSS v2")
st.sidebar.caption(f"Backend: **{engine.model_name}**")
st.sidebar.markdown(
    f"""
    **Synced Clinical Cutoffs**
    - 🟢 **Low Risk:** `< {clinical_thresh:.2f}`
    - 🟠 **Clinical Safety Alert:** `{clinical_thresh:.2f} – {max_f1_thresh:.2f}`
    - 🔴 **High Risk:** `>= {max_f1_thresh:.2f}`
    """
)
st.sidebar.divider()
st.sidebar.info(
    "Use **Tab 1** for live bedside vitals (raw clinical units), "
    "**Tab 2** for multi-patient CSV screening, and "
    "**Tab 3** for holdout counterfactual prescriptions."
)


# ==============================================================================
# MAIN HEADER & 3-TAB NAVIGATION
# ==============================================================================
st.title("🩺 Intradialytic Hypotension (IDH) Clinical Decision Support System")

tab_live, tab_batch, tab_cf = st.tabs([
    "🏥 Live Bedside Intake",
    "📋 Batch Ward Screening",
    "🤖 Holdout Counterfactual Simulator",
])


# ==============================================================================
# TAB 1: LIVE BEDSIDE PATIENT INTAKE (NURSE-FRIENDLY UX)
# ==============================================================================
with tab_live:
    # 1. Quick-Load Clinical Presets (Great for demos & rapid testing)
    PRESETS = {
        "stable": {
            "t1_sex": "Male", "t1_age": 54.0, "t1_dry_w": 72.0, "t1_pre_w": 73.5,
            "t1_sbp": 136.0, "t1_dbp": 82.0, "t1_pulse": 74.0, "t1_pt_temp": 36.6,
            "t1_qb": 320.0, "t1_qd": 500.0, "t1_dtemp": 36.5, "t1_cond": 14.0,
            "t1_ap": -130.0, "t1_vp": 135.0, "t1_tmp": 85.0, "t1_dialyzer": 1, "t1_bath": 1,
        },
        "moderate": {
            "t1_sex": "Female", "t1_age": 67.0, "t1_dry_w": 62.0, "t1_pre_w": 64.8,
            "t1_sbp": 112.0, "t1_dbp": 66.0, "t1_pulse": 84.0, "t1_pt_temp": 36.7,
            "t1_qb": 300.0, "t1_qd": 500.0, "t1_dtemp": 36.8, "t1_cond": 13.9,
            "t1_ap": -145.0, "t1_vp": 140.0, "t1_tmp": 120.0, "t1_dialyzer": 1, "t1_bath": 1,
        },
        "high_risk": {
            "t1_sex": "Female", "t1_age": 74.0, "t1_dry_w": 58.0, "t1_pre_w": 61.8,
            "t1_sbp": 94.0, "t1_dbp": 58.0, "t1_pulse": 94.0, "t1_pt_temp": 36.9,
            "t1_qb": 300.0, "t1_qd": 500.0, "t1_dtemp": 37.0, "t1_cond": 13.8,
            "t1_ap": -160.0, "t1_vp": 145.0, "t1_tmp": 165.0, "t1_dialyzer": 1, "t1_bath": 2,
        },
    }

    # Initialize defaults once
    if "t1_sbp" not in st.session_state:
        for k, v in PRESETS["stable"].items():
            st.session_state[k] = v

    p_col1, p_col2, p_col3, p_col4 = st.columns([1.4, 1, 1, 1])
    p_col1.markdown("#### Quick-Load Patient Profile")
    if p_col2.button("🟢 Stable Baseline", use_container_width=True):
        for k, v in PRESETS["stable"].items():
            st.session_state[k] = v
        st.rerun()
    if p_col3.button("🟠 Borderline / Elderly", use_container_width=True):
        for k, v in PRESETS["moderate"].items():
            st.session_state[k] = v
        st.rerun()
    if p_col4.button("🔴 High-Risk Crash", use_container_width=True):
        for k, v in PRESETS["high_risk"].items():
            st.session_state[k] = v
        st.rerun()

    st.divider()

    # 2. Clean 2-Column Clinical Intake (Left: Patient Vitals | Right: Machine Settings)
    left_intake, right_machine = st.columns([1.15, 1.0], gap="large")

    with left_intake:
        st.markdown("#### 🩺 Pre-Dialysis Patient Vitals")
        v1, v2, v3 = st.columns(3)
        age = v1.number_input("Age (yrs)", 18.0, 100.0, step=1.0, key="t1_age")
        sex_str = v2.selectbox("Sex", ["Female", "Male"], key="t1_sex")
        sex = 1 if sex_str == "Male" else 2
        patient_temp = v3.number_input("Body Temp (°C)", 35.0, 40.0, step=0.1, format="%.1f", key="t1_pt_temp")

        w1, w2, w3 = st.columns(3)
        dry_weight = w1.number_input("Dry Weight (kg)", 25.0, 220.0, step=0.5, format="%.1f", key="t1_dry_w")
        pre_weight = w2.number_input("Pre-Weight (kg)", 25.0, 240.0, step=0.5, format="%.1f", key="t1_pre_w")
        weight_gain = round(pre_weight - dry_weight, 2)
        w3.metric("Fluid Gain (IDWG)", f"{weight_gain:+.1f} kg")

        bp1, bp2, bp3 = st.columns(3)
        systolic_bp = bp1.number_input("Systolic BP (mmHg)", 60.0, 220.0, step=1.0, format="%.0f", key="t1_sbp")
        diastolic_bp = bp2.number_input("Diastolic BP (mmHg)", 30.0, 130.0, step=1.0, format="%.0f", key="t1_dbp")
        pulse = bp3.number_input("Pulse (bpm)", 40.0, 180.0, step=1.0, format="%.0f", key="t1_pulse")

        pulse_pressure = systolic_bp - diastolic_bp
        map_val = diastolic_bp + (pulse_pressure / 3.0)
        st.caption(f"Auto-Calculated — **Pulse Pressure:** `{pulse_pressure:.0f} mmHg` | **MAP:** `{map_val:.1f} mmHg`")

    with right_machine:
        st.markdown("#### ⚙️ Dialysis Machine Prescription")
        m_c1, m_c2 = st.columns(2)
        dialysate_temp = m_c1.number_input("Dialysate Temp (°C)", 35.0, 38.0, step=0.5, format="%.1f", key="t1_dtemp")
        blood_flow = m_c2.number_input("Blood Flow — Qb (mL/min)", 100.0, 500.0, step=25.0, format="%.0f", key="t1_qb")

        m_c3, m_c4 = st.columns(2)
        dialysate_flow = m_c3.number_input("Dialysate Flow (mL/min)", 300.0, 800.0, step=50.0, format="%.0f", key="t1_qd")
        dialysate_cond = m_c4.number_input("Conductivity (mS/cm)", 13.0, 15.5, step=0.1, format="%.1f", key="t1_cond")

        with st.expander("Circuit Pressures & Consumables (Clinic Defaults)", expanded=False):
            p1, p2, p3 = st.columns(3)
            tmp = p1.number_input("TMP (mmHg)", -50.0, 400.0, step=10.0, format="%.0f", key="t1_tmp")
            arterial_pressure = p2.number_input("Arterial (mmHg)", -350.0, 50.0, step=10.0, format="%.0f", key="t1_ap")
            venous_pressure = p3.number_input("Venous (mmHg)", 0.0, 350.0, step=10.0, format="%.0f", key="t1_vp")

            h1, h2 = st.columns(2)
            dialyzer = h1.number_input("Dialyzer Profile", 0, 20, step=1, key="t1_dialyzer")
            dialysate_bath = h2.number_input("Dialysate Bath Profile", 0, 20, step=1, key="t1_bath")

    payload = {
        "Sex": sex, "Age": age, "Dialyzer": float(dialyzer), "Dialysate_Bath": float(dialysate_bath),
        "Dry_Weight": dry_weight, "Pre_Weight": pre_weight, "Weight_Gain": weight_gain,
        "Dialysate_Temperature": dialysate_temp, "Blood_Flow": blood_flow,
        "Dialysate_Flow": dialysate_flow, "Dialysate_Conductivity": dialysate_cond,
        "Arterial_Pressure": arterial_pressure, "Venous_Pressure": venous_pressure,
        "Transmembrane_Pressure": tmp, "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp, "Pulse": pulse, "Patient_Temperature": patient_temp,
    }

    st.divider()

    # 3. Live Risk Assessment & Prescriptive AI
    try:
        pred = engine.predict_single(payload)

        if pred.risk_tier == "High Risk":
            st.markdown(
                f"""
                <div class="alert-danger">
                    <strong>🚨 HIGH RISK ALERT ({pred.crash_probability:.2%} &ge; {max_f1_thresh:.2f})</strong><br>
                    High likelihood of intradialytic hypotension. Review the AI prescription below or alert the attending nephrologist.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif pred.hypotension_alert:
            st.markdown(
                f"""
                <div class="alert-warning">
                    <strong>⚠️ MODERATE RISK — CLINICAL SAFETY ALERT ({pred.crash_probability:.2%} &ge; {clinical_thresh:.2f})</strong><br>
                    Patient falls in the early-warning safety zone. Increase BP monitoring frequency and consider preventive machine adjustments.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="alert-safe">
                    <strong>✅ LOW RISK — HEMODYNAMICALLY STABLE ({pred.crash_probability:.2%} &lt; {clinical_thresh:.2f})</strong><br>
                    Pre-dialysis profile sits safely below the clinical alert threshold.
                </div>
                """,
                unsafe_allow_html=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Crash Risk", f"{pred.crash_probability:.2%}")
        m2.metric("Risk Tier", pred.risk_tier.split(" (")[0])
        m3.metric("Alert Triggered", "YES" if pred.hypotension_alert else "NO")
        m4.metric("Safety Cutoff", f"{pred.clinical_threshold:.2%}")

        render_three_tier_gauge(pred.crash_probability, clinical_thresh, max_f1_thresh)

        # Live AI Prescription Button (Only shown when an alert is active)
        if pred.hypotension_alert:
            st.markdown("#### 🤖 Live Prescriptive AI (Raw Clinical Units)")
            if st.button("✨ Generate Live AI Prescription", type="primary"):
                with st.spinner("Searching safe dialysis machine adjustments..."):
                    rx = generate_live_prescription(engine, payload, pred.crash_probability)

                if not rx["changes"]:
                    st.warning(
                        "No machine-only adjustment reduced risk for this profile. "
                        "Direct clinical escalation (UF hold, saline bolus, or physician review) is recommended."
                    )
                else:
                    badge = (
                        f"RESCUED BELOW {clinical_thresh:.2f}"
                        if rx["rescued"]
                        else "PARTIAL MITIGATION — CLINICAL ESCALATION ADVISED"
                    )
                    lines = []
                    for lev, new_v in rx["changes"].items():
                        cfg = RAW_LEVER_CONFIG[lev]
                        orig_v = float(payload[lev])
                        action = "Increase" if new_v > orig_v else "Lower"
                        lines.append(
                            f"• <strong>{action} {cfg['label']}:</strong> "
                            f"<code>{orig_v:.1f} {cfg['unit']}</code> &rarr; <code>{new_v:.1f} {cfg['unit']}</code>"
                        )
                    st.markdown(
                        f"""
                        <div class="rx-box">
                            <strong>Recommended Bedside Adjustment ({badge})</strong><br>
                            <strong>Projected Risk Shift:</strong> {pred.crash_probability:.2%} &rarr; {rx['new_prob']:.2%}
                            (<strong>-{rx['risk_drop']:.2%}</strong>)<br><br>
                            {"<br>".join(lines)}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    except ValidationError as exc:
        st.error("⚠️ Clinical Input Validation Failed (Pydantic Guardrail):")
        for err in exc.errors():
            loc = " → ".join(str(x) for x in err["loc"]) if err["loc"] else "Physiological Check"
            st.warning(f"**{loc}:** {err['msg']}")

# ==============================================================================
# TAB 2: BATCH WARD SCREENING (POWERED BY InferenceEngine.predict_batch)
# ==============================================================================
with tab_batch:
    st.subheader("Batch Dialysis Ward Screening")
    st.caption(
        "Upload a CSV containing the 18 raw clinical features to triage an entire dialysis shift "
        "using vectorized batch inference."
    )

    sample_df = pd.DataFrame([
        {
            "Sex": 1, "Age": 55.0, "Dialyzer": 1.0, "Dialysate_Bath": 1.0,
            "Dry_Weight": 72.0, "Pre_Weight": 73.5, "Weight_Gain": 1.5,
            "Dialysate_Temperature": 36.5, "Blood_Flow": 320.0, "Dialysate_Flow": 500.0,
            "Dialysate_Conductivity": 14.0, "Arterial_Pressure": -130.0,
            "Venous_Pressure": 135.0, "Transmembrane_Pressure": 85.0,
            "Systolic_BP": 138.0, "Diastolic_BP": 82.0, "Pulse": 74.0, "Patient_Temperature": 36.6,
        },
        {
            "Sex": 0, "Age": 74.0, "Dialyzer": 1.0, "Dialysate_Bath": 2.0,
            "Dry_Weight": 58.0, "Pre_Weight": 61.8, "Weight_Gain": 3.8,
            "Dialysate_Temperature": 37.0, "Blood_Flow": 300.0, "Dialysate_Flow": 500.0,
            "Dialysate_Conductivity": 13.8, "Arterial_Pressure": -160.0,
            "Venous_Pressure": 145.0, "Transmembrane_Pressure": 165.0,
            "Systolic_BP": 96.0, "Diastolic_BP": 58.0, "Pulse": 92.0, "Patient_Temperature": 36.8,
        },
    ])

    st.download_button(
        label="⬇️ Download Sample Ward CSV Template",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_ward_sessions.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Raw Patient Sessions CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in engine.expected_features if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
            else:
                records = batch_df[engine.expected_features].to_dict(orient="records")
                preds = engine.predict_batch(records)

                results_df = batch_df.copy()
                results_df.insert(0, "Risk_Tier", [p.risk_tier for p in preds])
                results_df.insert(0, "Hypotension_Alert", [p.hypotension_alert for p in preds])
                results_df.insert(0, "Crash_Probability", [p.crash_probability for p in preds])
                results_df = results_df.sort_values("Crash_Probability", ascending=False).reset_index(drop=True)

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Patients Screened", len(results_df))
                b2.metric("High Risk", int((results_df["Risk_Tier"] == "High Risk").sum()))
                b3.metric("Clinical Safety Alerts", int(results_df["Risk_Tier"].str.startswith("Moderate").sum()))
                b4.metric("Low Risk", int((results_df["Risk_Tier"] == "Low Risk").sum()))

                st.dataframe(results_df, use_container_width=True, height=380)

                st.download_button(
                    label="⬇️ Export Triage Results CSV",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="ward_triage_predictions.csv",
                    mime="text/csv",
                )
        except ValidationError as exc:
            st.error("⚠️ One or more rows failed Pydantic clinical validation:")
            for err in exc.errors()[:5]:
                st.warning(f"• {err['msg']}")
        except Exception as exc:
            st.error(f"Batch screening failed: {exc}")


# ==============================================================================
# TAB 3: HOLDOUT AUDIT & AI COUNTERFACTUAL SIMULATOR (PROTOTYPE v1 PRESERVED)
# ==============================================================================
with tab_cf:
    st.subheader("Holdout Cohort Audit & 6-Dial What-If Counterfactual Simulator")

    ctrl_col1, ctrl_col2 = st.columns([1.0, 1.4], gap="large")

    with ctrl_col1:
        threshold_mode = st.radio(
            "Operational Cutoff Mode",
            options=[
                f"Clinical Safety Mode ({clinical_thresh:.2f} | >=65% Recall)",
                f"Balanced Max-F1 Mode ({max_f1_thresh:.2f} | Max F1)",
            ],
            index=0,
            key="cf_threshold_mode",
        )
        active_threshold = clinical_thresh if "Clinical Safety" in threshold_mode else max_f1_thresh

    with ctrl_col2:
        selection_mode = st.radio(
            "Patient Selection Method",
            ["Clinical Case-Study Presets", "Search Any Holdout Session"],
            index=0,
            horizontal=True,
            key="cf_selection_mode",
        )

        if selection_mode == "Clinical Case-Study Presets":
            preset_map = {
                f"Rescued Crash — Session {archetypes['tp_rescued']['df_index']} ({archetypes['tp_rescued']['pred_prob']:.1%})": archetypes["tp_rescued"]["pos_idx"],
                f"High-Risk Caught Crash — Session {archetypes['tp_caught']['df_index']} ({archetypes['tp_caught']['pred_prob']:.1%})": archetypes["tp_caught"]["pos_idx"],
                f"False Alarm Audit — Session {archetypes['fp_alarm']['df_index']} ({archetypes['fp_alarm']['pred_prob']:.1%})": archetypes["fp_alarm"]["pos_idx"],
                f"Stable Baseline — Session {archetypes['tn_stable']['df_index']} ({archetypes['tn_stable']['pred_prob']:.1%})": archetypes["tn_stable"]["pos_idx"],
            }
            chosen_preset = st.selectbox("Clinical Archetype", list(preset_map.keys()))
            pos_idx = int(preset_map[chosen_preset])
        else:
            high_risk_only = st.checkbox(
                f"Filter to High-Risk Sessions Only (Prob >= {clinical_thresh:.2f})", value=True
            )
            candidate_positions = (
                np.where(y_prob_test >= clinical_thresh)[0] if high_risk_only else np.arange(len(X_test))
            )
            session_options = [
                f"Session {int(X_test.index[p])} (Row #{p} | Orig Risk: {y_prob_test[p]:.1%})"
                for p in candidate_positions
            ]
            chosen_str = st.selectbox("Holdout Session", session_options)
            pos_idx = int(candidate_positions[session_options.index(chosen_str)])

    orig_row = X_test.iloc[pos_idx]
    session_id = int(X_test.index[pos_idx])
    actual_outcome = int(y_test.iloc[pos_idx])
    orig_prob = float(y_prob_test[pos_idx])

    if st.session_state.get("active_pos_idx") != pos_idx:
        st.session_state["active_pos_idx"] = pos_idx
        set_sliders_from_series(orig_row)

    st.markdown(
        f"**Active Patient:** `Session ID {session_id}` (Holdout Row `#{pos_idx}`) &nbsp;|&nbsp; "
        f"**Actual Outcome:** `{'IDH Crash (1)' if actual_outcome == 1 else 'Stable Session (0)'}` &nbsp;|&nbsp; "
        f"**Active Cutoff:** `{active_threshold:.2f}`"
    )

    cf_match = cf_df[cf_df["Holdout_Row"] == pos_idx]
    has_cf = len(cf_match) > 0 and pos_idx in cf_rows_dict

    sim_row = orig_row.copy()
    for lev in actionable_levers:
        sim_row[lev] = float(st.session_state[f"slider_{lev}"])

    sim_df = pd.DataFrame([sim_row.values], columns=feature_names)
    sim_prob = float(champion_model.predict_proba(sim_df)[0, 1])
    delta_prob = sim_prob - orig_prob
    modified_dials = [
        lev for lev in actionable_levers
        if not np.isclose(float(sim_row[lev]), float(orig_row[lev]), atol=1e-4)
    ]

    if sim_prob >= active_threshold:
        if has_cf and not bool(cf_match.iloc[0]["Rescued_Below_051"]):
            st.markdown(
                f"""
                <div class="alert-warning">
                    <strong>⚠️ SEVERE HIGH-RISK ALERT ({sim_prob:.2%} &ge; {active_threshold:.2f}) — MEDICAL ESCALATION ADVISED</strong><br>
                    Baseline frailty keeps this session above the safety cutoff even after 3 machine dial adjustments.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="alert-danger">
                    <strong>🚨 HIGH-RISK IDH CRASH ALERT ({sim_prob:.2%} &ge; {active_threshold:.2f})</strong><br>
                    Click <strong>Auto-Apply AI Prescription</strong> or adjust the bedside machine dials below.
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

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline Crash Risk", f"{orig_prob:.2%}")
    k2.metric("Simulated What-If Risk", f"{sim_prob:.2%}", delta=f"{delta_prob:+.2%}", delta_color="inverse")
    k3.metric("Active Safety Cutoff", f"{active_threshold:.2%}")
    k4.metric("Machine Dials Modified", f"{len(modified_dials)} / {len(actionable_levers)}")

    st.divider()

    col_left, col_right = st.columns([1.05, 1.25], gap="large")

    with col_left:
        st.markdown("#### 🤖 AI Counterfactual Prescription")
        if has_cf:
            r = cf_match.iloc[0]
            rescued_flag = bool(r["Rescued_Below_051"])
            status_badge = f"RESCUED BELOW {clinical_thresh:.2f}" if rescued_flag else "PARTIAL MITIGATION (ESCALATE)"

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
                if st.button("✨ Auto-Apply AI Prescription", type="primary", use_container_width=True):
                    set_sliders_from_series(cf_rows_dict[pos_idx])
                    st.rerun()
            with btn_col2:
                if st.button("↩️ Reset to Original Dials", use_container_width=True):
                    set_sliders_from_series(orig_row)
                    st.rerun()
        else:
            st.info(
                f"This session already starts safely below the {clinical_thresh:.2f} Clinical Safety Threshold. "
                "You can still test manual 'What-If' changes on the right."
            )
            if st.button("↩️ Reset to Original Dials", use_container_width=True):
                set_sliders_from_series(orig_row)
                st.rerun()

        fig_bar, ax_bar = plt.subplots(figsize=(6.2, 2.6))
        bar_labels = ["Baseline Risk", "Simulated Risk"]
        bar_vals = [orig_prob, sim_prob]
        bar_colors = [
            "#dc2626" if orig_prob >= active_threshold else "#10b981",
            "#dc2626" if sim_prob >= active_threshold else "#10b981",
        ]
        bars = ax_bar.barh(bar_labels, bar_vals, color=bar_colors, edgecolor="black", height=0.5)
        ax_bar.axvline(active_threshold, color="#111827", linestyle="--", linewidth=2.0, label=f"Cutoff ({active_threshold:.2f})")
        for b in bars:
            w = b.get_width()
            ax_bar.text(w + 0.02, b.get_y() + b.get_height() / 2.0, f"{w:.1%}", va="center", fontweight="bold")
        ax_bar.set_xlim(0, 1.05)
        ax_bar.set_xlabel("Predicted IDH Crash Probability")
        ax_bar.legend(loc="lower right", fontsize=8.5)
        ax_bar.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig_bar, clear_figure=True)

    with col_right:
        st.markdown("#### 🎛️ Actionable Dialysis Machine Levers (Normalized)")
        s_cols = st.columns(2)
        for idx, lev in enumerate(actionable_levers):
            cfg = get_lever_config(lev)
            disp_name = lever_display_names.get(lev, lev)
            orig_v = float(orig_row[lev])

            with s_cols[idx % 2]:
                label_txt = f"{disp_name} (Orig: {orig_v:.3f})"
                help_txt = f"Empirical 5th–95th percentile range: [{cfg['safe_min']:.3f}, {cfg['safe_max']:.3f}]"
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

        with st.expander("View Locked Immutable Patient Baselines (Normalized)"):
            key_vitals = [
                "Systolic_BP", "Diastolic_BP", "Pulse_Pressure", "Mean_Arterial_Pressure",
                "Relative_Weight_Gain", "Pre_Weight", "Dry_Weight", "Age", "Pulse", "Patient_Temperature",
            ]
            vital_rows = [
                {"Clinical Feature": f, "Normalized Value": f"{float(orig_row[f]):.3f}"}
                for f in key_vitals if f in orig_row.index
            ]
            st.dataframe(pd.DataFrame(vital_rows), hide_index=True, use_container_width=True)