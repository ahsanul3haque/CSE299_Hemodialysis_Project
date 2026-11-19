import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.inference import InferenceEngine

VIS_DIR = PROJECT_ROOT / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def plot_risk_tier_distribution(y_true, y_prob, clinical_thresh, max_f1_thresh):
    """Plot 1: Holdout probability distribution with Clinical Safety & Max-F1 cutoffs."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bins = np.linspace(0, 1, 41)
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, color="#2b8cbe", label="Stable Session (Actual = 0)", density=True)
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.65, color="#e34a33", label="Hypotension Crash (Actual = 1)", density=True)

    # Shade risk zones
    ax.axvspan(0, clinical_thresh, color="#a8ddb5", alpha=0.15, label="Low Risk Zone")
    ax.axvspan(clinical_thresh, max_f1_thresh, color="#fdbb84", alpha=0.25, label="Moderate Risk (Rescued Zone)")
    ax.axvspan(max_f1_thresh, 1.0, color="#fc9272", alpha=0.15, label="High Risk Zone")

    # Threshold lines
    ax.axvline(clinical_thresh, color="#d95f02", linestyle="--", linewidth=2, label=f"Clinical Safety ({clinical_thresh:.2f})")
    ax.axvline(max_f1_thresh, color="#b30000", linestyle="-.", linewidth=2, label=f"Max-F1 ({max_f1_thresh:.2f})")

    ax.set_title("Week 7 Inference Risk-Tier Distribution (Holdout Sessions)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Crash Probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    out_path = VIS_DIR / "week7_1_risk_tier_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_what_if_sensitivity(engine: InferenceEngine):
    """Plot 2: Live What-If sensitivity curve sweeping Systolic_BP through InferenceEngine."""
    base_patient = {
        "Sex": 1,
        "Age": 68.0,
        "Dialyzer": 1.0,
        "Dialysate_Bath": 2.0,
        "Dry_Weight": 70.0,
        "Pre_Weight": 72.5,
        "Weight_Gain": 2.5,
        "Dialysate_Temperature": 36.5,
        "Blood_Flow": 350.0,
        "Dialysate_Flow": 500.0,
        "Dialysate_Conductivity": 14.0,
        "Arterial_Pressure": -150.0,
        "Venous_Pressure": 140.0,
        "Transmembrane_Pressure": 120.0,
        "Systolic_BP": 120.0,
        "Diastolic_BP": 60.0,  #Kept strictly below min swept Systolic_BP (85)
        "Pulse": 82.0,
        "Patient_Temperature": 36.6,
    }

    sbp_values = np.linspace(85, 165, 41)
    payloads = [{**base_patient, "Systolic_BP": float(sbp)} for sbp in sbp_values]
    responses = engine.predict_batch(payloads)
    probs = [r.crash_probability for r in responses]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(sbp_values, probs, color="#1f78b4", linewidth=2.5, marker="o", markersize=4, label="Predicted Crash Probability")
    ax.axhline(engine.clinical_threshold, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Clinical Safety Alert ({engine.clinical_threshold:.2f})")
    ax.axhline(engine.max_f1_threshold, color="#b30000", linestyle="-.", linewidth=1.8, label=f"High Risk Cutoff ({engine.max_f1_threshold:.2f})")

    ax.fill_between(sbp_values, engine.clinical_threshold, 1.0, color="#fc9272", alpha=0.12)
    ax.set_title("Live 'What-If' Clinical Sensitivity: Systolic BP vs. Crash Risk", fontsize=13, fontweight="bold")
    ax.set_xlabel("Pre-Dialysis Systolic Blood Pressure (mmHg)", fontsize=11)
    ax.set_ylabel("Predicted Crash Probability", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(alpha=0.3)

    out_path = VIS_DIR / "week7_2_what_if_sensitivity.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_latency_benchmark(engine: InferenceEngine):
    """Plot 3: Per-patient inference latency (Single vs. Batch sizes)."""
    sample = {
        "Sex": 1, "Age": 68.0, "Dialyzer": 1.0, "Dialysate_Bath": 2.0,
        "Dry_Weight": 70.0, "Pre_Weight": 72.5, "Weight_Gain": 2.5,
        "Dialysate_Temperature": 36.5, "Blood_Flow": 350.0, "Dialysate_Flow": 500.0,
        "Dialysate_Conductivity": 14.0, "Arterial_Pressure": -150.0,
        "Venous_Pressure": 140.0, "Transmembrane_Pressure": 120.0,
        "Systolic_BP": 110.0, "Diastolic_BP": 65.0, "Pulse": 82.0,
        "Patient_Temperature": 36.6,
    }

    #Warm-up call
    engine.predict_single(sample)

    batch_sizes = [1, 10, 100, 1000]
    ms_per_patient = []

    for n in batch_sizes:
        payloads = [sample] * n
        start = time.perf_counter()
        if n == 1:
            engine.predict_single(sample)
        else:
            engine.predict_batch(payloads)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        ms_per_patient.append(elapsed_ms / n)

    labels = ["Single\n(n=1)", "Batch\n(n=10)", "Batch\n(n=100)", "Batch\n(n=1,000)"]
    colors = ["#e34a33", "#3182bd", "#3182bd", "#3182bd"]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(labels, ms_per_patient, color=colors, width=0.55, edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, ms_per_patient):
        ax.text(bar.get_x() + bar.get_width() / 2, val + (max(ms_per_patient) * 0.02),
                f"{val:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("InferenceEngine Latency Benchmark (Per-Patient Processing Time)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Latency per Patient (milliseconds)", fontsize=11)
    ax.set_ylim(0, max(ms_per_patient) * 1.18)
    ax.grid(axis="y", alpha=0.3)

    out_path = VIS_DIR / "week7_3_latency_benchmark.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_calibration_curve(y_true, y_prob, model_name):
    """Plot 4: Reliability diagram comparing predicted risk vs. observed crash rate."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, marker="s", linewidth=2.2, color="#2c7fb8", label=model_name)

    ax.set_title("Probability Calibration Curve (Reliability Diagram)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Crash Probability", fontsize=11)
    ax.set_ylabel("Observed Fraction of Crashes", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.3)

    out_path = VIS_DIR / "week7_4_calibration_curve.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    engine = InferenceEngine(PROJECT_ROOT / "models/champion_inference_pipeline.joblib")
    bundle = joblib.load(PROJECT_ROOT / "models/tuned_models_bundle.pkl")

    y_true = np.asarray(bundle["y_test"])
    y_prob = np.asarray(bundle["models"][engine.model_name]["y_prob_test"])

    plot_risk_tier_distribution(y_true, y_prob, engine.clinical_threshold, engine.max_f1_threshold)
    plot_what_if_sensitivity(engine)
    plot_latency_benchmark(engine)
    plot_calibration_curve(y_true, y_prob, engine.model_name)
    print("\n[SUCCESS] All 4 Week 7 plots generated in visualizations/")


if __name__ == "__main__":
    main()

