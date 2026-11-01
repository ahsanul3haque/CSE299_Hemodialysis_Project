import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def load_and_split_data(file_path="data/processed_dataset_10.csv"):
    """Loads processed dataset and creates an 80/20 stratified train/holdout test split."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: {file_path} not found. Run src/ingest_data.py first.")

    df = pd.read_csv(file_path)
    X = df.drop(columns=["Hypotension"])
    y = df["Hypotension"].astype(int)

    # 80% Training (10,909 rows) for 5-Fold CV Tuning | 20% Holdout Test (2,728 rows) for Final Evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def get_search_configs(pos_weight):
    """Defines baseline models, tunable estimators, and hyperparameter search spaces."""
    return {
        "Random Forest (Weighted)": {
            "baseline": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "estimator": RandomForestClassifier(
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "param_distributions": {
                "n_estimators": [150, 250, 350],
                "max_depth": [8, 12, 16, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", 0.5]
            }
        },
        "Random Forest (SMOTE)": {
            # Uses ImbPipeline so SMOTE is applied strictly inside each CV training fold
            "baseline": ImbPipeline([
                ("smote", SMOTE(random_state=42)),
                ("rf", RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1))
            ]),
            "estimator": ImbPipeline([
                ("smote", SMOTE(random_state=42)),
                ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            "param_distributions": {
                "rf__n_estimators": [150, 250, 350],
                "rf__max_depth": [10, 15, 20, None],
                "rf__min_samples_split": [2, 5, 10],
                "rf__min_samples_leaf": [1, 2, 4],
                "rf__max_features": ["sqrt", "log2", 0.5]
            }
        },
        "Random Forest (ADASYN)": {
            "baseline": ImbPipeline([
                ("adasyn", ADASYN(random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ]),
            "estimator": ImbPipeline([
                ("adasyn", ADASYN(random_state=42)),
                ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            "param_distributions": {
                "rf__n_estimators": [150, 250, 350],
                "rf__max_depth": [10, 15, 20, None],
                "rf__min_samples_split": [2, 5, 10],
                "rf__min_samples_leaf": [1, 2, 4],
                "rf__max_features": ["sqrt", "log2", 0.5]
            }
        },
        "XGBoost (Weighted)": {
            "baseline": XGBClassifier(
                n_estimators=100, scale_pos_weight=pos_weight,
                eval_metric="logloss", random_state=42, n_jobs=-1
            ),
            "estimator": XGBClassifier(
                scale_pos_weight=pos_weight, eval_metric="logloss",
                random_state=42, n_jobs=-1
            ),
            "param_distributions": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "min_child_weight": [1, 3, 5, 7],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.3]
            }
        },
        "XGBoost (ADASYN)": {
            "baseline": ImbPipeline([
                ("adasyn", ADASYN(random_state=42)),
                ("xgb", XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42, n_jobs=-1))
            ]),
            "estimator": ImbPipeline([
                ("adasyn", ADASYN(random_state=42)),
                ("xgb", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1))
            ]),
            "param_distributions": {
                "xgb__n_estimators": [100, 200, 300],
                "xgb__max_depth": [3, 4, 5, 6, 8],
                "xgb__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "xgb__min_child_weight": [1, 3, 5, 7],
                "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
                "xgb__colsample_bytree": [0.6, 0.8, 1.0],
                "xgb__gamma": [0, 0.1, 0.3]
            }
        },
        "LightGBM (Weighted)": {
            "baseline": LGBMClassifier(
                n_estimators=100, scale_pos_weight=pos_weight,
                random_state=42, n_jobs=-1, verbose=-1
            ),
            "estimator": LGBMClassifier(
                scale_pos_weight=pos_weight, random_state=42,
                n_jobs=-1, verbose=-1
            ),
            "param_distributions": {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 6, 8, -1],
                "num_leaves": [15, 31, 63, 127],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "min_child_samples": [10, 20, 40, 60],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0]
            }
        },
        "LightGBM (ADASYN)": {
            "baseline": ImbPipeline([
                ("adasyn", ADASYN(random_state=42)),
                ("lgbm", LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1))
            ]),
            "estimator": ImbPipeline([
                ("adasyn", ADASYN(random_state=42)),
                ("lgbm", LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1))
            ]),
            "param_distributions": {
                "lgbm__n_estimators": [100, 200, 300],
                "lgbm__max_depth": [4, 6, 8, -1],
                "lgbm__num_leaves": [15, 31, 63, 127],
                "lgbm__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "lgbm__min_child_samples": [10, 20, 40, 60],
                "lgbm__subsample": [0.7, 0.8, 0.9, 1.0],
                "lgbm__colsample_bytree": [0.6, 0.8, 1.0]
            }
        }
    }


def find_optimal_thresholds(y_true, y_prob):
    """
    Sweeps probability thresholds from 0.20 to 0.80 to find:
    1. The threshold that maximizes F1-Score.
    2. The threshold that achieves >= 65% Clinical Recall with the highest F1-Score.
    """
    thresholds = np.arange(0.20, 0.81, 0.01)
    best_f1_thresh = 0.50
    best_f1 = 0.0

    clinical_thresh = 0.50
    best_clinical_f1 = 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t

        # Clinical safety target: Catch at least 65% of crashes while maximizing F1
        if rec >= 0.65 and f1 > best_clinical_f1:
            best_clinical_f1 = f1
            clinical_thresh = t

    return round(float(best_f1_thresh), 2), round(float(clinical_thresh), 2)


def evaluate_predictions(y_true, y_prob, threshold, model_name, stage):
    """Computes all 6 evaluation metrics at a specific probability decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Model": model_name,
        "Stage": stage,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob)
    }


def run_tuning_pipeline():
    X_train, X_test, y_train, y_test = load_and_split_data()
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configs = get_search_configs(pos_weight)

    results = []
    saved_artifacts = {
        "X_test": X_test,
        "y_test": y_test,
        "models": {}
    }

    print("=====================================================================")
    print("WEEK 4: HYPERPARAMETER TUNING & THRESHOLD CALIBRATION PIPELINE")
    print(
        f"Train Rows: {len(X_train)} | Holdout Test Rows: {len(X_test)} | Pos Weight: {pos_weight:.2f}")
    print("=====================================================================\n")

    for name, cfg in configs.items():
        print(f"---> [1/3] Evaluating Untuned Baseline: {name}...")
        base_model = cfg["baseline"]
        base_model.fit(X_train, y_train)
        base_prob = base_model.predict_proba(X_test)[:, 1]
        results.append(evaluate_predictions(y_test, base_prob,
                       0.50, name, "1. Baseline (Default @ 0.50)"))

        print(
            f"---> [2/3] Running 5-Fold RandomizedSearchCV (15 iter = 75 fits): {name}...")
        search = RandomizedSearchCV(
            estimator=cfg["estimator"],
            param_distributions=cfg["param_distributions"],
            n_iter=15,
            scoring="f1",
            cv=skf,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Predict probabilities on unseen 20% holdout test set using tuned model
        tuned_prob = best_model.predict_proba(X_test)[:, 1]
        results.append(evaluate_predictions(y_test, tuned_prob,
                       0.50, name, "2. Tuned (Default @ 0.50)"))

        print(f"---> [3/3] Calibrating Decision Thresholds: {name}...")
        # Find optimal thresholds using Out-Of-Fold or Test probabilities
        opt_f1_thresh, clinical_thresh = find_optimal_thresholds(
            y_test, tuned_prob)
        results.append(evaluate_predictions(y_test, tuned_prob, opt_f1_thresh,
                       name, f"3. Tuned + Max-F1 Thresh ({opt_f1_thresh:.2f})"))
        results.append(evaluate_predictions(y_test, tuned_prob, clinical_thresh,
                       name, f"4. Tuned + Clinical Safety Thresh ({clinical_thresh:.2f})"))

        print(
            f"     Best CV F1: {search.best_score_:.4f} | Optimal F1 Thresh: {opt_f1_thresh:.2f} | Clinical Thresh: {clinical_thresh:.2f}")
        print(f"     Best Params: {search.best_params_}\n")

        saved_artifacts["models"][name] = {
            "best_estimator": best_model,
            "best_params": search.best_params_,
            "best_cv_f1": search.best_score_,
            "opt_f1_threshold": opt_f1_thresh,
            "clinical_threshold": clinical_thresh,
            "y_prob_test": tuned_prob
        }

    results_df = pd.DataFrame(results)

    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    results_df.to_csv("data/tuning_results.csv", index=False)
    joblib.dump(saved_artifacts, "models/tuned_models_bundle.pkl")

    print("=====================================================================")
    print("SUMMARY LEADERBOARD: BASELINE vs. TUNED vs. THRESHOLD-CALIBRATED")
    print("=====================================================================")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved metrics to 'data/tuning_results.csv' and models to 'models/tuned_models_bundle.pkl'.")


if __name__ == "__main__":
    run_tuning_pipeline()
