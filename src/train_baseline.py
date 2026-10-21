import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_processed_data(file_path="data/processed_dataset_10.csv"):
    """Loads the preprocessed dataset and separates features from the target."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: {file_path} not found. Run src/ingest_data.py first.")

    df = pd.read_csv(file_path)
    X = df.drop(columns=["Hypotension"])
    y = df["Hypotension"].astype(int)
    return X, y


def get_models(strategy, pos_weight):
    """Initializes the 6 baseline models with strategy-specific class weights."""
    cw = "balanced" if strategy == "Class-Weighted" else None
    xgb_weight = pos_weight if strategy == "Class-Weighted" else 1.0

    return {
        "Dummy Classifier": DummyClassifier(strategy="prior", random_state=42),
        "Logistic Regression": LogisticRegression(class_weight=cw, max_iter=1000, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(class_weight=cw, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, scale_pos_weight=xgb_weight, eval_metric="logloss", random_state=42, n_jobs=-1)
    }


def evaluate_baselines(X, y):
    """Runs 5-fold stratified cross-validation across all models and imbalance strategies."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strategies = ["Unadjusted", "Class-Weighted", "SMOTE"]

    # Calculate negative-to-positive ratio (~3.90) for cost-sensitive weighting
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    pos_weight = neg_count / pos_count

    results = []

    print("Starting 5-Fold Stratified Cross-Validation across 18 experiments...")
    print(
        f"Total rows: {len(X)} | Features: {X.shape[1]} | Negative/Positive Ratio: {pos_weight:.2f}\n")

    for strategy in strategies:
        models = get_models(strategy, pos_weight)

        for model_name, model in models.items():
            fold_metrics = {
                "Accuracy": [], "Precision": [], "Recall": [],
                "F1_Score": [], "ROC_AUC": [], "PR_AUC": []
            }

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Apply SMOTE strictly inside the training fold to prevent data leakage
                if strategy == "SMOTE":
                    smote = SMOTE(random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)

                # Pass sample weights during fit for models without a class_weight parameter
                if strategy == "Class-Weighted" and model_name in ["Dummy Classifier", "Gaussian Naive Bayes"]:
                    sample_weights = compute_sample_weight("balanced", y_train)
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    model.fit(X_train, y_train)

                # Generate class predictions and positive-class probabilities
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

                fold_metrics["Accuracy"].append(accuracy_score(y_val, y_pred))
                fold_metrics["Precision"].append(
                    precision_score(y_val, y_pred, zero_division=0))
                fold_metrics["Recall"].append(
                    recall_score(y_val, y_pred, zero_division=0))
                fold_metrics["F1_Score"].append(
                    f1_score(y_val, y_pred, zero_division=0))
                fold_metrics["ROC_AUC"].append(roc_auc_score(y_val, y_prob))
                fold_metrics["PR_AUC"].append(
                    average_precision_score(y_val, y_prob))

            # Record mean performance across all 5 folds
            results.append({
                "Model": model_name,
                "Strategy": strategy,
                "Accuracy": np.mean(fold_metrics["Accuracy"]),
                "Precision": np.mean(fold_metrics["Precision"]),
                "Recall": np.mean(fold_metrics["Recall"]),
                "F1_Score": np.mean(fold_metrics["F1_Score"]),
                "ROC_AUC": np.mean(fold_metrics["ROC_AUC"]),
                "PR_AUC": np.mean(fold_metrics["PR_AUC"])
            })
            print(f"Completed: {model_name:<22} | Strategy: {strategy}")

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    # 1. Load the preprocessed feature matrix and target vector
    X, y = load_processed_data()

    # 2. Run the 18-experiment benchmark matrix
    benchmark_df = evaluate_baselines(X, y)

    # 3. Display and export final benchmark results
    print("\n--- Week 3 Baseline Model Benchmark Results (5-Fold CV Mean) ---")
    print(benchmark_df.to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))

    os.makedirs("data", exist_ok=True)
    benchmark_df.to_csv("data/baseline_benchmark_results.csv", index=False)
    print("\nSaved benchmark table to data/baseline_benchmark_results.csv")
