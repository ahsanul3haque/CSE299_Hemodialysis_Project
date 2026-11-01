import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score, 
    confusion_matrix, 
    brier_score_loss
)

#Ensure output directory exists
os.makedirs("visualizations", exist_ok=True)

def plot_pr_auc_curves(models_dict, X_test, y_test):
    """
    Computes Brier scores and plots the PR-AUC curve for all provided models.
    """
    plt.figure(figsize=(8, 6))
    
    print("--- Clinical Evaluation Metrics ---")
    for name, model in models_dict.items():
        #Extract probabilities for the positive class (Crash)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        #Brier Score (lower is better, measures probability accuracy)
        brier = brier_score_loss(y_test, y_proba)
        
        #PR-AUC (Precision-Recall Area Under Curve)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        print(f"{name:<20} | PR-AUC: {pr_auc:.4f} | Brier Score: {brier:.4f}")
        
        #Plot the curve
        plt.plot(recall, precision, lw=2, label=f'{name} (PR-AUC = {pr_auc:.2f})')
        
    #Baseline PR-AUC is the fraction of positives in the dataset
    baseline_pr = y_test.mean()
    plt.axhline(y=baseline_pr, color='navy', linestyle='--', label=f'Random Baseline ({baseline_pr:.2f})')
    
    plt.xlabel('Recall (Capturing actual crashes)')
    plt.ylabel('Precision (Reliability of crash alarms)')
    plt.title('Precision-Recall (PR) Curve: Imbalanced Crash Detection')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    plt.savefig("visualizations/pr_auc_curves.png", dpi=300)
    plt.close()
    print("Saved PR-AUC plot to visualizations/pr_auc_curves.png\n")


def plot_custom_threshold_cm(model, model_name, X_test, y_test, threshold=0.5):
    """
    Generates a confusion matrix using an arbitrary probability threshold.
    """
    #Extract probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    
    #Apply custom threshold instead of default 0.5
    y_pred_custom = (y_proba >= threshold).astype(int)
    
    #Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_custom)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Safe', 'Predicted Crash'],
                yticklabels=['Actual Safe', 'Actual Crash'])
    
    plt.title(f'{model_name}: Confusion Matrix (Threshold = {threshold})')
    plt.tight_layout()
    
    safe_name = model_name.replace(' ', '_').lower()
    filename = f"visualizations/{safe_name}_cm_{threshold}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"Saved custom threshold CM to {filename}")