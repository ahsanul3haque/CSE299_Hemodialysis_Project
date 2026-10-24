import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


os.makedirs("visualizations", exist_ok=True)
df = pd.read_csv("data/processed_dataset_10.csv")

X = df.drop('Hypotension', axis=1)
y = df['Hypotension']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)


print("Generating and saving 7 visualizations...")
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000).fit(X_train, y_train)
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42).fit(X_train, y_train)
gb_model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)


#DIAGRAMS: CONFUSION MATRICES

def plot_confusion_matrix(model, name, filename):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Predicted Safe', 'Predicted Crash'],
                yticklabels=['Actual Safe', 'Actual Crash'])
    plt.title(f'{name}: Crash Detection Accuracy')
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300)
    plt.close()

plot_confusion_matrix(lr_model, "Logistic Regression", "logistic_confusion_matrix")
plot_confusion_matrix(rf_model, "Random Forest", "rf_confusion_matrix")
plot_confusion_matrix(gb_model, "Gradient Boosting", "gb_confusion_matrix")


#DIAGRAMS: FEATURE IMPORTANCES

def plot_tree_importance(model, name, filename, color):
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette=color)
    plt.title(f'Top 10 Clinical Risk Factors ({name})')
    plt.xlabel('Algorithm Importance Score')
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300)
    plt.close()

def plot_lr_importance(model, name, filename, color):
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    }).sort_values(by='Abs_Coefficient', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=importance_df.head(10), palette=color)
    plt.title(f'Top 10 Clinical Risk Factors ({name})')
    plt.xlabel('Coefficient (Negative = Protective, Positive = Risk)')
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}.png", dpi=300)
    plt.close()

plot_tree_importance(rf_model, "Random Forest", "rf_feature_importance", "Reds_r")
plot_tree_importance(gb_model, "Gradient Boosting", "gb_feature_importance", "Oranges_r")
plot_lr_importance(lr_model, "Logistic Regression", "logistic_feature_importance", "Purples_r")


#DIAGRAM: COMBINED ROC CURVE

plt.figure(figsize=(8, 6))

for model, name, color in [(lr_model, 'Logistic Regression', 'purple'), 
                           (rf_model, 'Random Forest', 'red'), 
                           (gb_model, 'Gradient Boosting', 'orange')]:
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # The random guessing baseline
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve Comparison: Hemodialysis Crash Detection')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("visualizations/combined_roc_curve.png", dpi=300)
plt.close()

print("Success! Check the 'visualizations' folder for all 7 PNG images.")