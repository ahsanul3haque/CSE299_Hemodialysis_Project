import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


df = pd.read_csv("data/processed_dataset_10.csv")

X = df.drop('Hypotension', axis=1)
y = df['Hypotension']


X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}\n")


models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}


for name, model in models.items():
    print(f"{'='*40}")
    print(f"Evaluating: {name}")
    print(f"{'='*40}")
    
    
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n")


print("="*40)
print("TOP FEATURES: LOGISTIC REGRESSION")
lr_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': models["Logistic Regression"].coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)
print(lr_importance.head(5))

print("\nTOP FEATURES: RANDOM FOREST")
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': models["Random Forest"].feature_importances_
}).sort_values(by='Importance', ascending=False)
print(rf_importance.head(5))

print("\nTOP FEATURES: GRADIENT BOOSTING")
gb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': models["Gradient Boosting"].feature_importances_
}).sort_values(by='Importance', ascending=False)
print(gb_importance.head(5))
