import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.preprocessing.transformers import (
    ClinicalDataCleaner,
    ClinicalImputer,
    HemodynamicFeatureEngineer,
    OutlierCapper,
    DataFrameScaler,
)
from src.models.trainer import ChampionModelTrainer

# 1. Load raw data (update path if your raw file has a different name)
RAW_DATA_PATH = "data/dataset_10.csv"
df_raw = pd.read_csv(RAW_DATA_PATH, sep=';', index_col=False)

# 2. Clean columns, binarize target, and drop leakable features
cleaner = ClinicalDataCleaner()
df_clean = cleaner.fit_transform(df_raw)

X = df_clean.drop(columns=["Hypotension"])
y = df_clean["Hypotension"].astype(int)

# 3. Stratified 80/20 split BEFORE any fold-dependent fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 4. Leak-free preprocessing pipeline (fit strictly on X_train)
preprocess_pipeline = Pipeline([
    ("imputer", ClinicalImputer()),
    ("feature_engineer", HemodynamicFeatureEngineer()),
    ("outlier_capper", OutlierCapper()),
    ("scaler", DataFrameScaler()),
])

X_train_proc = preprocess_pipeline.fit_transform(X_train)
X_test_proc = preprocess_pipeline.transform(X_test)

# 5. Train Champion LightGBM with Optuna
trainer = ChampionModelTrainer(random_state=42, n_trials=50, cv_folds=5)
trainer.fit(X_train_proc, y_train)

# 6. Export bundle in the exact format explainability.py expects
bundle = trainer.get_champion_bundle(X_test_proc, y_test)
os.makedirs("models", exist_ok=True)
joblib.dump(bundle, "models/tuned_models_bundle.pkl")
print("Saved new Week 6 bundle to models/tuned_models_bundle.pkl")