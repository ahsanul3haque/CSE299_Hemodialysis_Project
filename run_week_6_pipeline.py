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

#Load raw data (update path if your raw file has a different name)
RAW_DATA_PATH = "data/dataset_10.csv"
df_raw = pd.read_csv(RAW_DATA_PATH, sep=';', index_col=False)

#Clean columns, binarize target, and drop leakable features
cleaner = ClinicalDataCleaner()
df_clean = cleaner.fit_transform(df_raw)

X = df_clean.drop(columns=["Hypotension"])
y = df_clean["Hypotension"].astype(int)

#Stratified 80/20 split BEFORE any fold-dependent fitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

#Leak-free preprocessing pipeline (fit strictly on X_train)
preprocess_pipeline = Pipeline([
    ("imputer", ClinicalImputer()),
    ("feature_engineer", HemodynamicFeatureEngineer()),
    ("outlier_capper", OutlierCapper()),
    ("scaler", DataFrameScaler()),
])

X_train_proc = preprocess_pipeline.fit_transform(X_train)
X_test_proc = preprocess_pipeline.transform(X_test)

#Train Champion LightGBM with Optuna
trainer = ChampionModelTrainer(random_state=42, n_trials=50, cv_folds=5)
trainer.fit(X_train_proc, y_train)

#Export bundle in the exact format explainability.py expects
bundle = trainer.get_champion_bundle(X_test_proc, y_test)

#Extract the Champion LightGBM entry from the bundle
champion_key = "LightGBM (Weighted)"
champion_info = bundle["models"][champion_key]

#Chain fitted preprocessing + fitted Champion LightGBM into one end-to-end Pipeline
final_pipeline = Pipeline([
    ("preprocessing", preprocess_pipeline),
    ("classifier", champion_info["best_estimator"]),
])

#Package the pipeline with inference metadata
production_artifact = {
    "pipeline": final_pipeline,
    "clinical_threshold": champion_info["clinical_threshold"],
    "max_f1_threshold": champion_info["opt_f1_threshold"],
    "expected_features": list(X_train.columns),
    "model_name": champion_key,
}

ARTIFACT_PATH = "models/champion_inference_pipeline.joblib"
joblib.dump(production_artifact, ARTIFACT_PATH)
print(f"Serialized final production pipeline to {ARTIFACT_PATH}")