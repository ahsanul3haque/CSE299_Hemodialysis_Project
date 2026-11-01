import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN

def create_versioned_datasets():
    """
    Applies data-level imbalance strategies (SMOTE, ADASYN) strictly to the training split
    and exports them as cleanly versioned MLOps artifacts.
    """
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")

    input_path = os.path.join(data_dir, "processed_dataset_10.csv")
    output_dir = os.path.join(data_dir, "versions")

    os.makedirs(output_dir, exist_ok=True)
    version_tag = "v1" 

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find {input_path}. Your script resolved the path to: {os.path.abspath(input_path)}")
        
    df = pd.read_csv(input_path)
    X = df.drop(columns=["Hypotension"])
    y = df["Hypotension"].astype(int)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    print(f"Original Training Imbalance:\n{y_train.value_counts()}\n")

    #Implement Strategy 1: SMOTE (Synthetic Minority Over-sampling Technique)
    print("Applying SMOTE interpolation...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    #Implement Strategy 2: ADASYN (Adaptive Synthetic Sampling)
    print("Applying ADASYN density distribution...")
    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    #Recombine features (X) and targets (y) into clean DataFrames
    train_baseline_df = pd.concat([X_train, y_train], axis=1)
    train_smote_df = pd.concat([X_train_smote, y_train_smote], axis=1)
    train_adasyn_df = pd.concat([X_train_adasyn, y_train_adasyn], axis=1)
    test_holdout_df = pd.concat([X_test, y_test], axis=1)

    #Export version-controlled CSV artifacts
    baseline_path = os.path.join(output_dir, f"train_baseline_{version_tag}.csv")
    smote_path = os.path.join(output_dir, f"train_smote_{version_tag}.csv")
    adasyn_path = os.path.join(output_dir, f"train_adasyn_{version_tag}.csv")
    test_path = os.path.join(output_dir, f"test_holdout_{version_tag}.csv")

    train_baseline_df.to_csv(baseline_path, index=False)
    train_smote_df.to_csv(smote_path, index=False)
    train_adasyn_df.to_csv(adasyn_path, index=False)
    test_holdout_df.to_csv(test_path, index=False)

    print("✅ MLOps Data Versioning Complete!")
    print(f"Exported: {baseline_path} (Rows: {len(train_baseline_df)})")
    print(f"Exported: {smote_path} (Rows: {len(train_smote_df)})")
    print(f"Exported: {adasyn_path} (Rows: {len(train_adasyn_df)})")
    print(f"Exported (Untouched Vault): {test_path} (Rows: {len(test_holdout_df)})")

if __name__ == "__main__":
    create_versioned_datasets()