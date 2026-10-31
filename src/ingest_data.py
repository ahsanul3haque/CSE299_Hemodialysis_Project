import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import engineer_clinical_features

pd.set_option('display.max_columns', None)


def load_and_validate_data(file_name):
    """Loads the Mendeley dialysis dataset and validates the schema."""
    file_path = os.path.join("data", file_name)

    print(f"--- Ingesting {file_name} ---")

    # 1. Load the raw data
    try:
        # Mendeley datasets often use semicolons
        df = pd.read_csv(file_path, sep=';', index_col=False)
    except FileNotFoundError:
        print(
            f"Error: {file_path} not found. Ensure it is in the data/ folder.")
        return

    # 2. Schema Validation (Checking row counts and missingness)
    print(f"Total Patient Sessions (Rows): {df.shape[0]}")
    print(f"Clinical Features (Columns): {df.shape[1]}")

    # 3. Identify the Target Variable (Hypotension label)
    target_cols = [
        col for col in df.columns if 'hypotension' in col.lower() or 'idh' in col.lower()
    ]
    if target_cols:
        target = target_cols[0]
        print(f"\nTarget Variable Found: '{target}'")
        print("Class Imbalance Breakdown (%):")
        print(df[target].value_counts(normalize=True) * 100)
    else:
        print("\nWarning: No explicit hypotension target column found in this subset.")
        print("Available columns:", df.columns.tolist())

    # 4. Missing Data Check
    missing_data = df.isnull().sum()
    print("\nMissing Values per Feature:")
    print(missing_data[missing_data > 0])


def load_and_align_data(file_name):
    file_path = os.path.join("data", file_name)
    print(f"--- Ingesting {file_name} ---")

    try:
        df = pd.read_csv(file_path, sep=';', index_col=False)
        return df
    except FileNotFoundError:
        print(
            f"Error: {file_path} not found. Ensure it is in the data/ folder.")
        return


def translate_columns(df):
    column_translations = {
        'D_Hipotension': 'Hypotension',
        'D_Sexo': 'Sex',
        'D_Edad': 'Age',
        'D_Dializador': 'Dialyzer',
        'D_Bano': 'Dialysate_Bath',
        'D_Tecnica': 'Technique',
        'D_PesoSeco': 'Dry_Weight',
        'D_PesoPre': 'Pre_Weight',
        'D_PesoPost': 'Post_Weight',
        'D_Ganancia': 'Weight_Gain',
        'D_TemperaturaBano': 'Dialysate_Temperature',
        'D_VolumenReposicion': 'Replacement_Volume',
        'D_Kt': 'Kt',
        'D_FlujoSangre': 'Blood_Flow',
        'D_FlujoBano': 'Dialysate_Flow',
        'D_ConductividadBano': 'Dialysate_Conductivity',
        'D_ConductividadBicarbonato': 'Bicarbonate_Conductivity',
        'D_PresionArterial': 'Arterial_Pressure',
        'D_PresionVenosa': 'Venous_Pressure',
        'D_PTM': 'Transmembrane_Pressure',
        'D_TAS': 'Systolic_BP',
        'D_TAD': 'Diastolic_BP',
        'D_Pulso': 'Pulse',
        'D_UF': 'Ultrafiltration',
        'D_Temperatura': 'Patient_Temperature'
    }

    df = df.rename(columns=column_translations)
    return df


def optimize_memory(df):
    if 'Sex' in df.columns:
        df["Sex"] = df["Sex"].fillna(0).astype('int8')
    if 'Age' in df.columns:
        df["Age"] = df["Age"].fillna(0).astype('int8')

    return df


def remove_leakable_features(df):
    # Drop post-treatment leakable features AND zero-variance constant features
    cols_to_drop = [
        'Post_Weight', 'Kt', 'Replacement_Volume', 'Ultrafiltration',  # Leakable
        'Technique', 'Bicarbonate_Conductivity'                       # Zero variance
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df


def hypotension_binarization(df):
    # Convert string values to uppercase and strip whitespace
    clean_series = df["Hypotension"].astype(str).str.strip().str.upper()

    # Map raw numeric codes (1/2) and string labels (NO/SI) to binary classification targets
    # 1 / NO -> 0 (Stable), 2 / SI / YES -> 1 (IDH Crash)
    target_map = {
        '1': 0, '1.0': 0, 'NO': 0,
        '2': 1, '2.0': 1, 'SI': 1, 'YES': 1
    }
    df["Hypotension"] = clean_series.map(target_map)

    print("\n--- Target Variable Binarization Summary ---")
    print("Null count remaining:", df["Hypotension"].isnull().sum())
    print("Class Distribution (%):")
    print(df["Hypotension"].value_counts(normalize=True) * 100)
    print("--------------------------------------------\n")

    return df


def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.drop(
        'Hypotension', errors='ignore')

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def cap_outliers(df):
    numeric_cols = df.select_dtypes(include=['number']).columns

    if 'Hypotension' in numeric_cols:
        numeric_cols = numeric_cols.drop('Hypotension')

    lower_bound = df[numeric_cols].quantile(0.01)
    upper_bound = df[numeric_cols].quantile(0.99)

    df[numeric_cols] = df[numeric_cols].clip(
        lower=lower_bound, upper=upper_bound, axis=1)

    return df


def scale_features(df):
    numeric_cols = df.select_dtypes(include=['number']).columns.drop(
        ['Hypotension', 'Sex'], errors='ignore')

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


if __name__ == "__main__":
    target_file = "dataset_10.csv"
    
    loaded_only_dataset = load_and_align_data(target_file)
    translated_dataset = translate_columns(loaded_only_dataset)
    binarized_dataset = hypotension_binarization(translated_dataset)
    optimized_dataset = optimize_memory(binarized_dataset)
    leakproof_dataset = remove_leakable_features(optimized_dataset)
    imputed_dataset = handle_missing_values(leakproof_dataset)
    engineered_dataset = engineer_clinical_features(imputed_dataset)
    capped_dataset = cap_outliers(engineered_dataset)
    scaled_features_dataset = scale_features(capped_dataset)

    scaled_features_dataset.to_csv("data/processed_dataset_10.csv", index=False)
    print("Pipeline complete! Exported processed_dataset_10.csv to the data/ folder.")