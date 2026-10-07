import pandas as pd
pd.set_option('display.max_columns', None)
import os


def load_and_validate_data(file_name):
    """Loads the Mendeley dialysis dataset and validates the schema."""
    file_path = os.path.join("data", file_name)

    print(f"--- Ingesting {file_name} ---")

    # 1. Load the raw data
    try:
        # Mendeley datasets often use semicolons
        df = pd.read_csv(file_path, sep=';', index_col=False)

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

        df=df.rename(columns=column_translations)


        df["Age"]=df["Age"].astype('int8')
        df["Sex"]=df["Sex"].astype('int8')


        print("\n")
    except FileNotFoundError:
        print(
            f"Error: {file_path} not found. Ensure it is in the data/ folder.")
        return

    # 2. Schema Validation (Checking row counts and missingness)
    print(f"Total Patient Sessions (Rows): {df.shape[0]}")
    print(f"Clinical Features (Columns): {df.shape[1]}")

    # 3. Identify the Target Variable (Hypotension label)
    target_cols = [
        col for col in df.columns if 'hypotension' in col.lower() or 'idh' in col.lower()]
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


if __name__ == "__main__":
    # 1. Create a list of all downloaded dataset filenames
    dataset_files = [
        "dataset_1.csv", "dataset_2.csv", "dataset_3.csv",
        "dataset_4.csv", "dataset_5.csv", "dataset_6.csv",
        "dataset_7.csv", "dataset_8.csv", "dataset_9.csv",
        "dataset_10.csv", "dataset_11.csv"
    ]

    # 2. Loop through the list and process each file one by one
    target_file="dataset_10.csv"
    load_and_validate_data(target_file)
    print("\n" + "="*50 + "\n")  # Prints a visual divider between datasets
