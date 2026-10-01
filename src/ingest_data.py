import pandas as pd
import os


def load_and_validate_data(file_name):
    """Loads the Mendeley dialysis dataset and validates the schema."""
    file_path = os.path.join("data", file_name)

    print(f"--- Ingesting {file_name} ---")

    # 1. Load the raw data
    try:
        # Mendeley datasets often use semicolons
        df = pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        print(
            f"Error: {file_path} not found. Ensure it is in the data/ folder.")
        return

    # 2. Schema Validation (Checking row counts and missingness)
    print(f"Total Patient Sessions (Rows): {df.shape[0]}")
    print(f"Clinical Features (Columns): {df.shape[1]}")

    # 3. Identify the Target Variable (Hypotension label)
    target_cols = [
        col for col in df.columns if 'hipotension' in col.lower() or 'idh' in col.lower()]
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
    for current_file in dataset_files:
        load_and_validate_data(current_file)
        print("\n" + "="*50 + "\n")  # Prints a visual divider between datasets
