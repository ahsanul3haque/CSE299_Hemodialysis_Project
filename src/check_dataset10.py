import pandas as pd

# Load the raw dataset (Mendeley CSVs use semicolon delimiters)
df_raw = pd.read_csv("data/dataset_10.csv", sep=";")

# 1. Print unique raw values in the column
print("Unique raw values:", df_raw["D_Hipotension"].unique())

# 2. Print exact class count and percentages
print("\nPercentage Breakdown (%):")
print(df_raw["D_Hipotension"].value_counts(normalize=True) * 100)
