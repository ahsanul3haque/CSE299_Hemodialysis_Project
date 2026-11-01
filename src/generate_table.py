import pandas as pd
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "tuning_results.csv")

#Load the results
df = pd.read_csv(csv_path)

#Format the numeric columns to 4 decimal places
metrics = ["Threshold", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC", "PR_AUC"]
for col in metrics:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: f"{float(x):.4f}")

#Save the table directly to a Markdown file instead of printing
output_path = os.path.join(script_dir, "..", "new_table.md")
with open(output_path, "w") as f:
    f.write(df.to_markdown(index=False))

print(f"Success! Open the newly created '{output_path}' file to copy your perfectly aligned table.")