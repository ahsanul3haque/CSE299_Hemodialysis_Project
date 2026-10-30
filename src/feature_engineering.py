import pandas as pd


def engineer_clinical_features(df):
    """Generates domain-specific hemodynamic and fluid strain features."""
    epsilon = 1e-5

    # 1. Pulse Pressure (arterial stiffness indicator)
    df["Pulse_Pressure"] = df["Systolic_BP"] - df["Diastolic_BP"]

    # 2. Mean Arterial Pressure (organ perfusion pressure)
    df["Mean_Arterial_Pressure"] = df["Diastolic_BP"] + \
        (df["Pulse_Pressure"] / 3.0)

    # 3. Shock Index (heart rate to systolic BP ratio)
    df["Shock_Index"] = df["Pulse"] / (df["Systolic_BP"] + epsilon)

    # 4. Relative Weight Gain (interdialytic fluid overload ratio)
    df["Relative_Weight_Gain"] = (
        df["Pre_Weight"] - df["Dry_Weight"]) / (df["Dry_Weight"] + epsilon)

    # 5. Ultrafiltration Strain Proxy (fluid removal relative to blood flow)
    df["UF_Strain_Proxy"] = df["Weight_Gain"] / (df["Blood_Flow"] + epsilon)

    print(f"Feature engineering complete. Total columns: {df.shape[1]}")
    return df
