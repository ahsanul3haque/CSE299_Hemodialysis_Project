import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClinicalDataCleaner(BaseEstimator, TransformerMixin):
    """
    Handles translation, target binarization, memory optimization, 
    and removal of leakable/zero-variance features.
    """
    def __init__(self):
        self.column_translations = {
            'D_Hipotension': 'Hypotension', 'D_Sexo': 'Sex', 'D_Edad': 'Age',
            'D_Dializador': 'Dialyzer', 'D_Bano': 'Dialysate_Bath',
            'D_Tecnica': 'Technique', 'D_PesoSeco': 'Dry_Weight',
            'D_PesoPre': 'Pre_Weight', 'D_PesoPost': 'Post_Weight',
            'D_Ganancia': 'Weight_Gain', 'D_TemperaturaBano': 'Dialysate_Temperature',
            'D_VolumenReposicion': 'Replacement_Volume', 'D_Kt': 'Kt',
            'D_FlujoSangre': 'Blood_Flow', 'D_FlujoBano': 'Dialysate_Flow',
            'D_ConductividadBano': 'Dialysate_Conductivity',
            'D_ConductividadBicarbonato': 'Bicarbonate_Conductivity',
            'D_PresionArterial': 'Arterial_Pressure', 'D_PresionVenosa': 'Venous_Pressure',
            'D_PTM': 'Transmembrane_Pressure', 'D_TAS': 'Systolic_BP',
            'D_TAD': 'Diastolic_BP', 'D_Pulso': 'Pulse', 'D_UF': 'Ultrafiltration',
            'D_Temperatura': 'Patient_Temperature'
        }
        self.leakable_cols = [
            'Post_Weight', 'Kt', 'Replacement_Volume', 'Ultrafiltration', 
            'Technique', 'Bicarbonate_Conductivity'
        ]
        self.target_map = {
            '1': 0, '1.0': 0, 'NO': 0,
            '2': 1, '2.0': 1, 'SI': 1, 'YES': 1
        }
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Cleaning data: translating columns and removing leakages...")
        X_out = X.copy()
        
        #Translate Columns
        X_out = X_out.rename(columns=self.column_translations)
        
        #Target Binarization
        if "Hypotension" in X_out.columns:
            clean_series = X_out["Hypotension"].astype(str).str.strip().str.upper()
            X_out["Hypotension"] = clean_series.map(self.target_map)
            # Drop unmappable targets
            X_out = X_out.dropna(subset=["Hypotension"])
            
        #Memory Optimization
        if 'Sex' in X_out.columns:
            X_out["Sex"] = X_out["Sex"].fillna(0).astype('int8')
        if 'Age' in X_out.columns:
            X_out["Age"] = X_out["Age"].fillna(0).astype('int8')
            
        #Remove Leakable Features
        cols_to_drop = [c for c in self.leakable_cols if c in X_out.columns]
        X_out = X_out.drop(columns=cols_to_drop)
        
        self.feature_names_out_ = X_out.columns.tolist()
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class ClinicalImputer(BaseEstimator, TransformerMixin):
    """
    Calculates medians strictly on the training fold during fit() 
    to prevent data leakage into the validation/test folds.
    """
    def __init__(self):
        self.medians_ = {}
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        numeric_cols = X.select_dtypes(include=['number']).columns.drop('Hypotension', errors='ignore')
        self.medians_ = X[numeric_cols].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Imputing missing values using training fold medians...")
        X_out = X.copy()
        for col, median_val in self.medians_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(median_val)
        
        self.feature_names_out_ = X_out.columns.tolist()
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class HemodynamicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Combines Phase 1 clinical formulas with Week 6 SHAP-informed non-linear interactions.
    """
    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Engineering Phase 1 and Week 6 SHAP interaction features...")
        X_out = X.copy()

        # PHASE 1: Legacy Clinical Features
        if 'Systolic_BP' in X_out.columns and 'Diastolic_BP' in X_out.columns:
            X_out["Pulse_Pressure"] = X_out["Systolic_BP"] - X_out["Diastolic_BP"]
            X_out["Mean_Arterial_Pressure"] = X_out["Diastolic_BP"] + (X_out["Pulse_Pressure"] / 3.0)

        if 'Pulse' in X_out.columns and 'Systolic_BP' in X_out.columns:
            X_out["Shock_Index"] = X_out["Pulse"] / (X_out["Systolic_BP"] + self.eps)

        if 'Pre_Weight' in X_out.columns and 'Dry_Weight' in X_out.columns:
            X_out["Relative_Weight_Gain"] = (X_out["Pre_Weight"] - X_out["Dry_Weight"]) / (X_out["Dry_Weight"] + self.eps)

        if 'Weight_Gain' in X_out.columns and 'Blood_Flow' in X_out.columns:
            X_out["UF_Strain_Proxy"] = X_out["Weight_Gain"] / (X_out["Blood_Flow"] + self.eps)

       
        #SHAP-Informed Interactions
        if 'Systolic_BP' in X_out.columns and 'Relative_Weight_Gain' in X_out.columns:
            X_out['BP_Weight_Interaction'] = X_out['Systolic_BP'] * X_out['Relative_Weight_Gain']

        if 'Pulse_Pressure' in X_out.columns and 'Systolic_BP' in X_out.columns:
            X_out['Pulse_BP_Ratio'] = X_out['Pulse_Pressure'] / (X_out['Systolic_BP'] + self.eps)

        if 'Transmembrane_Pressure' in X_out.columns and 'Dry_Weight' in X_out.columns:
            X_out['TMP_Weight_Ratio'] = X_out['Transmembrane_Pressure'] / (X_out['Dry_Weight'] + self.eps)

        self.feature_names_out_ = X_out.columns.tolist()
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Learns 1st and 99th percentiles strictly on the training fold 
    to prevent data leakage, replacing the leaky global clipping method.
    """
    def __init__(self):
        self.bounds_ = {}
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        numeric_cols = X.select_dtypes(include=['number']).columns.drop('Hypotension', errors='ignore')
        
        for col in numeric_cols:
            lower = X[col].quantile(0.01)
            upper = X[col].quantile(0.99)
            self.bounds_[col] = (lower, upper)
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Capping outliers based on learned training bounds...")
        X_out = X.copy()
        
        for col, (lower, upper) in self.bounds_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].clip(lower=lower, upper=upper)
                
        self.feature_names_out_ = X_out.columns.tolist()
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """
    Wraps MinMaxScaler to return a Pandas DataFrame with original column names
    instead of stripping them into a bare Numpy array.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.scale_cols_ = None
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.scale_cols_ = X.select_dtypes(include=['number']).columns.drop(['Hypotension', 'Sex'], errors='ignore')
        self.scaler.fit(X[self.scale_cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Scaling numeric features (MinMax)...")
        X_out = X.copy()
        if self.scale_cols_ is not None:
            X_out[self.scale_cols_] = self.scaler.transform(X_out[self.scale_cols_])
        
        self.feature_names_out_ = X_out.columns.tolist()
        return X_out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)