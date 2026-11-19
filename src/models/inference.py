from pathlib import Path
from typing import List, Union
import joblib
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class PatientSessionInput(BaseModel):
    """Validated clinical input schema matching the 18 expected pipeline features."""

    Sex: int = Field(..., ge=0, le=2, description="Encoded patient sex")
    Age: float = Field(..., ge=18.0, le=120.0, description="Patient age in years")
    Dialyzer: float = Field(..., ge=0.0, description="Encoded dialyzer type")
    Dialysate_Bath: float = Field(..., ge=0.0, description="Encoded dialysate bath type")
    Dry_Weight: float = Field(..., gt=20.0, le=250.0, description="Target dry weight (kg)")
    Pre_Weight: float = Field(..., gt=20.0, le=260.0, description="Pre-dialysis weight (kg)")
    Weight_Gain: float = Field(..., ge=-5.0, le=20.0, description="Interdialytic weight gain (kg)")
    Dialysate_Temperature: float = Field(..., ge=34.0, le=39.0, description="Dialysate temp (°C)")
    Blood_Flow: float = Field(..., ge=50.0, le=600.0, description="Blood flow rate (mL/min)")
    Dialysate_Flow: float = Field(..., ge=100.0, le=1000.0, description="Dialysate flow rate (mL/min)")
    Dialysate_Conductivity: float = Field(..., ge=12.0, le=16.0, description="Conductivity (mS/cm)")
    Arterial_Pressure: float = Field(..., ge=-400.0, le=100.0, description="Arterial pressure (mmHg)")
    Venous_Pressure: float = Field(..., ge=0.0, le=400.0, description="Venous pressure (mmHg)")
    Transmembrane_Pressure: float = Field(..., ge=-100.0, le=500.0, description="TMP (mmHg)")
    Systolic_BP: float = Field(..., ge=50.0, le=260.0, description="Systolic BP (mmHg)")
    Diastolic_BP: float = Field(..., ge=20.0, le=160.0, description="Diastolic BP (mmHg)")
    Pulse: float = Field(..., ge=30.0, le=220.0, description="Heart rate (bpm)")
    Patient_Temperature: float = Field(..., ge=34.0, le=42.0, description="Body temperature (°C)")

    @model_validator(mode="after")
    def check_blood_pressure_logic(self) -> "PatientSessionInput":
        if self.Systolic_BP <= self.Diastolic_BP:
            raise ValueError("Systolic_BP must be strictly greater than Diastolic_BP.")
        return self


class PredictionResponse(BaseModel):
    """Structured inference output returned by InferenceEngine."""

    crash_probability: float
    hypotension_alert: bool
    risk_tier: str
    clinical_threshold: float
    model_name: str


class InferenceEngine:
    """Production inference wrapper around champion_inference_pipeline.joblib."""

    def __init__(self, artifact_path: Union[str, Path] = "models/champion_inference_pipeline.joblib"):
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")

        artifact = joblib.load(artifact_path)
        self.pipeline = artifact["pipeline"]
        self.clinical_threshold: float = float(artifact["clinical_threshold"])
        self.max_f1_threshold: float = float(artifact["max_f1_threshold"])
        self.expected_features: List[str] = list(artifact["expected_features"])
        self.model_name: str = artifact["model_name"]

    def _classify_tier(self, prob: float) -> str:
        if prob >= self.max_f1_threshold:
            return "High Risk"
        if prob >= self.clinical_threshold:
            return "Moderate Risk (Clinical Safety Alert)"
        return "Low Risk"

    def predict_single(self, session: Union[PatientSessionInput, dict]) -> PredictionResponse:
        validated = session if isinstance(session, PatientSessionInput) else PatientSessionInput(**session)
        df = pd.DataFrame([validated.model_dump()])[self.expected_features]

        prob = float(self.pipeline.predict_proba(df)[0, 1])
        return PredictionResponse(
            crash_probability=round(prob, 4),
            hypotension_alert=prob >= self.clinical_threshold,
            risk_tier=self._classify_tier(prob),
            clinical_threshold=self.clinical_threshold,
            model_name=self.model_name,
        )

    def predict_batch(self, sessions: List[Union[PatientSessionInput, dict]]) -> List[PredictionResponse]:
        validated = [
            s if isinstance(s, PatientSessionInput) else PatientSessionInput(**s)
            for s in sessions
        ]
        df = pd.DataFrame([s.model_dump() for s in validated])[self.expected_features]
        probs = self.pipeline.predict_proba(df)[:, 1]

        return [
            PredictionResponse(
                crash_probability=round(float(p), 4),
                hypotension_alert=float(p) >= self.clinical_threshold,
                risk_tier=self._classify_tier(float(p)),
                clinical_threshold=self.clinical_threshold,
                model_name=self.model_name,
            )
            for p in probs
        ]