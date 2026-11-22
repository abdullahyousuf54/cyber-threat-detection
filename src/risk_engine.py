from dataclasses import dataclass
import numpy as np
import pandas as pd

LABEL_SEVERITY = {
    "Normal": 0.00,
    "Probe": 0.45,
    "DoS": 0.70,
    "R2L": 0.85,
    "U2R": 1.00,
    "Other": 0.60,
}

@dataclass
class RiskResult:
    threat_family: str
    supervised_confidence: float
    anomaly_score: float
    risk_score: float
    risk_level: str

    def as_dict(self):
        return {
            "threat_family": self.threat_family,
            "supervised_confidence": float(self.supervised_confidence),
            "anomaly_score": float(self.anomaly_score),
            "risk_score": float(self.risk_score),
            "risk_level": self.risk_level,
        }

def risk_level(score):
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MEDIUM"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"

def calculate_risk(threat_family, supervised_confidence, anomaly_score):
    threat_family = str(threat_family)
    confidence = float(np.clip(supervised_confidence, 0.0, 1.0))
    anomaly = float(np.clip(anomaly_score, 0.0, 1.0))
    severity = LABEL_SEVERITY.get(threat_family, LABEL_SEVERITY["Other"])

    supervised_signal = severity * confidence
    score = 0.65 * supervised_signal + 0.35 * anomaly

    if threat_family == "Normal":
        score = 0.35 * anomaly

    score = float(np.clip(score, 0.0, 1.0))
    return RiskResult(
        threat_family=threat_family,
        supervised_confidence=confidence,
        anomaly_score=anomaly,
        risk_score=score,
        risk_level=risk_level(score),
    )

def add_risk_columns(df):
    out = df.copy()
    values = [
        calculate_risk(
            row["rf_prediction"],
            row["rf_confidence"],
            row["anomaly_score"],
        ).as_dict()
        for _, row in out.iterrows()
    ]
    risks = pd.DataFrame(values, index=out.index)
    for col in ["risk_score", "risk_level", "threat_family"]:
        out[col] = risks[col]
    return out
