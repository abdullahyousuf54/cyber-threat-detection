from pathlib import Path
from datetime import datetime, timezone
import csv
import pandas as pd

FEEDBACK_COLUMNS = [
    "timestamp",
    "risk_level",
    "threat_family",
    "risk_score",
    "anomaly_score",
    "recommended_action",
    "accepted",
    "comment",
]

def append_feedback(
    path,
    risk_level,
    threat_family,
    risk_score,
    anomaly_score,
    recommended_action,
    accepted,
    comment="",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "risk_level": risk_level,
            "threat_family": threat_family,
            "risk_score": float(risk_score),
            "anomaly_score": float(anomaly_score),
            "recommended_action": recommended_action,
            "accepted": bool(accepted),
            "comment": comment,
        })

def load_feedback(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)
    return pd.read_csv(path)
