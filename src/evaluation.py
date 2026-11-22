from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .nsl_models import LABELS

def classification_metrics(y_true, pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )
    _, _, f_macro, _ = precision_recall_fscore_support(
        y_true, pred, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "f1_macro": float(f_macro),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(
            y_true, pred, labels=LABELS
        ).tolist(),
    }

def roc_payload(y_true, probabilities):
    y_bin = label_binarize(y_true, classes=LABELS)
    output = {}
    aucs = []

    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
        auc = roc_auc_score(y_bin[:, i], probabilities[:, i])
        output[label] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(auc),
        }
        aucs.append(auc)

    output["macro_auc"] = float(np.mean(aucs))
    return output

def save_evaluation(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
