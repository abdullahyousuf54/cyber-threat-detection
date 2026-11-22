from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report

from .nsl_preprocessing import fit_transform, transform

LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]

def _metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    _, _, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "f1_macro": float(f_macro),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }

def anomaly_score_from_model(anomaly, X, steepness=12.0):
    """Map IsolationForest.decision_function to a 0-1 anomaly score.

    decision_function > 0 means "inlier" and < 0 means "outlier" (the 0 point
    is the contamination threshold). A sigmoid centred on 0 gives ~0.5 at the
    threshold, <0.35 for clearly normal points and >0.70 for clear outliers.
    """
    raw = anomaly.decision_function(X)
    return 1.0 / (1.0 + np.exp(steepness * raw))

def train_models(train_df, test_df, model_dir="models/nslkdd"):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, _ = fit_transform(train_df, model_dir)
    X_test = transform(test_df, model_dir)

    y_train = train_df["attack_family"].astype(str)
    y_test = test_df["attack_family"].astype(str)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_features="sqrt",
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = _metrics(y_test, rf_pred)
    joblib.dump(rf, model_dir / "random_forest.joblib")

    # Linear SVM: considerably more practical for the one-hot NSL-KDD matrix
    # than an RBF SVM while still being a genuine support-vector classifier.
    svm = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        max_iter=5000,
    )
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_metrics = _metrics(y_test, svm_pred)
    joblib.dump(svm, model_dir / "linear_svm.joblib")

    # Isolation Forest is trained only on Normal training records.
    normal_mask = y_train == "Normal"
    anomaly = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    anomaly.fit(X_train[normal_mask])
    joblib.dump(anomaly, model_dir / "isolation_forest.joblib")

    metrics = {
        "dataset": "NSL-KDD",
        "random_forest": rf_metrics,
        "linear_svm": svm_metrics,
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def load_models(model_dir="models/nslkdd"):
    model_dir = Path(model_dir)
    return (
        joblib.load(model_dir / "random_forest.joblib"),
        joblib.load(model_dir / "linear_svm.joblib"),
        joblib.load(model_dir / "isolation_forest.joblib"),
    )

def predict(df, model_dir="models/nslkdd"):
    rf, svm, anomaly = load_models(model_dir)
    X = transform(df, model_dir)

    rf_pred = rf.predict(X)
    rf_prob = rf.predict_proba(X).max(axis=1)

    svm_pred = svm.predict(X)
    decision = svm.decision_function(X)
    svm_conf = 1.0 / (1.0 + np.exp(-np.max(decision, axis=1)))

    anomaly_score = anomaly_score_from_model(anomaly, X)

    result = df.copy()
    result["rf_prediction"] = rf_pred
    result["rf_confidence"] = rf_prob
    result["svm_prediction"] = svm_pred
    result["svm_confidence"] = svm_conf
    result["anomaly_score"] = anomaly_score
    return result
