from pathlib import Path
import numpy as np
import joblib

from .nsl_preprocessing import transform
from .nsl_models import LABELS

DEFAULT_WEIGHTS = {
    "random_forest": 0.45,
    "linear_svm": 0.25,
    "pytorch_dnn": 0.30,
}

def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def _svm_probabilities(svm, X):
    decision = svm.decision_function(X)
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    return _softmax(decision)

def _align_to_labels(probs, classes):
    """Reorder probability columns so they follow LABELS order.

    scikit-learn sorts classes_ alphabetically (DoS, Normal, Probe, R2L, U2R)
    whereas LABELS is (Normal, DoS, Probe, R2L, U2R). Without this alignment
    argmax indices map to the wrong class names.
    """
    classes = [str(c) for c in classes]
    out = np.zeros((probs.shape[0], len(LABELS)), dtype=float)
    for j, name in enumerate(LABELS):
        if name in classes:
            out[:, j] = probs[:, classes.index(name)]
    return out

def predict_ensemble(df, model_dir="models/nslkdd", weights=None):
    model_dir = Path(model_dir)
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

    rf = joblib.load(model_dir / "random_forest.joblib")
    svm = joblib.load(model_dir / "linear_svm.joblib")
    X = transform(df, model_dir)

    rf_probs = _align_to_labels(rf.predict_proba(X), rf.classes_)
    svm_probs = _align_to_labels(_svm_probabilities(svm, X), svm.classes_)

    components = {
        "random_forest": rf_probs,
        "linear_svm": svm_probs,
    }
    active_weights = {
        "random_forest": weights["random_forest"],
        "linear_svm": weights["linear_svm"],
    }

    dnn_path = model_dir / "pytorch_dnn.pt"
    if dnn_path.exists():
        try:
            from .dnn_models import predict_proba
            dnn_probs, _ = predict_proba(X, dnn_path)
            components["pytorch_dnn"] = dnn_probs
            active_weights["pytorch_dnn"] = weights["pytorch_dnn"]
        except Exception:
            pass

    total = sum(active_weights.values())
    ensemble_probs = sum(
        (active_weights[name] / total) * components[name]
        for name in components
    )

    pred_ids = ensemble_probs.argmax(axis=1)
    confidence = ensemble_probs.max(axis=1)

    result = df.copy()
    result["ensemble_prediction"] = [LABELS[i] for i in pred_ids]
    result["ensemble_confidence"] = confidence

    for name, probs in components.items():
        key = name.replace("pytorch_dnn", "dnn").replace("random_forest", "rf")
        result[f"{key}_prediction"] = [
            LABELS[i] for i in probs.argmax(axis=1)
        ]
        result[f"{key}_confidence"] = probs.max(axis=1)

    return result, components, active_weights
