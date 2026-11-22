#!/usr/bin/env python3
"""
Cyber Threat and Error Detection Using Machine Learning Algorithms
===================================================================
B.Tech Minor Project - complete single-file implementation.

Pipeline
--------
NSL-KDD -> preprocessing -> Random Forest + Linear SVM + PyTorch DNN
        -> weighted ensemble -> Isolation Forest anomaly score
        -> hybrid risk score -> Q-learning response recommendation
        -> human-in-the-loop feedback (updates the Q-table)

Usage
-----
    pip install numpy pandas scikit-learn joblib streamlit
    pip install torch            # optional: enables the PyTorch DNN

    python cyber_threat_detection.py run             # train + evaluate + tune + report
    python cyber_threat_detection.py run --skip-dnn --skip-tuning   # fast mode
    python cyber_threat_detection.py demo            # console demo of the full pipeline
    streamlit run cyber_threat_detection.py          # interactive dashboard

Other commands: download | train | qlearning | dnn | evaluate | tune | report

SAFETY: the response layer only *recommends* MONITOR / ALERT / BLOCK / ISOLATE.
It never touches a firewall, a device or the network.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import LinearSVC

try:  # PyTorch is optional
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False

# --------------------------------------------------------------------------
# Paths (everything lives next to this file)
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "nsl_kdd"
MODEL_DIR = BASE / "models" / "nslkdd"
RESULTS_DIR = BASE / "results"
STEP7_DIR = RESULTS_DIR / "step7"
FEEDBACK_PATH = BASE / "data" / "feedback.csv"
REPORT_PATH = BASE / "PROJECT_REPORT.md"

LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# ==========================================================================
# 1. DATASET  (NSL-KDD)
# ==========================================================================
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURE_NAMES if c not in CATEGORICAL]

DOS = {"back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
       "udpstorm", "processtable", "mailbomb"}
PROBE = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L = {"ftp_write", "guess_passwd", "imap", "multihop", "phf", "warezmaster",
       "warezclient", "spy", "named", "sendmail", "xlock", "xsnoop",
       "snmpgetattack", "snmpguess", "worm", "httptunnel"}
U2R = {"buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack",
       "xterm"}

DATASET_URLS = {
    "KDDTrain+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "https://raw.githubusercontent.com/4mirhosein/NSL-KDD/master/NSL_KDD_Dataset/KDDTrain%2B.txt",
    ],
    "KDDTest+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "https://raw.githubusercontent.com/4mirhosein/NSL-KDD/master/NSL_KDD_Dataset/KDDTest%2B.txt",
    ],
}


def download_dataset(data_dir: Path = DATA_DIR):
    """Download KDDTrain+/KDDTest+ (tries several mirrors) if missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, urls in DATASET_URLS.items():
        target = data_dir / name
        if target.exists() and target.stat().st_size > 10_000:
            continue
        last = None
        for url in urls:
            try:
                print(f"[DOWNLOAD] {name} <- {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    target.write_bytes(r.read())
                if target.stat().st_size > 10_000:
                    break
            except Exception as exc:  # try next mirror
                last = exc
                print(f"[WARN] {exc}")
        else:
            raise RuntimeError(
                f"Could not download {name}. Put it manually in {data_dir} and rerun."
            ) from last
    return data_dir / "KDDTrain+.txt", data_dir / "KDDTest+.txt"


def load_nsl_file(path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=FEATURE_NAMES + ["label", "difficulty"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    return df


def map_attack_family(label: str) -> str:
    if label == "normal":
        return "Normal"
    if label in DOS:
        return "DoS"
    if label in PROBE:
        return "Probe"
    if label in R2L:
        return "R2L"
    if label in U2R:
        return "U2R"
    return "Other"  # never silently treat unknown attacks as normal


def add_project_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["attack_family"] = out["label"].map(map_attack_family)
    return out


def load_data():
    train_path, test_path = download_dataset()
    return add_project_label(load_nsl_file(train_path)), add_project_label(load_nsl_file(test_path))


# ==========================================================================
# 2. PREPROCESSING
# ==========================================================================
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])


def fit_transform(train_df, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    pre = build_preprocessor()
    X = pre.fit_transform(train_df[FEATURE_NAMES])
    joblib.dump(pre, model_dir / "preprocessor.joblib")
    return X, pre


def transform(df, model_dir=MODEL_DIR):
    pre = joblib.load(Path(model_dir) / "preprocessor.joblib")
    return pre.transform(df[FEATURE_NAMES])


# ==========================================================================
# 3. SUPERVISED MODELS + ISOLATION FOREST
# ==========================================================================
def _metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    _, _, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "f1_macro": float(f_macro),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
    }


def anomaly_score_from_model(anomaly, X, steepness: float = 12.0):
    """Map IsolationForest.decision_function to a calibrated 0-1 score.

    decision_function > 0 -> inlier, < 0 -> outlier (0 = contamination
    threshold). A sigmoid centred on 0 gives ~0.5 at the threshold,
    <0.35 for clearly normal and >0.70 for clear outliers.
    """
    return 1.0 / (1.0 + np.exp(steepness * anomaly.decision_function(X)))


def train_models(train_df, test_df, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    X_train, _ = fit_transform(train_df, model_dir)
    X_test = transform(test_df, model_dir)
    y_train = train_df["attack_family"].astype(str)
    y_test = test_df["attack_family"].astype(str)

    rf = RandomForestClassifier(n_estimators=150, random_state=42,
                                class_weight="balanced_subsample",
                                n_jobs=-1, max_features="sqrt")
    rf.fit(X_train, y_train)
    rf_metrics = _metrics(y_test, rf.predict(X_test))
    joblib.dump(rf, model_dir / "random_forest.joblib")

    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=5000)
    svm.fit(X_train, y_train)
    svm_metrics = _metrics(y_test, svm.predict(X_test))
    joblib.dump(svm, model_dir / "linear_svm.joblib")

    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train[(y_train == "Normal").to_numpy(dtype=bool)])  # trained on Normal traffic only
    joblib.dump(iso, model_dir / "isolation_forest.joblib")

    metrics = {"dataset": "NSL-KDD", "random_forest": rf_metrics, "linear_svm": svm_metrics}
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_models(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    return (joblib.load(model_dir / "random_forest.joblib"),
            joblib.load(model_dir / "linear_svm.joblib"),
            joblib.load(model_dir / "isolation_forest.joblib"))


def predict(df, model_dir=MODEL_DIR):
    """RF + SVM predictions and the Isolation-Forest anomaly score."""
    rf, svm, iso = load_models(model_dir)
    X = transform(df, model_dir)
    decision = svm.decision_function(X)
    result = df.copy()
    result["rf_prediction"] = rf.predict(X)
    result["rf_confidence"] = rf.predict_proba(X).max(axis=1)
    result["svm_prediction"] = svm.predict(X)
    result["svm_confidence"] = 1.0 / (1.0 + np.exp(-np.max(decision, axis=1)))
    result["anomaly_score"] = anomaly_score_from_model(iso, X)
    return result


# ==========================================================================
# 4. PYTORCH DNN (optional)
# ==========================================================================
DNN_PATH = MODEL_DIR / "pytorch_dnn.pt"

if HAS_TORCH:
    class IDSNet(nn.Module):
        def __init__(self, input_dim, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.20),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.net(x)


def dnn_predict_proba(X, model_path=DNN_PATH):
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    model = IDSNet(payload["input_dim"], len(payload["labels"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1).numpy()


def train_dnn(train_df, test_df, model_dir=MODEL_DIR, epochs=10):
    if not HAS_TORCH:
        raise RuntimeError("PyTorch not installed. Run: pip install torch")
    model_dir = Path(model_dir)
    torch.manual_seed(42)
    np.random.seed(42)
    X_train, _ = fit_transform(train_df, model_dir)
    X_test = transform(test_df, model_dir)
    idx = {l: i for i, l in enumerate(LABELS)}
    y_train = np.array([idx[x] for x in train_df["attack_family"]])
    y_test = np.array([idx[x] for x in test_df["attack_family"]])

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=512, shuffle=True)
    model = IDSNet(X_train.shape[1], len(LABELS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {ep + 1:02d}/{epochs} - loss={total / len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        pred = torch.softmax(model(torch.tensor(X_test, dtype=torch.float32)), dim=1).numpy().argmax(1)
    metrics = _metrics([LABELS[i] for i in y_test], [LABELS[i] for i in pred])
    print(f"PyTorch DNN accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")

    torch.save({"state_dict": model.state_dict(), "input_dim": X_train.shape[1],
                "labels": LABELS}, model_dir / "pytorch_dnn.pt")
    mp = model_dir / "metrics.json"
    payload = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"dataset": "NSL-KDD"}
    payload["pytorch_dnn"] = metrics
    mp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return metrics


# ==========================================================================
# 5. WEIGHTED ENSEMBLE  (RF + SVM + DNN)
# ==========================================================================
DEFAULT_WEIGHTS = {"random_forest": 0.45, "linear_svm": 0.25, "pytorch_dnn": 0.30}


def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _align_to_labels(probs, classes):
    """Reorder probability columns to LABELS order.

    scikit-learn sorts classes_ alphabetically (DoS, Normal, Probe, ...), but
    LABELS is (Normal, DoS, Probe, ...). Without this step the argmax index
    maps to the wrong class name.
    """
    classes = [str(c) for c in classes]
    out = np.zeros((probs.shape[0], len(LABELS)))
    for j, name in enumerate(LABELS):
        if name in classes:
            out[:, j] = probs[:, classes.index(name)]
    return out


def predict_ensemble(df, model_dir=MODEL_DIR, weights=None):
    model_dir = Path(model_dir)
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    rf = joblib.load(model_dir / "random_forest.joblib")
    svm = joblib.load(model_dir / "linear_svm.joblib")
    X = transform(df, model_dir)

    dec = svm.decision_function(X)
    if dec.ndim == 1:
        dec = np.column_stack([-dec, dec])
    components = {
        "random_forest": _align_to_labels(rf.predict_proba(X), rf.classes_),
        "linear_svm": _align_to_labels(_softmax(dec), svm.classes_),
    }
    active = {"random_forest": weights["random_forest"], "linear_svm": weights["linear_svm"]}

    dnn_path = model_dir / "pytorch_dnn.pt"
    if HAS_TORCH and dnn_path.exists():
        try:
            components["pytorch_dnn"] = dnn_predict_proba(X, dnn_path)
            active["pytorch_dnn"] = weights["pytorch_dnn"]
        except Exception:
            pass

    total = sum(active.values())
    ens = sum((active[n] / total) * components[n] for n in components)

    result = df.copy()
    result["ensemble_prediction"] = [LABELS[i] for i in ens.argmax(1)]
    result["ensemble_confidence"] = ens.max(1)
    for name, probs in components.items():
        key = name.replace("pytorch_dnn", "dnn").replace("random_forest", "rf")
        result[f"{key}_prediction"] = [LABELS[i] for i in probs.argmax(1)]
        result[f"{key}_confidence"] = probs.max(1)
    return result, components, active


# ==========================================================================
# 6. EVALUATION (metrics, confusion matrix, one-vs-rest ROC / AUC)
# ==========================================================================
def classification_metrics(y_true, pred) -> dict:
    m = _metrics(y_true, pred)
    m.pop("classification_report")
    return m


def roc_payload(y_true, probabilities) -> dict:
    y_bin = label_binarize(y_true, classes=LABELS)
    out, aucs = {}, []
    for i, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
        auc = roc_auc_score(y_bin[:, i], probabilities[:, i])
        out[label] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}
        aucs.append(auc)
    out["macro_auc"] = float(np.mean(aucs))
    return out


def evaluate_all(model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    _, test_path = download_dataset()
    test_df = add_project_label(load_nsl_file(test_path))
    y_true = test_df["attack_family"].astype(str).tolist()
    ens_df, components, weights = predict_ensemble(test_df, model_dir)

    payload = {"dataset": "NSL-KDD", "active_ensemble_weights": weights, "models": {}}
    pred_col = {"random_forest": "rf_prediction", "linear_svm": "linear_svm_prediction",
                "pytorch_dnn": "dnn_prediction"}
    for key, probs in components.items():
        m = classification_metrics(y_true, ens_df[pred_col[key]].tolist())
        m["roc"] = roc_payload(y_true, probs)
        payload["models"][key] = m
    total = sum(weights.values())
    ens_probs = sum((weights[n] / total) * p for n, p in components.items())
    m = classification_metrics(y_true, ens_df["ensemble_prediction"].tolist())
    m["roc"] = roc_payload(y_true, ens_probs)
    payload["models"]["ensemble"] = m

    for name, r in payload["models"].items():
        print(f"{name:14s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
              f"macroF1={r['f1_macro']:.4f} macroAUC={r['roc']['macro_auc']:.4f}")
    (model_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ==========================================================================
# 7. RISK ENGINE
# ==========================================================================
LABEL_SEVERITY = {"Normal": 0.00, "Probe": 0.45, "DoS": 0.70,
                  "R2L": 0.85, "U2R": 1.00, "Other": 0.60}


@dataclass
class RiskResult:
    threat_family: str
    supervised_confidence: float
    anomaly_score: float
    risk_score: float
    risk_level: str


def risk_level(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MEDIUM"
    if score < 0.75:
        return "HIGH"
    return "CRITICAL"


def calculate_risk(threat_family, supervised_confidence, anomaly_score) -> RiskResult:
    """risk = 0.65 * severity * confidence + 0.35 * anomaly  (Normal: 0.35 * anomaly)."""
    threat_family = str(threat_family)
    conf = float(np.clip(supervised_confidence, 0.0, 1.0))
    anom = float(np.clip(anomaly_score, 0.0, 1.0))
    severity = LABEL_SEVERITY.get(threat_family, LABEL_SEVERITY["Other"])
    score = 0.65 * severity * conf + 0.35 * anom
    if threat_family == "Normal":
        score = 0.35 * anom
    score = float(np.clip(score, 0.0, 1.0))
    return RiskResult(threat_family, conf, anom, score, risk_level(score))


# ==========================================================================
# 8. Q-LEARNING RESPONSE ENGINE
# ==========================================================================
ACTIONS = ["MONITOR", "ALERT", "BLOCK", "ISOLATE"]
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
THREAT_FAMILIES = ["Normal", "Probe", "DoS", "R2L", "U2R", "Other"]
ANOMALY_LEVELS = ["LOW", "MEDIUM", "HIGH"]
IDEAL_ACTION = {"LOW": "MONITOR", "MEDIUM": "ALERT", "HIGH": "BLOCK", "CRITICAL": "ISOLATE"}
ACTION_PENALTY = {"MONITOR": 0.0, "ALERT": 0.8, "BLOCK": 1.5, "ISOLATE": 2.0}


def anomaly_level(score) -> str:
    score = float(score)
    return "LOW" if score < 0.35 else "MEDIUM" if score < 0.70 else "HIGH"


def state_key(risk_lvl, threat_family, anomaly_score):
    return (str(risk_lvl), str(threat_family), anomaly_level(anomaly_score))


def reward_for(risk_lvl, action, human_feedback=None) -> float:
    ideal = IDEAL_ACTION.get(risk_lvl, "ALERT")
    if action == ideal:
        reward = 5.0
    else:
        d = ACTIONS.index(action) - ACTIONS.index(ideal)
        reward = (-3.0 - abs(d) * 1.5) if d < 0 else (1.0 - abs(d) * 0.8)
    reward -= ACTION_PENALTY[action]
    if human_feedback == "accept":
        reward += 2.0
    elif human_feedback == "reject":
        reward -= 2.0
    return float(reward)


class QLearningResponseEngine:
    def __init__(self, alpha=0.10, gamma=0.90, epsilon=0.20, seed=42,
                 model_path=MODEL_DIR / "q_table.json"):
        self.alpha, self.gamma, self.epsilon = float(alpha), float(gamma), float(epsilon)
        self.rng = random.Random(seed)
        self.model_path = Path(model_path)
        self.q = {}
        self._load()

    @staticmethod
    def _key(state):
        return "|".join(map(str, state))

    def _ensure_state(self, state):
        key = self._key(state)
        if key not in self.q:
            self.q[key] = {a: 0.0 for a in ACTIONS}
        return self.q[key]

    def choose_action(self, state, explore=False):
        values = self._ensure_state(state)
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(ACTIONS)
        best = max(values.values())
        return self.rng.choice([a for a, v in values.items() if v == best])

    def update(self, state, action, reward, next_state):
        cur, fut = self._ensure_state(state), self._ensure_state(next_state)
        target = float(reward) + self.gamma * max(fut.values())
        cur[action] += self.alpha * (target - cur[action])

    def train_simulated(self, episodes=50000):
        states = [(r, t, a) for r in RISK_LEVELS for t in THREAT_FAMILIES for a in ANOMALY_LEVELS]
        for _ in range(int(episodes)):
            state = self.rng.choice(states)
            i = RISK_LEVELS.index(state[0])
            j = max(0, min(len(RISK_LEVELS) - 1, i + self.rng.choice([-1, 0, 0, 0, 1])))
            nxt = (RISK_LEVELS[j], self.rng.choice(THREAT_FAMILIES), self.rng.choice(ANOMALY_LEVELS))
            action = self.choose_action(state, explore=True)
            self.update(state, action, reward_for(state[0], action), nxt)
        self.save()
        return self

    def policy_agreement(self):
        states = [(r, t, a) for r in RISK_LEVELS for t in THREAT_FAMILIES for a in ANOMALY_LEVELS]
        ok = sum(self.choose_action(s) == IDEAL_ACTION[s[0]] for s in states)
        return ok, len(states)

    def learn_from_feedback(self, state, action, accepted):
        reward = reward_for(state[0], action, "accept" if accepted else "reject")
        self.update(state, action, reward, state)
        self.save()
        return reward

    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.write_text(json.dumps({
            "alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon,
            "actions": ACTIONS, "q_table": self.q}, indent=2), encoding="utf-8")

    def _load(self):
        if self.model_path.exists():
            try:
                self.q = json.loads(self.model_path.read_text(encoding="utf-8")).get("q_table", {})
            except (json.JSONDecodeError, OSError):
                self.q = {}


def train_qlearning(episodes=50000):
    engine = QLearningResponseEngine()
    engine.q = {}  # always learn from a clean table
    engine.train_simulated(episodes)
    ok, total = engine.policy_agreement()
    print(f"Q-table saved. Policy matches intended risk->action mapping in {ok}/{total} states")
    for r in RISK_LEVELS:
        print(f"  {r:8s} -> {engine.choose_action(state_key(r, 'DoS', 0.8))}")
    return engine


# ==========================================================================
# 9. HUMAN-IN-THE-LOOP FEEDBACK LOG
# ==========================================================================
FEEDBACK_COLUMNS = ["timestamp", "risk_level", "threat_family", "risk_score",
                    "anomaly_score", "recommended_action", "accepted", "comment"]


def append_feedback(path, risk_lvl, threat_family, risk_score, anomaly_score,
                    recommended_action, accepted, comment=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow({"timestamp": datetime.now(timezone.utc).isoformat(),
                    "risk_level": risk_lvl, "threat_family": threat_family,
                    "risk_score": float(risk_score), "anomaly_score": float(anomaly_score),
                    "recommended_action": recommended_action,
                    "accepted": bool(accepted), "comment": comment})


def load_feedback(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=FEEDBACK_COLUMNS)


# ==========================================================================
# 10. END-TO-END ANALYSIS (ensemble -> anomaly -> risk -> Q-learning)
# ==========================================================================
def analyze(df, model_dir=MODEL_DIR, engine=None):
    """Full pipeline for a DataFrame of NSL-KDD records."""
    engine = engine or QLearningResponseEngine(model_path=Path(model_dir) / "q_table.json")
    base = predict(df, model_dir)
    ens, _, weights = predict_ensemble(df, model_dir)
    rows = []
    for idx, row in ens.iterrows():
        risk = calculate_risk(row["ensemble_prediction"], row["ensemble_confidence"],
                              float(base.loc[idx, "anomaly_score"]))
        state = state_key(risk.risk_level, risk.threat_family, risk.anomaly_score)
        rows.append({
            "Ensemble Prediction": risk.threat_family,
            "Ensemble Confidence": risk.supervised_confidence,
            "RF Prediction": row.get("rf_prediction", ""),
            "RF Confidence": row.get("rf_confidence", 0.0),
            "SVM Prediction": row.get("linear_svm_prediction", ""),
            "SVM Confidence": row.get("linear_svm_confidence", 0.0),
            "DNN Prediction": row.get("dnn_prediction", "Unavailable"),
            "DNN Confidence": row.get("dnn_confidence", 0.0),
            "Anomaly Score": risk.anomaly_score,
            "Risk Score": risk.risk_score,
            "Risk Level": risk.risk_level,
            "Recommended Response": engine.choose_action(state),
        })
    return pd.DataFrame(rows, index=ens.index), weights


def demo(n=12):
    """Console demonstration of the full pipeline on sampled KDDTest+ records."""
    _, test_path = download_dataset()
    df = add_project_label(load_nsl_file(test_path)).sample(n, random_state=7)
    out, weights = analyze(df)
    out.insert(0, "Actual", df["attack_family"].values)
    print("Ensemble weights:", {k: round(v, 2) for k, v in weights.items()})
    cols = ["Actual", "Ensemble Prediction", "Ensemble Confidence", "Anomaly Score",
            "Risk Score", "Risk Level", "Recommended Response"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out[cols].round(2).to_string(index=False))
    print("\n(Responses are simulated recommendations; nothing is executed on a network.)")


# ==========================================================================
# 11. STEP 7 - class imbalance, bounded RF tuning, permutation importance
# ==========================================================================
def tune_analysis():
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import RandomizedSearchCV

    STEP7_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_data()

    rows = []
    for split, d in [("train", train), ("test", test)]:
        c = d["attack_family"].value_counts().reindex(LABELS, fill_value=0)
        rows += [{"split": split, "class": k, "count": int(v),
                  "percentage": 100.0 * v / c.sum()} for k, v in c.items()]
    pd.DataFrame(rows).to_csv(STEP7_DIR / "class_distribution.csv", index=False)

    pre = build_preprocessor()
    Xtr, Xte = pre.fit_transform(train[FEATURE_NAMES]), pre.transform(test[FEATURE_NAMES])
    ytr = train["attack_family"].to_numpy(dtype=object)
    yte = test["attack_family"].to_numpy(dtype=object)

    def summarise(model, tag):
        pred = model.predict(Xte)
        rep = classification_report(yte, pred, labels=LABELS, target_names=LABELS,
                                    output_dict=True, zero_division=0)
        return {f"{tag}_accuracy": float(rep["accuracy"]),
                f"{tag}_f1_macro": float(rep["macro avg"]["f1-score"]),
                f"{tag}_f1_weighted": float(rep["weighted avg"]["f1-score"]),
                f"{tag}_per_class_f1": {c: float(rep[c]["f1-score"]) for c in LABELS},
                f"{tag}_confusion_matrix": confusion_matrix(yte, pred, labels=LABELS).tolist()}

    baseline = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1,
                                      class_weight="balanced_subsample",
                                      max_features="sqrt").fit(Xtr, ytr)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        {"n_estimators": [100, 150, 250], "max_depth": [None, 12, 20, 30],
         "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", "log2"]},
        n_iter=8, scoring="f1_macro", cv=3, random_state=42, n_jobs=1, verbose=1)
    t0 = time.time()
    search.fit(Xtr, ytr)

    res = {"best_params": search.best_params_, "best_cv_f1_macro": float(search.best_score_),
           "training_seconds": float(time.time() - t0), "labels": LABELS}
    res.update(summarise(baseline, "baseline"))
    res.update(summarise(search.best_estimator_, "tuned"))
    res["tuned_improves_macro_f1"] = bool(res["tuned_f1_macro"] > res["baseline_f1_macro"])
    (STEP7_DIR / "tuned_random_forest.json").write_text(json.dumps(res, indent=2))

    rng = np.random.default_rng(42)
    sel = rng.choice(Xte.shape[0], size=min(3000, Xte.shape[0]), replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi = permutation_importance(search.best_estimator_, Xte[sel], yte[sel], n_repeats=3,
                                    random_state=42, n_jobs=-1, scoring="f1_macro")
    pd.DataFrame({"feature": pre.get_feature_names_out(),
                  "importance_mean": pi.importances_mean,
                  "importance_std": pi.importances_std}
                 ).sort_values("importance_mean", ascending=False
                 ).to_csv(STEP7_DIR / "permutation_importance.csv", index=False)

    print("Best params :", search.best_params_)
    print(f"Baseline RF : acc={res['baseline_accuracy']:.4f} macroF1={res['baseline_f1_macro']:.4f}")
    print(f"Tuned RF    : acc={res['tuned_accuracy']:.4f} macroF1={res['tuned_f1_macro']:.4f}")
    print("Tuned improves macro-F1:", res["tuned_improves_macro_f1"])
    return res


# ==========================================================================
# 12. EXPORT + REPORT
# ==========================================================================
def export_results():
    RESULTS_DIR.mkdir(exist_ok=True)
    for fn in ["metrics.json", "evaluation.json"]:
        p = MODEL_DIR / fn
        if p.exists():
            (RESULTS_DIR / fn).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    p = MODEL_DIR / "evaluation.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = [{"model": n, "accuracy": round(v["accuracy"], 4),
                 "precision_weighted": round(v["precision"], 4),
                 "recall_weighted": round(v["recall"], 4),
                 "f1_weighted": round(v["f1"], 4),
                 "f1_macro": round(v.get("f1_macro", float("nan")), 4),
                 "macro_auc": round(v["roc"]["macro_auc"], 4)}
                for n, v in d["models"].items()]
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        print("Wrote results/model_comparison.csv")


def generate_report():
    pct = lambda x: f"{x * 100:.2f}%"
    ev_path = MODEL_DIR / "evaluation.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else {}
    L = ["# Cyber Threat and Error Detection Using Machine Learning Algorithms", "",
         f"**Implementation report - generated {date.today().isoformat()}**", "",
         "## 1. Overview", "",
         "Hybrid framework: supervised detection (Random Forest, Linear SVM, PyTorch DNN), "
         "unsupervised anomaly detection (Isolation Forest), a hybrid risk score, "
         "Q-learning response recommendation and human-in-the-loop feedback.", "",
         "## 2. Architecture", "", "```text",
         "NSL-KDD -> preprocessing -> RF + SVM + DNN -> weighted ensemble",
         "        -> Isolation Forest anomaly score -> risk score",
         "        -> Q-learning response -> analyst feedback -> Q-table update", "```", "",
         "## 3. Dataset", "",
         "NSL-KDD (KDDTrain+ 125,973 records, KDDTest+ 22,544 records), mapped to five "
         "classes: Normal, DoS, Probe, R2L, U2R. Numeric features are standardised and "
         "protocol/service/flag are one-hot encoded.", "",
         "## 4. Risk scoring and response", "",
         "risk = 0.65 x severity x confidence + 0.35 x anomaly (Normal: 0.35 x anomaly).", "",
         "| Risk | Score | Typical action |", "|---|---:|---|",
         "| LOW | < 0.25 | MONITOR |", "| MEDIUM | 0.25-0.50 | ALERT |",
         "| HIGH | 0.50-0.75 | BLOCK |", "| CRITICAL | >= 0.75 | ISOLATE |", "",
         "## 5. Evaluation (KDDTest+)", ""]
    if ev:
        L += ["| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) | Macro AUC |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for n, r in ev["models"].items():
            L.append(f"| {n.replace('_', ' ').title()} | {pct(r['accuracy'])} | {pct(r['precision'])} | "
                     f"{pct(r['recall'])} | {pct(r['f1'])} | {pct(r.get('f1_macro', float('nan')))} | "
                     f"{r['roc']['macro_auc']:.3f} |")
    else:
        L.append("_Run `python cyber_threat_detection.py evaluate` to populate._")
    tp = STEP7_DIR / "tuned_random_forest.json"
    if tp.exists():
        t = json.loads(tp.read_text(encoding="utf-8"))
        L += ["", "### 5.1 Random Forest tuning", "", "| Model | Accuracy | Macro-F1 |", "|---|---:|---:|",
              f"| Baseline RF | {pct(t['baseline_accuracy'])} | {pct(t['baseline_f1_macro'])} |",
              f"| Tuned RF | {pct(t['tuned_accuracy'])} | {pct(t['tuned_f1_macro'])} |", "",
              f"Best parameters: `{t['best_params']}`; CV macro-F1 (train) = {t['best_cv_f1_macro']:.4f}. "
              f"Tuned model improved test macro-F1: **{'yes' if t['tuned_improves_macro_f1'] else 'no'}**.", "",
              "Per-class F1 (tuned): " + ", ".join(f"{k} {v:.3f}" for k, v in t["tuned_per_class_f1"].items()) + "."]
    ip = STEP7_DIR / "permutation_importance.csv"
    if ip.exists():
        L += ["", "Top-10 permutation importances:", "", "| Feature | Macro-F1 drop |", "|---|---:|"]
        L += [f"| {r.feature} | {r.importance_mean:.4f} |" for r in pd.read_csv(ip).head(10).itertuples()]
    L += ["", "## 6. Limitations", "",
          "- NSL-KDD is an academic benchmark, not modern enterprise traffic.",
          "- U2R (52 training records) and R2L (995) are heavily under-represented; minority-class recall is weak.",
          "- KDDTest+ is distributed differently from KDDTrain+, so test accuracy is far below CV accuracy.",
          "- The risk formula and Q-learning rewards are project-defined; Q-learning is trained in a simulation.",
          "- Responses are recommendations only; no real firewall/isolation action is executed.", ""]
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")
    print("Generated", REPORT_PATH)


# ==========================================================================
# 13. COMMAND LINE
# ==========================================================================
def run_all(skip_dnn=False, skip_tuning=False):
    t0 = time.time()
    print("\n[1/7] Dataset");            train, test = load_data()
    print(f"      train={len(train):,}  test={len(test):,}")
    print(train["attack_family"].value_counts().to_string())
    print("\n[2/7] RF + SVM + Isolation Forest")
    for n, m in train_models(train, test).items():
        if isinstance(m, dict):
            print(f"      {n}: acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    print("\n[3/7] Q-learning");         train_qlearning()
    if skip_dnn or not HAS_TORCH:
        print("\n[4/7] PyTorch DNN skipped" + ("" if skip_dnn else " (torch not installed)"))
        (MODEL_DIR / "pytorch_dnn.pt").unlink(missing_ok=True)
    else:
        print("\n[4/7] PyTorch DNN");    train_dnn(train, test)
    print("\n[5/7] Evaluation + ensemble"); evaluate_all()
    if skip_tuning:
        print("\n[6/7] RF tuning skipped")
    else:
        print("\n[6/7] RF tuning + feature importance"); tune_analysis()
    print("\n[7/7] Export + report");    export_results(); generate_report()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Dashboard: streamlit run {Path(__file__).name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cyber threat detection - single-file project")
    ap.add_argument("command", choices=["run", "download", "train", "qlearning", "dnn",
                                        "evaluate", "tune", "report", "demo"])
    ap.add_argument("--skip-dnn", action="store_true")
    ap.add_argument("--skip-tuning", action="store_true")
    a = ap.parse_args(argv)
    c = a.command
    if c == "run":
        run_all(a.skip_dnn, a.skip_tuning)
    elif c == "download":
        print(download_dataset())
    elif c == "train":
        tr, te = load_data(); train_models(tr, te)
    elif c == "qlearning":
        train_qlearning()
    elif c == "dnn":
        tr, te = load_data(); train_dnn(tr, te)
    elif c == "evaluate":
        evaluate_all(); export_results()
    elif c == "tune":
        tune_analysis()
    elif c == "report":
        export_results(); generate_report()
    elif c == "demo":
        demo()


# ==========================================================================
# 14. STREAMLIT DASHBOARD   (streamlit run cyber_threat_detection.py)
# ==========================================================================
def dashboard():
    import streamlit as st
    # set_page_config must be the first Streamlit call and may only be used once.
    st.set_page_config(
        page_title="Cyber Threat Detection & Response",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 12px;
    }
    div[data-testid="stMetricValue"] {font-size: 1.55rem;}
    section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
    </style>
    """, unsafe_allow_html=True)

    METRICS_PATH = MODEL_DIR / "metrics.json"
    EVAL_PATH = MODEL_DIR / "evaluation.json"
    TEST_PATH = DATA_DIR / "KDDTest+.txt"
    Q_PATH = MODEL_DIR / "q_table.json"

    st.title("🛡️ Cyber Threat and Error Detection")
    st.caption(
        "Hybrid ML + anomaly detection + adaptive Q-learning + "
        "human-in-the-loop"
    )

    if not METRICS_PATH.exists():
        st.error("Models are missing. Run `python cyber_threat_detection.py run` first.")
        st.stop()

    if not Q_PATH.exists():
        st.warning(
            "Q-learning model is missing. Run `python cyber_threat_detection.py qlearning`."
        )
        st.stop()

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    evaluation = (
        json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        if EVAL_PATH.exists()
        else None
    )
    q_engine = QLearningResponseEngine(model_path=Q_PATH)

    pages = [
        "Dashboard",
        "Model Comparison",
        "ROC Curves",
        "Detection & Response",
        "Anomaly Detection",
        "Human Feedback",
        "Explainability & Tuning",
        "Project Notes",
    ]
    page = st.sidebar.radio("Navigation", pages)

    st.sidebar.divider()
    st.sidebar.success("Pipeline ready")
    st.sidebar.caption(
        "Responses are simulated recommendations; no real network controls "
        "are executed."
    )

    if page == "Dashboard":
        st.header("System Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training Records", "125,973")
        c2.metric("Test Records", "22,544")
        c3.metric("Classes", "5")
        c4.metric("Input Features", "41")

        st.subheader("Architecture")
        st.code(
            "NSL-KDD\n"
            "   ↓\n"
            "Preprocessing\n"
            "   ↓\n"
            "Random Forest + Linear SVM + PyTorch DNN\n"
            "   ↓\n"
            "Isolation Forest\n"
            "   ↓\n"
            "Risk Scoring\n"
            "   ↓\n"
            "Q-Learning Response\n"
            "   ↓\n"
            "Human Feedback",
            language="text",
        )

        if evaluation:
            rows = []
            for name, result in evaluation["models"].items():
                rows.append({
                    "Model": name.replace("_", " ").title(),
                    "Accuracy": result["accuracy"],
                    "F1": result["f1"],
                    "Macro AUC": result["roc"]["macro_auc"],
                })
            st.subheader("Current evaluation")
            st.dataframe(
                pd.DataFrame(rows)
                .sort_values("F1", ascending=False)
                .style.format({
                    "Accuracy": "{:.2%}",
                    "F1": "{:.2%}",
                    "Macro AUC": "{:.3f}",
                }),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info(
                "Run `python cyber_threat_detection.py evaluate` after training to populate "
                "the comparison and ROC panels."
            )

        st.subheader("Risk and response policy")
        st.dataframe(
            pd.DataFrame({
                "Risk": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                "Score": ["< 0.25", "0.25–<0.50", "0.50–<0.75", "≥ 0.75"],
                "Q-Learning Action": ["MONITOR", "ALERT", "BLOCK", "ISOLATE"],
            }),
            width="stretch",
            hide_index=True,
        )

    elif page == "Model Comparison":
        st.header("Model Comparison")

        if evaluation:
            rows = []
            for name, result in evaluation["models"].items():
                rows.append({
                    "Model": name.replace("_", " ").title(),
                    "Accuracy": result["accuracy"],
                    "Precision": result["precision"],
                    "Recall": result["recall"],
                    "F1": result["f1"],
                    "Macro AUC": result["roc"]["macro_auc"],
                })

            df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(
                df.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1": "{:.2%}",
                    "Macro AUC": "{:.3f}",
                }),
                width="stretch",
            )

            st.subheader("F1 comparison")
            st.bar_chart(df["F1"])

            st.subheader("Accuracy comparison")
            st.bar_chart(df["Accuracy"])

            for name, result in evaluation["models"].items():
                st.subheader(name.replace("_", " ").title())
                labels = result["labels"]
                cm = pd.DataFrame(
                    result["confusion_matrix"],
                    index=labels,
                    columns=labels,
                )
                st.dataframe(cm, width="stretch")

        else:
            st.warning("Evaluation file not found. Run `python cyber_threat_detection.py evaluate`.")

    elif page == "ROC Curves":
        st.header("Multiclass ROC Curves")

        if not evaluation:
            st.warning("Run `python cyber_threat_detection.py evaluate` first.")
            st.stop()

        model_name = st.selectbox(
            "Model",
            list(evaluation["models"].keys()),
        )
        roc = evaluation["models"][model_name]["roc"]

        chart_df = pd.DataFrame({"False Positive Rate": np.linspace(0, 1, 101)})
        for label in evaluation["models"][model_name]["labels"]:
            points = pd.DataFrame({
                "fpr": roc[label]["fpr"],
                "tpr": roc[label]["tpr"],
            }).sort_values("fpr")
            y = np.interp(
                chart_df["False Positive Rate"],
                points["fpr"],
                points["tpr"],
            )
            chart_df[label] = y

        st.line_chart(chart_df.set_index("False Positive Rate"))
        st.metric("Macro AUC", f"{roc['macro_auc']:.3f}")

        st.caption(
            "Each class is evaluated one-vs-rest. Macro AUC is the mean of the "
            "per-class AUC values."
        )

    elif page == "Detection & Response":
        st.header("Detection → Anomaly → Risk → Q-Learning")

        if not TEST_PATH.exists():
            st.error("KDDTest+.txt is missing. Run `python cyber_threat_detection.py download`.")
            st.stop()

        uploaded = st.file_uploader(
            "Upload NSL-KDD-compatible CSV/TXT",
            type=["csv", "txt"],
        )

        if uploaded is None:
            input_df = add_project_label(
                load_nsl_file(TEST_PATH)
            ).sample(25, random_state=42)
            st.caption("Using 25 records sampled from KDDTest+.")
        else:
            raw = pd.read_csv(uploaded, header=None)
            if raw.shape[1] >= 43:
                raw = raw.iloc[:, :43]
                raw.columns = FEATURE_NAMES + ["label", "difficulty"]
            elif raw.shape[1] == 41:
                raw.columns = FEATURE_NAMES
            else:
                st.error("Expected 41 feature columns or 43 NSL-KDD columns.")
                st.stop()
            input_df = raw

        if st.button("Run Full Analysis", type="primary"):
            base = predict(input_df)
            ensemble_df, _, weights = predict_ensemble(input_df)

            rows = []
            for idx, row in ensemble_df.iterrows():
                risk = calculate_risk(
                    row["ensemble_prediction"],
                    row["ensemble_confidence"],
                    float(base.loc[idx, "anomaly_score"]),
                )
                state = state_key(
                    risk.risk_level,
                    risk.threat_family,
                    risk.anomaly_score,
                )
                rows.append({
                    "Ensemble Prediction": risk.threat_family,
                    "Ensemble Confidence": risk.supervised_confidence,
                    "RF Prediction": row.get("rf_prediction", ""),
                    "RF Confidence": row.get("rf_confidence", 0.0),
                    "SVM Prediction": row.get("linear_svm_prediction", ""),
                    "SVM Confidence": row.get("linear_svm_confidence", 0.0),
                    "DNN Prediction": row.get("dnn_prediction", "Unavailable"),
                    "DNN Confidence": row.get("dnn_confidence", 0.0),
                    "Anomaly Score": risk.anomaly_score,
                    "Risk Score": risk.risk_score,
                    "Risk Level": risk.risk_level,
                    "Recommended Response": q_engine.choose_action(state),
                })

            result_df = pd.DataFrame(rows)
            st.session_state["step4_results"] = result_df
            st.session_state["ensemble_weights"] = weights

        result_df = st.session_state.get("step4_results")

        if result_df is not None:
            st.subheader("Results")
            st.dataframe(
                result_df.style.format({
                    "Ensemble Confidence": "{:.1%}",
                    "RF Confidence": "{:.1%}",
                    "SVM Confidence": "{:.1%}",
                    "DNN Confidence": "{:.1%}",
                    "Anomaly Score": "{:.2f}",
                    "Risk Score": "{:.2f}",
                }),
                width="stretch",
            )

            weights = st.session_state.get("ensemble_weights", {})
            st.subheader("Ensemble weights")
            st.write({
                k.replace("_", " ").title(): round(v, 2)
                for k, v in weights.items()
            })

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "High/Critical",
                int(result_df["Risk Level"].isin(["HIGH", "CRITICAL"]).sum()),
            )
            c2.metric(
                "Anomalous",
                int((result_df["Anomaly Score"] >= 0.70).sum()),
            )
            c3.metric(
                "Rows analyzed",
                len(result_df),
            )

            st.subheader("Risk distribution")
            st.bar_chart(
                result_df["Risk Level"]
                .value_counts()
                .reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"])
                .fillna(0)
            )

            st.subheader("Recommended responses")
            st.bar_chart(result_df["Recommended Response"].value_counts())

    elif page == "Anomaly Detection":
        st.header("Isolation Forest Anomaly Detection")

        if not TEST_PATH.exists():
            st.error("KDDTest+.txt is missing.")
            st.stop()

        test_df = add_project_label(
            load_nsl_file(TEST_PATH)
        ).sample(1000, random_state=42)
        results = predict(test_df)

        c1, c2 = st.columns(2)
        c1.metric(
            "High anomaly-score records",
            int((results["anomaly_score"] >= 0.70).sum()),
        )
        c2.metric(
            "Average anomaly score",
            f"{results['anomaly_score'].mean():.3f}",
        )

        st.subheader("Anomaly score distribution")
        hist = pd.cut(
            results["anomaly_score"],
            bins=[0, .2, .4, .6, .8, 1.0],
            include_lowest=True,
        ).value_counts().sort_index()
        hist.index = hist.index.astype(str)  # interval labels are not plottable
        st.bar_chart(hist)

        st.subheader("Most anomalous records")
        st.dataframe(
            results.sort_values("anomaly_score", ascending=False)[
                ["label", "attack_family", "anomaly_score"]
            ].head(20),
            width="stretch",
        )

    elif page == "Human Feedback":
        st.header("Human-in-the-Loop Feedback")

        feedback_df = load_feedback(FEEDBACK_PATH)

        if feedback_df.empty:
            st.info("No feedback has been recorded yet.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Feedback records", len(feedback_df))
            c2.metric(
                "Acceptance rate",
                f"{feedback_df['accepted'].mean():.1%}",
            )
            st.dataframe(
                feedback_df.tail(25),
                width="stretch",
            )

        st.subheader("Submit feedback")
        risk_level = st.selectbox(
            "Risk level",
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )
        threat_family = st.selectbox(
            "Threat family",
            ["Normal", "Probe", "DoS", "R2L", "U2R", "Other"],
        )
        anomaly_score = st.slider(
            "Anomaly score",
            0.0, 1.0, 0.70, 0.01,
        )

        state = state_key(risk_level, threat_family, anomaly_score)
        recommended = q_engine.choose_action(state)
        st.info(f"Current Q-learning recommendation: **{recommended}**")

        accepted = st.radio(
            "Was the recommendation appropriate?",
            ["Accept", "Reject"],
            horizontal=True,
        )
        comment = st.text_input("Optional analyst comment")

        if st.button("Save Feedback & Update Q-table"):
            is_accepted = accepted == "Accept"
            risk_score = {
                "LOW": 0.20,
                "MEDIUM": 0.40,
                "HIGH": 0.65,
                "CRITICAL": 0.90,
            }[risk_level]

            append_feedback(
                FEEDBACK_PATH,
                risk_level,
                threat_family,
                risk_score,
                anomaly_score,
                recommended,
                is_accepted,
                comment,
            )

            reward = q_engine.learn_from_feedback(
                state,
                recommended,
                is_accepted,
            )

            st.success(
                f"Feedback saved. Q-learning reward applied: {reward:.2f}."
            )

    elif page == "Explainability & Tuning":
        st.header("Explainability, Class Imbalance & RF Tuning")
        S7 = STEP7_DIR
        tuned_path = S7 / "tuned_random_forest.json"
        dist_path = S7 / "class_distribution.csv"
        imp_path = S7 / "permutation_importance.csv"

        if not tuned_path.exists():
            st.warning("Run `python cyber_threat_detection.py tune` to generate these results.")
            st.stop()

        tuned = json.loads(tuned_path.read_text(encoding="utf-8"))

        if dist_path.exists():
            st.subheader("Class distribution")
            dist = pd.read_csv(dist_path)
            pivot = dist.pivot(index="class", columns="split", values="count")
            pivot = pivot.reindex(["Normal", "DoS", "Probe", "R2L", "U2R"])
            st.bar_chart(pivot)
            st.dataframe(
                dist.style.format({"percentage": "{:.2f}%"}),
                width="stretch", hide_index=True,
            )
            st.caption(
                "R2L and U2R are rare in training data, which is why overall "
                "accuracy hides weak minority-class performance."
            )

        st.subheader("Baseline vs tuned Random Forest (KDDTest+)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline accuracy", f"{tuned['baseline_accuracy']:.2%}")
        c2.metric(
            "Tuned accuracy", f"{tuned['tuned_accuracy']:.2%}",
            f"{(tuned['tuned_accuracy'] - tuned['baseline_accuracy']) * 100:+.2f} pp",
        )
        c3.metric("Baseline macro-F1", f"{tuned['baseline_f1_macro']:.3f}")
        c4.metric(
            "Tuned macro-F1", f"{tuned['tuned_f1_macro']:.3f}",
            f"{tuned['tuned_f1_macro'] - tuned['baseline_f1_macro']:+.3f}",
        )
        if tuned["tuned_improves_macro_f1"]:
            st.success("Tuning improved macro-F1 on the held-out test set.")
        else:
            st.info(
                "Tuning did not improve macro-F1 on the held-out test set. "
                "(Cross-validation on the training split does not always "
                "transfer to the shifted KDDTest+ distribution.)"
            )

        st.markdown("**Best hyper-parameters**")
        st.json(tuned["best_params"])
        st.metric("Cross-validated macro-F1 (train)", f"{tuned['best_cv_f1_macro']:.3f}")

        st.markdown("**Per-class F1 on KDDTest+**")
        st.dataframe(
            pd.DataFrame({
                "Baseline": tuned["baseline_per_class_f1"],
                "Tuned": tuned["tuned_per_class_f1"],
            }).style.format("{:.3f}"),
            width="stretch",
        )

        st.markdown("**Tuned model confusion matrix**")
        labels = tuned["labels"]
        st.dataframe(
            pd.DataFrame(tuned["tuned_confusion_matrix"], index=labels, columns=labels),
            width="stretch",
        )

        if imp_path.exists():
            st.subheader("Top permutation feature importances")
            imp = pd.read_csv(imp_path).head(15).set_index("feature")
            st.bar_chart(imp["importance_mean"])
            st.caption(
                "Permutation importance = drop in macro-F1 when a feature is "
                "shuffled (tuned model, 3,000-record test sample)."
            )

    elif page == "Project Notes":
        st.header("Project / Viva Notes")

        st.subheader("What the system contains")
        st.markdown(
            """
            - PyTorch DNN as a third supervised model
            - Weighted ensemble of RF + SVM + DNN when the DNN is available
            - Confusion matrices
            - One-vs-rest ROC curves and macro AUC
            - Risk scoring and adaptive Q-learning response
            - Human-in-the-loop feedback
            - Evaluation and report generation
            """
        )

        st.subheader("Important interpretation")
        st.warning(
            "The Q-learning response layer is an academic simulation. "
            "MONITOR/ALERT/BLOCK/ISOLATE are recommendations only; this "
            "prototype does not execute firewall rules, device isolation, "
            "packet manipulation, or other real containment actions."
        )

        st.subheader("Limitations")
        st.markdown(
            """
            1. NSL-KDD is a benchmark dataset rather than a complete representation
               of modern enterprise traffic.
            2. The Linear SVM confidence is derived from its decision score.
            3. The risk formula is a project-defined weighted score.
            4. Q-learning is trained in a simulated defensive environment.
            5. The DNN is optional because PyTorch is a heavier dependency.
            """
        )


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if _running_in_streamlit():
    dashboard()
elif __name__ == "__main__":
    main()
