
from pathlib import Path
import json
import hmac
import logging
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nslkdd import FEATURE_NAMES, load_nsl_file, add_project_label
from src.nsl_models import predict
from src.ensemble import predict_ensemble
from src.risk_engine import calculate_risk
from src.q_learning import QLearningResponseEngine, state_key
from src.feedback import append_feedback, load_feedback
from src.asset_loader import ensure_assets

# set_page_config must be the first Streamlit call and may only be used once.
st.set_page_config(
    page_title="Cyber Threat Detection & Response",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
try:
    ensure_assets(ROOT)
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()
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

MODEL_DIR = ROOT / "models" / "nslkdd"
DATA_DIR = ROOT / "data" / "nsl_kdd"
METRICS_PATH = MODEL_DIR / "metrics.json"
EVAL_PATH = MODEL_DIR / "evaluation.json"
TEST_PATH = DATA_DIR / "KDDTest+.txt"
Q_PATH = MODEL_DIR / "q_table.json"
FEEDBACK_PATH = ROOT / "data" / "feedback.csv"
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")

st.title("🛡️ Cyber Threat and Error Detection")
st.caption(
    "Hybrid ML + anomaly detection + adaptive Q-learning + "
    "human-in-the-loop"
)

if not METRICS_PATH.exists():
    st.error("Models are missing. Run `python run_all.py` first.")
    st.stop()

if not Q_PATH.exists():
    st.warning(
        "Q-learning model is missing. Run `python train_qlearning.py`."
    )
    st.stop()

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
evaluation = (
    json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if EVAL_PATH.exists()
    else None
)
q_engine = QLearningResponseEngine(model_path=Q_PATH)
cloud_store = None
if DATABASE_URL:
    from src.cloud_store import CloudStore

    cloud_store = CloudStore(DATABASE_URL)
    try:
        q_engine.q = cloud_store.load_or_seed_policy(q_engine.q)
    except Exception:
        LOG.exception("Cannot connect to the feedback database")
        st.error("Cannot connect to the feedback database. Check app logs and DATABASE_URL.")
        st.stop()

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
            "Run `python evaluate_step4.py` after training to populate "
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
        st.warning("Evaluation file not found. Run `python evaluate_step4.py`.")

elif page == "ROC Curves":
    st.header("Multiclass ROC Curves")

    if not evaluation:
        st.warning("Run `python evaluate_step4.py` first.")
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
        st.error("KDDTest+.txt is missing. Run `python step2_train.py`.")
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

    if cloud_store:
        if not ADMIN_PASSWORD:
            st.warning("Set ADMIN_PASSWORD in Streamlit secrets to enable analyst feedback.")
            st.stop()
        entered_password = st.text_input("Analyst password", type="password")
        if not hmac.compare_digest(entered_password, ADMIN_PASSWORD):
            st.info("Enter the analyst password to view or submit feedback.")
            st.stop()
        feedback_df = cloud_store.load_feedback()
    else:
        st.caption("Local mode: feedback is stored in data/feedback.csv.")
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

        if cloud_store:
            try:
                reward, q_engine.q = cloud_store.record_feedback(
                    q_engine, state, recommended, is_accepted,
                    risk_score, anomaly_score, comment,
                )
            except Exception:
                LOG.exception("Feedback was not saved")
                st.error("Feedback was not saved. Check the app logs.")
                st.stop()
        else:
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
    S7 = ROOT / "results" / "step7"
    tuned_path = S7 / "tuned_random_forest.json"
    dist_path = S7 / "class_distribution.csv"
    imp_path = S7 / "permutation_importance.csv"

    if not tuned_path.exists():
        st.warning("Run `python step7_analysis.py` to generate these results.")
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
