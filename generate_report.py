from pathlib import Path
import json
from datetime import date

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "models" / "nslkdd" / "metrics.json"
EVAL = ROOT / "models" / "nslkdd" / "evaluation.json"
OUT = ROOT / "PROJECT_REPORT.md"

def pct(x):
    return f"{x * 100:.2f}%"

def main():
    metrics = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    evaluation = json.loads(EVAL.read_text(encoding="utf-8")) if EVAL.exists() else {}

    lines = [
        "# Cyber Threat and Error Detection Using Machine Learning Algorithms",
        "",
        f"**Implementation report — {date.today().isoformat()}**",
        "",
        "## 1. Project Overview",
        "",
        "This implementation follows the proposed hybrid cybersecurity framework "
        "with supervised threat detection, unsupervised anomaly detection, "
        "reinforcement-learning-based adaptive response, and human-in-the-loop "
        "feedback.",
        "",
        "## 2. Implemented Architecture",
        "",
        "```text",
        "NSL-KDD dataset",
        "      ↓",
        "Cleaning + categorical encoding + numerical scaling",
        "      ↓",
        "Random Forest + Linear SVM + PyTorch DNN",
        "      ↓",
        "Isolation Forest anomaly signal",
        "      ↓",
        "Hybrid risk score",
        "      ↓",
        "Q-learning response recommendation",
        "      ↓",
        "Human analyst feedback",
        "      ↓",
        "Q-table update",
        "```",
        "",
        "## 3. Dataset and Preprocessing",
        "",
        "The current implementation uses NSL-KDD as the primary benchmark. "
        "It maps traffic into five project classes: Normal, DoS, Probe, R2L, "
        "and U2R. The preprocessing pipeline scales numerical features and "
        "one-hot encodes protocol/service/flag categorical features.",
        "",
        "## 4. Supervised Detection",
        "",
        "The implemented supervised layer contains:",
        "",
        "- Random Forest",
        "- Linear Support Vector Machine",
        "- Optional PyTorch Deep Neural Network",
        "",
        "When the DNN model is available, the system forms a weighted ensemble "
        "using RF, SVM, and DNN class probabilities.",
        "",
        "## 5. Unsupervised Anomaly Detection",
        "",
        "Isolation Forest is trained on normal training traffic and produces "
        "an anomaly score used as a second signal in risk calculation.",
        "",
        "## 6. Risk Scoring",
        "",
        "The risk engine combines attack-family severity, supervised confidence, "
        "and anomaly score. Risk bands are:",
        "",
        "| Risk | Score | Typical recommendation |",
        "|---|---:|---|",
        "| LOW | < 0.25 | MONITOR |",
        "| MEDIUM | 0.25–<0.50 | ALERT |",
        "| HIGH | 0.50–<0.75 | BLOCK |",
        "| CRITICAL | ≥ 0.75 | ISOLATE |",
        "",
        "## 7. Reinforcement Learning Response",
        "",
        "The Q-learning state is represented by risk level, threat family, and "
        "anomaly level. The action space contains MONITOR, ALERT, BLOCK, and "
        "ISOLATE. Analyst feedback is converted into an additional reward and "
        "updates the Q-table.",
        "",
        "The response actions are simulated recommendations and are not applied "
        "to real network infrastructure.",
        "",
        "## 8. Evaluation Results",
        "",
    ]

    if evaluation:
        lines += [
            "Evaluated on the official NSL-KDD test split (KDDTest+, "
            "22,544 records). Precision/Recall/F1 are support-weighted; "
            "Macro-F1 weights all five classes equally and therefore "
            "reflects minority-class (R2L/U2R) performance.",
            "",
            "| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) | Macro AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, result in evaluation["models"].items():
            lines.append(
                f"| {name.replace('_', ' ').title()} | "
                f"{pct(result['accuracy'])} | "
                f"{pct(result['precision'])} | "
                f"{pct(result['recall'])} | "
                f"{pct(result['f1'])} | "
                f"{pct(result.get('f1_macro', float('nan')))} | "
                f"{result['roc']['macro_auc']:.3f} |"
            )
    else:
        lines += [
            "_Run `python evaluate_step4.py` to populate this table with "
            "measured results._",
        ]

    tuned_path = ROOT / "results" / "step7" / "tuned_random_forest.json"
    if tuned_path.exists():
        t = json.loads(tuned_path.read_text(encoding="utf-8"))
        lines += [
            "",
            "### 8.1 Random Forest tuning (Step 7)",
            "",
            "| Model | Accuracy | Macro-F1 |",
            "|---|---:|---:|",
            f"| Baseline RF | {pct(t['baseline_accuracy'])} | {pct(t['baseline_f1_macro'])} |",
            f"| Tuned RF | {pct(t['tuned_accuracy'])} | {pct(t['tuned_f1_macro'])} |",
            "",
            f"Best parameters: `{t['best_params']}`; cross-validated "
            f"macro-F1 on the training split = {t['best_cv_f1_macro']:.4f}.",
            "",
            "Tuned model improved test macro-F1: "
            f"**{'yes' if t['tuned_improves_macro_f1'] else 'no'}**.",
            "",
            "Per-class F1 (tuned): "
            + ", ".join(f"{k} {v:.3f}" for k, v in t["tuned_per_class_f1"].items())
            + ".",
        ]
    imp_path = ROOT / "results" / "step7" / "permutation_importance.csv"
    if imp_path.exists():
        import pandas as pd
        top = pd.read_csv(imp_path).head(10)
        lines += ["", "Top-10 permutation feature importances:", "",
                  "| Feature | Importance (macro-F1 drop) |", "|---|---:|"]
        for _, r in top.iterrows():
            lines.append(f"| {r['feature']} | {r['importance_mean']:.4f} |")

    lines += [
        "",
        "## 9. Confusion Matrix and ROC Analysis",
        "",
        "The dashboard provides confusion matrices for each evaluated "
        "model and one-vs-rest ROC curves for the five attack classes. "
        "Macro AUC is reported as the mean of the class-wise AUC values.",
        "",
        "## 10. Human-in-the-Loop Module",
        "",
        "The dashboard records analyst acceptance/rejection decisions and "
        "comments. Feedback is persisted in `data/feedback.csv` and used to "
        "update the Q-learning policy.",
        "",
        "## 11. Alignment With Proposed Project",
        "",
        "The implementation is designed around the project's stated goal of "
        "combining supervised, unsupervised, and reinforcement learning in "
        "one cooperating cybersecurity system, with centralized alerting and "
        "human validation.",
        "",
        "## 12. Limitations",
        "",
        "- NSL-KDD is an academic benchmark and does not cover every modern threat.",
        "- The DNN is optional because PyTorch is a heavier dependency.",
        "- U2R (52 training records) and R2L (995) are heavily under-represented, so minority-class recall is weak despite class weighting.",
        "- KDDTest+ has a different distribution from KDDTrain+, so test accuracy (~73-76%) is much lower than training/CV accuracy.",
        "- Linear SVM confidence is a decision-score-based heuristic.",
        "- The risk formula is a project-defined weighting scheme.",
        "- Q-learning is trained in a simulated defensive environment.",
        "- No real firewall, isolation, or containment action is executed.",
        "",
        "## 13. Demo Procedure",
        "",
        "```text",
        "python run_all.py",
        "streamlit run dashboard/app.py",
        "        ↓",
        "Dashboard",
        "        ↓",
        "Detection & Response",
        "        ↓",
        "Show model predictions + anomaly score",
        "        ↓",
        "Show risk level + Q-learning recommendation",
        "        ↓",
        "Human Feedback",
        "        ↓",
        "Accept / Reject",
        "        ↓",
        "Q-table update",
        "```",
        "",
        "## 14. Conclusion",
        "",
        "The prototype provides an end-to-end academic implementation "
        "of the proposed hybrid architecture: supervised classification for "
        "known attack families, anomaly detection for unusual behaviour, "
        "adaptive Q-learning response recommendations, and human feedback "
        "for continuous policy refinement.",
    ]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {OUT}")

if __name__ == "__main__":
    main()
