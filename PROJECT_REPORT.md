# Cyber Threat and Error Detection Using Machine Learning Algorithms

**Implementation report — 2026-11-21**

## 1. Project Overview

This implementation follows the proposed hybrid cybersecurity framework with supervised threat detection, unsupervised anomaly detection, reinforcement-learning-based adaptive response, and human-in-the-loop feedback.

## 2. Implemented Architecture

```text
NSL-KDD dataset
      ↓
Cleaning + categorical encoding + numerical scaling
      ↓
Random Forest + Linear SVM + PyTorch DNN
      ↓
Isolation Forest anomaly signal
      ↓
Hybrid risk score
      ↓
Q-learning response recommendation
      ↓
Human analyst feedback
      ↓
Q-table update
```

## 3. Dataset and Preprocessing

The current implementation uses NSL-KDD as the primary benchmark. It maps traffic into five project classes: Normal, DoS, Probe, R2L, and U2R. The preprocessing pipeline scales numerical features and one-hot encodes protocol/service/flag categorical features.

## 4. Supervised Detection

The implemented supervised layer contains:

- Random Forest
- Linear Support Vector Machine
- Optional PyTorch Deep Neural Network

When the DNN model is available, the system forms a weighted ensemble using RF, SVM, and DNN class probabilities.

## 5. Unsupervised Anomaly Detection

Isolation Forest is trained on normal training traffic and produces an anomaly score used as a second signal in risk calculation.

## 6. Risk Scoring

The risk engine combines attack-family severity, supervised confidence, and anomaly score. Risk bands are:

| Risk | Score | Typical recommendation |
|---|---:|---|
| LOW | < 0.25 | MONITOR |
| MEDIUM | 0.25–<0.50 | ALERT |
| HIGH | 0.50–<0.75 | BLOCK |
| CRITICAL | ≥ 0.75 | ISOLATE |

## 7. Reinforcement Learning Response

The Q-learning state is represented by risk level, threat family, and anomaly level. The action space contains MONITOR, ALERT, BLOCK, and ISOLATE. Analyst feedback is converted into an additional reward and updates the Q-table.

The response actions are simulated recommendations and are not applied to real network infrastructure.

## 8. Evaluation Results

Evaluated on the official NSL-KDD test split (KDDTest+, 22,544 records). Precision/Recall/F1 are support-weighted; Macro-F1 weights all five classes equally and therefore reflects minority-class (R2L/U2R) performance.

| Model | Accuracy | Precision | Recall | F1 (weighted) | F1 (macro) | Macro AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 73.23% | 78.37% | 73.23% | 68.45% | 47.25% | 0.920 |
| Linear Svm | 75.28% | 77.22% | 75.28% | 72.56% | 53.38% | 0.883 |
| Pytorch Dnn | 74.79% | 80.16% | 74.79% | 71.45% | 59.31% | 0.910 |
| Ensemble | 76.25% | 82.48% | 76.25% | 72.44% | 57.40% | 0.927 |

### 8.1 Random Forest tuning (Step 7)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline RF | 73.23% | 47.25% |
| Tuned RF | 74.51% | 53.40% |

Best parameters: `{'n_estimators': 150, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': None}`; cross-validated macro-F1 on the training split = 0.9312.

Tuned model improved test macro-F1: **yes**.

Per-class F1 (tuned): Normal 0.774, DoS 0.850, Probe 0.714, R2L 0.110, U2R 0.222.

Top-10 permutation feature importances:

| Feature | Importance (macro-F1 drop) |
|---|---:|
| num__dst_host_same_src_port_rate | 0.0207 |
| cat__service_ftp_data | 0.0203 |
| num__duration | 0.0133 |
| num__dst_host_srv_count | 0.0128 |
| num__dst_host_count | 0.0096 |
| num__count | 0.0089 |
| num__dst_host_same_srv_rate | 0.0078 |
| cat__flag_SF | 0.0062 |
| cat__service_http | 0.0056 |
| num__diff_srv_rate | 0.0050 |

## 9. Confusion Matrix and ROC Analysis

The dashboard provides confusion matrices for each evaluated model and one-vs-rest ROC curves for the five attack classes. Macro AUC is reported as the mean of the class-wise AUC values.

## 10. Human-in-the-Loop Module

The dashboard records analyst acceptance/rejection decisions and comments. Feedback is persisted in `data/feedback.csv` and used to update the Q-learning policy.

## 11. Alignment With Proposed Project

The implementation is designed around the project's stated goal of combining supervised, unsupervised, and reinforcement learning in one cooperating cybersecurity system, with centralized alerting and human validation.

## 12. Limitations

- NSL-KDD is an academic benchmark and does not cover every modern threat.
- The DNN is optional because PyTorch is a heavier dependency.
- U2R (52 training records) and R2L (995) are heavily under-represented, so minority-class recall is weak despite class weighting.
- KDDTest+ has a different distribution from KDDTrain+, so test accuracy (~73-76%) is much lower than training/CV accuracy.
- Linear SVM confidence is a decision-score-based heuristic.
- The risk formula is a project-defined weighting scheme.
- Q-learning is trained in a simulated defensive environment.
- No real firewall, isolation, or containment action is executed.

## 13. Demo Procedure

```text
python run_all.py
streamlit run dashboard/app.py
        ↓
Dashboard
        ↓
Detection & Response
        ↓
Show model predictions + anomaly score
        ↓
Show risk level + Q-learning recommendation
        ↓
Human Feedback
        ↓
Accept / Reject
        ↓
Q-table update
```

## 14. Conclusion

The prototype provides an end-to-end academic implementation of the proposed hybrid architecture: supervised classification for known attack families, anomaly detection for unusual behaviour, adaptive Q-learning response recommendations, and human feedback for continuous policy refinement.
