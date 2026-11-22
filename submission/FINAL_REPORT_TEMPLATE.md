# Final Project Report — Submission Version

## Abstract
This project presents a machine-learning-based cyber threat detection and adaptive response prototype. The system combines supervised classification using Random Forest, Linear SVM, and a PyTorch DNN with Isolation Forest anomaly detection. Model outputs are combined into a project-defined risk score, followed by a Q-learning response recommendation. Human analyst feedback is incorporated into the response policy. The system is demonstrated through a Streamlit dashboard.

## 1. Introduction
Cybersecurity monitoring must identify known attack patterns while also detecting unusual behavior. A combined supervised and anomaly-detection architecture can provide complementary signals. This project therefore integrates classification, anomaly detection, risk assessment, adaptive response, and human feedback.

## 2. Objectives
- Detect cyber threats using machine-learning classifiers.
- Detect anomalous observations.
- Combine model outputs into a risk assessment.
- Recommend adaptive responses using Q-learning.
- Incorporate analyst feedback.
- Provide a demonstrable dashboard.
- Evaluate the models using reproducible metrics.

## 3. Dataset
The experimental pipeline uses NSL-KDD. The exact numerical results must be copied from the generated Step 5/Step 7 artifacts after running them on the actual dataset files.

## 4. Methodology
### 4.1 Preprocessing
Categorical features are one-hot encoded and numerical features are standardized.

### 4.2 Supervised detection
Random Forest, Linear SVM, and a PyTorch multilayer neural network are trained for the project threat classes.

### 4.3 Anomaly detection
Isolation Forest is trained using normal training observations and provides an additional anomaly signal.

### 4.4 Ensemble
The ensemble combines available classifier outputs with configured weights. If the DNN is unavailable, active weights are renormalized.

### 4.5 Risk scoring
Detection and anomaly signals are converted into project-defined risk levels.

### 4.6 Adaptive response
Q-learning selects among MONITOR, ALERT, BLOCK, and ISOLATE based on the risk state and learned policy.

### 4.7 Human feedback
Analyst Accept/Reject feedback is stored and used to update the Q-learning policy.

## 5. Evaluation
Insert the measured values from:
- `models/nslkdd/metrics.json`
- `models/nslkdd/evaluation.json`
- `results/step7/tuned_random_forest.json`

Do not insert unmeasured or estimated values.

## 6. Explainability
Step 7 generates permutation feature importance in `results/step7/permutation_importance.csv`.

## 7. Limitations
The benchmark dataset is not equivalent to live enterprise traffic. The response actions are simulated recommendations. Model confidence and risk scoring require further calibration for production deployment.

## 8. Future Scope
Future work can include live log/packet ingestion, Autoencoders, SHAP explanations, calibrated probabilities, online learning, larger contemporary datasets, and authorized integration with real security controls.

## 9. Conclusion
The final prototype demonstrates an end-to-end academic cybersecurity ML workflow from data preprocessing through detection, anomaly analysis, risk assessment, adaptive response, and analyst feedback. Its measured results should be reported only after executing the supplied experiment pipeline.
