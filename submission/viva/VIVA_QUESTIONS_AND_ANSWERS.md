# Viva Questions and Answers

## 1. What is the objective of the project?
To build a machine-learning-based cyber threat detection and adaptive response prototype that combines supervised classification, anomaly detection, risk scoring, reinforcement-learning-based response recommendations, and human feedback.

## 2. Why NSL-KDD?
NSL-KDD provides a structured benchmark for intrusion-detection research and contains normal traffic plus multiple attack families. It is suitable for reproducible academic experimentation.

## 3. Why use Random Forest?
Random Forest is a strong baseline for tabular classification, can model nonlinear relationships, and provides feature-importance capabilities.

## 4. Why SVM?
SVM provides a different decision boundary and gives a useful comparison against tree-based methods.

## 5. Why DNN?
The DNN provides a neural-network-based classifier and demonstrates a modern deep-learning approach for the same tabular intrusion-detection task.

## 6. Why an ensemble?
Different models can make different errors. Combining their outputs can provide a more robust prediction than relying on a single classifier.

## 7. What is anomaly detection doing?
Isolation Forest provides an additional anomaly signal, especially useful for identifying observations that differ from the normal training distribution.

## 8. What is the risk score?
It is a project-defined score combining the classification/anomaly signals into LOW, MEDIUM, HIGH, or CRITICAL risk levels.

## 9. Why Q-learning?
Q-learning provides an adaptive policy mechanism in which response actions receive rewards/penalties based on feedback.

## 10. What actions are supported?
MONITOR, ALERT, BLOCK, and ISOLATE.

## 11. Does the project actually block a real computer or network?
No. The response layer is a simulated recommendation system for safe academic demonstration.

## 12. What is human-in-the-loop?
An analyst can accept or reject a recommended action. The feedback is used to update the Q-learning policy.

## 13. What metrics are used?
Accuracy, precision, recall, F1-score, macro-F1, confusion matrices, and ROC/AUC where applicable.

## 14. Why macro-F1?
Macro-F1 gives equal weight to each class and therefore exposes poor minority-class performance that can be hidden by overall accuracy.

## 15. What is class imbalance?
It occurs when some attack classes contain far fewer examples than others. The project explicitly measures the class distribution and uses class-aware evaluation.

## 16. What does Step 7 add?
Bounded Random Forest hyperparameter tuning, class-distribution analysis, and permutation feature importance.

## 17. What is permutation importance?
It measures how much model performance changes when a feature's values are randomly permuted, indicating the feature's contribution to prediction quality.

## 18. What are the main limitations?
NSL-KDD is a benchmark rather than live enterprise traffic; the response actions are simulated; SVM confidence is not a calibrated probability; and the risk formula is project-defined.

## 19. What is future scope?
Live packet/log ingestion, stronger datasets, calibrated probabilities, Autoencoders, SHAP-based explanations, online learning, and integration with real security controls under strict authorization.

## 20. What is the key contribution?
The project connects detection, anomaly scoring, adaptive response, and analyst feedback into one demonstrable pipeline rather than treating classification as an isolated task.
