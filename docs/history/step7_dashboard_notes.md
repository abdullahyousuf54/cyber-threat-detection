# Step 7 Dashboard Additions

After running `python step7_analysis.py`, the following artifacts are available:

- `results/step7/class_distribution.csv`
- `results/step7/tuned_random_forest.json`
- `results/step7/permutation_importance.csv`

These support:
- class-imbalance visualization
- tuned RF vs baseline comparison
- top feature-importance chart
- best hyperparameters
- cross-validation macro-F1
- measured test macro-F1/accuracy
- confusion matrix from the tuned model

The importance method is permutation importance rather than SHAP, so it does not require the additional SHAP dependency.
