# Step 7 Run Guide

Step 7 adds explainability, class-imbalance analysis, and bounded hyperparameter tuning.

## 1. Prepare the dataset

If needed:

```powershell
python step5_experiment.py
```

## 2. Run Step 7

```powershell
python step7_analysis.py
```

## 3. Results

```text
results/step7/class_distribution.csv
results/step7/tuned_random_forest.json
results/step7/permutation_importance.csv
```

## 4. Dashboard

```powershell
streamlit run dashboard/app.py
```

## Academic interpretation

- Class distribution documents imbalance instead of hiding it.
- RandomizedSearchCV uses macro-F1 so minority classes matter.
- Permutation importance shows which transformed input features most affect the tuned model's predictions.
- The tuned model should be compared with the Step 5 baseline; do not claim improvement unless the measured test metrics improve.
