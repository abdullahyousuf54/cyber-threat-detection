"""Step 7 - class-imbalance analysis, bounded RF tuning, permutation importance.

Run after the dataset is present (run_all.py downloads it):
    python step7_analysis.py
Outputs go to results/step7/.
"""
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "step7"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.inspection import permutation_importance

    from src.nslkdd import (
        FEATURE_NAMES, load_nsl_file, add_project_label,
    )
    from src.nsl_models import LABELS
    from src.nsl_preprocessing import build_preprocessor

    train_path = ROOT / "data" / "nsl_kdd" / "KDDTrain+.txt"
    test_path = ROOT / "data" / "nsl_kdd" / "KDDTest+.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Run `python run_all.py` or place KDDTrain+.txt/KDDTest+.txt "
            "in data/nsl_kdd/ first."
        )

    train = add_project_label(load_nsl_file(train_path))
    test = add_project_label(load_nsl_file(test_path))

    # ---- class imbalance report (train and test) ----
    rows = []
    for split, df in [("train", train), ("test", test)]:
        counts = df["attack_family"].value_counts().reindex(LABELS, fill_value=0)
        for cls, n in counts.items():
            rows.append({
                "split": split, "class": cls, "count": int(n),
                "percentage": 100.0 * n / counts.sum(),
            })
    pd.DataFrame(rows).to_csv(OUT / "class_distribution.csv", index=False)

    pre = build_preprocessor()
    Xtr = pre.fit_transform(train[FEATURE_NAMES])
    Xte = pre.transform(test[FEATURE_NAMES])
    ytr = train["attack_family"].to_numpy(dtype=object)
    yte = test["attack_family"].to_numpy(dtype=object)

    def summarise(model, tag):
        pred = model.predict(Xte)
        rep = classification_report(
            yte, pred, labels=LABELS, target_names=LABELS,
            output_dict=True, zero_division=0,
        )
        return {
            f"{tag}_accuracy": float(rep["accuracy"]),
            f"{tag}_f1_macro": float(rep["macro avg"]["f1-score"]),
            f"{tag}_f1_weighted": float(rep["weighted avg"]["f1-score"]),
            f"{tag}_per_class_f1": {
                c: float(rep[c]["f1-score"]) for c in LABELS
            },
            f"{tag}_confusion_matrix": confusion_matrix(
                yte, pred, labels=LABELS
            ).tolist(),
        }

    # ---- baseline RF (same hyper-parameters as the main pipeline) ----
    baseline = RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1,
        class_weight="balanced_subsample", max_features="sqrt",
    ).fit(Xtr, ytr)

    # ---- bounded RF tuning on macro-F1 ----
    params = {
        "n_estimators": [100, 150, 250],
        "max_depth": [None, 12, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        ),
        params, n_iter=8, scoring="f1_macro", cv=3,
        random_state=42, n_jobs=1, verbose=1,
    )
    t0 = time.time()
    search.fit(Xtr, ytr)
    elapsed = time.time() - t0

    metrics = {
        "best_params": search.best_params_,
        "best_cv_f1_macro": float(search.best_score_),
        "training_seconds": float(elapsed),
        "labels": LABELS,
    }
    metrics.update(summarise(baseline, "baseline"))
    metrics.update(summarise(search.best_estimator_, "tuned"))
    metrics["tuned_improves_macro_f1"] = bool(
        metrics["tuned_f1_macro"] > metrics["baseline_f1_macro"]
    )
    # Backwards-compatible keys used by earlier notes
    metrics["test_f1_macro"] = metrics["tuned_f1_macro"]
    metrics["test_accuracy"] = metrics["tuned_accuracy"]
    metrics["confusion_matrix"] = metrics["tuned_confusion_matrix"]
    (OUT / "tuned_random_forest.json").write_text(json.dumps(metrics, indent=2))

    # ---- permutation importance on the tuned model ----
    rng = np.random.default_rng(42)
    n = min(3000, Xte.shape[0])
    idx = rng.choice(Xte.shape[0], size=n, replace=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi = permutation_importance(
            search.best_estimator_, Xte[idx], yte[idx], n_repeats=3,
            random_state=42, n_jobs=-1, scoring="f1_macro",
        )
    imp = pd.DataFrame({
        "feature": pre.get_feature_names_out(),
        "importance_mean": pi.importances_mean,
        "importance_std": pi.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp.to_csv(OUT / "permutation_importance.csv", index=False)

    print("Step 7 analysis complete.")
    print("Best parameters  :", search.best_params_)
    print(f"CV macro-F1      : {search.best_score_:.4f}")
    print(f"Baseline RF      : acc={metrics['baseline_accuracy']:.4f} "
          f"macro-F1={metrics['baseline_f1_macro']:.4f}")
    print(f"Tuned RF         : acc={metrics['tuned_accuracy']:.4f} "
          f"macro-F1={metrics['tuned_f1_macro']:.4f}")
    print("Tuned beats baseline on macro-F1:", metrics["tuned_improves_macro_f1"])
    print("Outputs:", OUT)


if __name__ == "__main__":
    main()
