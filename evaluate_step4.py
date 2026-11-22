from pathlib import Path
import json

from sklearn.metrics import accuracy_score

from src.nslkdd import download_dataset, load_nsl_file, add_project_label
from src.nsl_models import predict, LABELS
from src.nsl_preprocessing import transform
from src.ensemble import predict_ensemble
from src.evaluation import classification_metrics, roc_payload, save_evaluation

def main():
    print("\n=== STEP 4: MODEL EVALUATION + ENSEMBLE ===\n")

    _, test_path = download_dataset()
    test_df = add_project_label(load_nsl_file(test_path))
    y_true = test_df["attack_family"].astype(str).tolist()

    base = predict(test_df)

    rf_prob = None
    svm_prob = None

    # Recreate probabilities using the ensemble utility so ROC data uses
    # the same probability representation as the ensemble.
    ensemble_df, components, weights = predict_ensemble(test_df)

    payload = {
        "dataset": "NSL-KDD",
        "active_ensemble_weights": weights,
        "models": {},
    }

    for key, probs in components.items():
        if key == "random_forest":
            pred = ensemble_df["rf_prediction"].tolist()
            name = "random_forest"
        elif key == "linear_svm":
            pred = ensemble_df["linear_svm_prediction"].tolist()
            name = "linear_svm"
        else:
            pred = ensemble_df["dnn_prediction"].tolist()
            name = "pytorch_dnn"

        metrics = classification_metrics(y_true, pred)
        metrics["roc"] = roc_payload(y_true, probs)
        payload["models"][name] = metrics

        print(
            f"{name:16s} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"macro_auc={metrics['roc']['macro_auc']:.4f}"
        )

    ensemble_pred = ensemble_df["ensemble_prediction"].tolist()
    ensemble_probs = sum(
        (weights[name] / sum(weights.values())) * probs
        for name, probs in components.items()
    )
    ensemble_metrics = classification_metrics(y_true, ensemble_pred)
    ensemble_metrics["roc"] = roc_payload(y_true, ensemble_probs)
    payload["models"]["ensemble"] = ensemble_metrics

    print(
        f"{'ensemble':16s} "
        f"accuracy={ensemble_metrics['accuracy']:.4f} "
        f"f1={ensemble_metrics['f1']:.4f} "
        f"macro_auc={ensemble_metrics['roc']['macro_auc']:.4f}"
    )

    save_evaluation("models/nslkdd/evaluation.json", payload)
    print("\nSaved models/nslkdd/evaluation.json")

if __name__ == "__main__":
    main()
