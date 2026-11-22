"""
Step 4 PyTorch DNN training.

Install:
    pip install -r requirements-pytorch.txt

Run:
    python train_pytorch.py

The DNN uses the same NSL-KDD preprocessing as the RF/SVM pipeline and
stores both the model and evaluation metrics for the Step 4 dashboard.
"""

from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from src.nslkdd import download_dataset, load_nsl_file, add_project_label
from src.nsl_preprocessing import fit_transform, transform
from src.dnn_models import IDSNet, LABELS

MODEL_DIR = Path("models/nslkdd")

def metrics_dict(y_true, pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(
            y_true,
            pred,
            labels=list(range(len(LABELS))),
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            pred,
            labels=list(range(len(LABELS))),
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_path, test_path = download_dataset()
    train_df = add_project_label(load_nsl_file(train_path))
    test_df = add_project_label(load_nsl_file(test_path))

    X_train, _ = fit_transform(train_df, MODEL_DIR)
    X_test = transform(test_df, MODEL_DIR)

    label_to_id = {label: i for i, label in enumerate(LABELS)}
    y_train = np.array([label_to_id[x] for x in train_df["attack_family"]])
    y_test = np.array([label_to_id[x] for x in test_df["attack_family"]])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=512,
        shuffle=True,
    )

    model = IDSNet(X_train.shape[1], len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mean_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1:02d}/10 - loss={mean_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).numpy()
        pred = probs.argmax(axis=1)

    metrics = metrics_dict(y_test, pred)
    print(f"\nPyTorch DNN accuracy: {metrics['accuracy']:.4f}")
    print(f"PyTorch DNN F1:       {metrics['f1']:.4f}")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "labels": LABELS,
        },
        MODEL_DIR / "pytorch_dnn.pt",
    )

    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        payload = {"dataset": "NSL-KDD"}

    payload["pytorch_dnn"] = metrics
    metrics_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(f"  {MODEL_DIR / 'pytorch_dnn.pt'}")
    print(f"  {metrics_path}")

if __name__ == "__main__":
    main()
