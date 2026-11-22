from pathlib import Path
import numpy as np
import torch
from torch import nn

LABELS = ["Normal", "DoS", "Probe", "R2L", "U2R"]

class IDSNet(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_dnn(model_path="models/nslkdd/pytorch_dnn.pt"):
    payload = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    model = IDSNet(payload["input_dim"], len(payload["labels"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload["labels"]

def predict_proba(X, model_path="models/nslkdd/pytorch_dnn.pt"):
    model, labels = load_dnn(model_path)
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs, labels
