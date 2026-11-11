import json

import numpy as np
import torch
import torch.nn as nn


class RiskModel(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def predict_scores(model, x):
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32)
        logits = model(tensor)
        probs = torch.sigmoid(logits).numpy()
    return probs


def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    count = 0.0
    for p in pos:
        count += float(np.sum(neg < p)) + 0.5 * float(np.sum(neg == p))
    return float(count / (len(pos) * len(neg)))


def pr_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total_pos = float(np.sum(y_true == 1))
    if total_pos == 0:
        return 0.0
    precisions = []
    recalls = []
    tp = 0
    fp = 0
    for i, label in enumerate(y_sorted):
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_pos)
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return float(ap)


def compute_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(pr_auc_score(y_true, y_score)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": int(total),
        "threshold": float(threshold),
    }


def train_model(x_train, y_train, x_val, y_val, epochs=30, lr=0.001, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    input_dim = x_train.shape[1]
    model = RiskModel(input_dim=input_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = float(np.sum(y_train == 1))
    n_neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor(n_neg / max(1.0, n_pos), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    n = x_train.shape[0]
    batch = 64
    history = {"train_loss": [], "val_roc_auc": []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        steps = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = x_t[idx]
            yb = y_t[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1
        history["train_loss"].append(total_loss / max(1, steps))
        val_scores = predict_scores(model, x_val)
        history["val_roc_auc"].append(float(roc_auc_score(y_val, val_scores)))
    return model, history


def save_model(model, feature_names, scaler_mean, scaler_std, path):
    payload = {
        "state": model.state_dict(),
        "input_dim": int(model.net[0].in_features),
        "feature_names": list(feature_names),
        "scaler_mean": list(map(float, scaler_mean)),
        "scaler_std": list(map(float, scaler_std)),
    }
    torch.save(payload, path)
    return path


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = RiskModel(input_dim=int(payload["input_dim"]))
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload
