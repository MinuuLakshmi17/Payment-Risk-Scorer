import json
from pathlib import Path

import numpy as np

from src.generator import FEATURE_NAMES, generate_transactions, encode_transactions
from src.model import compute_metrics, predict_scores, save_model, train_model

SEED = 42
N = 8000
EPOCHS = 60


def main():
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    data = generate_transactions(n=N, seed=SEED)
    x, y_fraud, y_comp, names = encode_transactions(data)
    assert names == FEATURE_NAMES
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(data))
    n_train = int(0.70 * len(data))
    n_val = int(0.15 * len(data))
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    x_train = x[train_idx].copy()
    x_val = x[val_idx].copy()
    x_test = x[test_idx].copy()
    y_train = y_fraud[train_idx]
    y_val = y_fraud[val_idx]
    y_test = y_fraud[test_idx]
    y_comp_test = y_comp[test_idx]
    mean = np.zeros((x.shape[1],), dtype=np.float32)
    std = np.ones((x.shape[1],), dtype=np.float32)
    for col in [0, 1]:
        mean[col] = float(np.mean(x_train[:, col]))
        std[col] = float(np.std(x_train[:, col])) or 1.0
    def scale(a):
        b = a.copy()
        b[:, 0] = (b[:, 0] - mean[0]) / std[0]
        b[:, 1] = (b[:, 1] - mean[1]) / std[1]
        return b
    x_train_s = scale(x_train)
    x_val_s = scale(x_val)
    x_test_s = scale(x_test)
    model, history = train_model(x_train_s, y_train, x_val_s, y_val, epochs=EPOCHS, seed=SEED)
    test_scores = predict_scores(model, x_test_s)
    fraud_metrics = compute_metrics(y_test, test_scores, threshold=0.5)
    comp_metrics = compute_metrics(y_comp_test, test_scores, threshold=0.5)
    comp_metrics["roc_auc"] = float(__import__("src.model", fromlist=["roc_auc_score"]).roc_auc_score(y_comp_test, test_scores))
    comp_metrics["pr_auc"] = float(__import__("src.model", fromlist=["pr_auc_score"]).pr_auc_score(y_comp_test, test_scores))
    save_model(model, FEATURE_NAMES, mean, std, str(models_dir / "model.pt"))
    from export_weights import export_weights

    export_weights(models_dir / "model.pt", models_dir / "weights.json")
    out = {
        "n": int(N),
        "seed": int(SEED),
        "epochs": int(EPOCHS),
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
        "fraud_rate_train": float(np.mean(y_train)),
        "fraud_rate_test": float(np.mean(y_test)),
        "compliance_rate_test": float(np.mean(y_comp_test)),
        "fraud_metrics": fraud_metrics,
        "compliance_metrics_using_same_score": comp_metrics,
        "val_roc_auc_last": float(history["val_roc_auc"][-1]),
        "train_loss_last": float(history["train_loss"][-1]),
    }
    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Training complete")
    print("Test n:", out["test_n"])
    print("Fraud ROC AUC:", fraud_metrics["roc_auc"])
    print("Fraud PR AUC:", fraud_metrics["pr_auc"])
    print("Fraud accuracy:", fraud_metrics["accuracy"])
    print("Fraud precision:", fraud_metrics["precision"])
    print("Fraud recall:", fraud_metrics["recall"])
    print("Fraud F1:", fraud_metrics["f1"])
    print("Compliance ROC AUC:", comp_metrics["roc_auc"])
    print("Compliance PR AUC:", comp_metrics["pr_auc"])
    print("Model saved to models/model.pt")
    print("Metrics saved to models/metrics.json")


if __name__ == "__main__":
    main()
