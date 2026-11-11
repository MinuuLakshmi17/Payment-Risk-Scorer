import json
from pathlib import Path

from src.model import load_model


def export_weights(model_path, out_path):
    model, payload = load_model(str(model_path))
    state = model.state_dict()
    data = {
        "input_dim": int(payload["input_dim"]),
        "w1": state["net.0.weight"].tolist(),
        "b1": state["net.0.bias"].tolist(),
        "w2": state["net.3.weight"].tolist(),
        "b2": state["net.3.bias"].tolist(),
        "w3": state["net.5.weight"].tolist(),
        "b3": state["net.5.bias"].tolist(),
        "scaler_mean": list(map(float, payload["scaler_mean"])),
        "scaler_std": list(map(float, payload["scaler_std"])),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path


def main():
    root = Path(__file__).resolve().parent
    out = export_weights(root / "models" / "model.pt", root / "models" / "weights.json")
    print("Weights written to", out)


if __name__ == "__main__":
    main()
