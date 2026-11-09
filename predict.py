import argparse
import json

import torch

from data import load_voltage
from model import ModelConfig, PGDTAN


def main():
    parser = argparse.ArgumentParser(description="使用 PG-DTAN 权重预测单段电压")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--voltage", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PGDTAN(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    voltage = load_voltage(args.voltage, checkpoint["baseline_samples"])
    if len(voltage) != checkpoint["num_samples"]:
        raise ValueError("预测样本长度必须与训练时间窗一致")
    tensor = torch.tensor(voltage / checkpoint["scale"], dtype=torch.float32)
    with torch.no_grad():
        details = model(tensor.unsqueeze(0), return_details=True)
        probabilities = details["logits"].softmax(dim=-1)[0]
    labels = checkpoint["labels"]
    print(json.dumps({
        "prediction": labels[probabilities.argmax().item()],
        "probabilities": dict(zip(labels, probabilities.tolist())),
        "frame_times_seconds": details["frame_times_seconds"].tolist(),
        "attention": details["attention"][0].tolist(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
