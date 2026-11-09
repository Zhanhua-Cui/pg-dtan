import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from data import fit_scale, load_voltage, read_manifest
from model import ModelConfig, PGDTAN, VARIANTS


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    for length in (256, 271):
        for variant in VARIANTS:
            model = PGDTAN(ModelConfig(
                num_classes=3, sample_rate=100, variant=variant,
                n_fft=64, hop_length=16, channels=4,
                hidden_size=4, fusion_size=8,
            ))
            details = model(torch.randn(2, length), return_details=True)
            assert details["logits"].shape == (2, 3)
            assert details["attention"].shape == (2, (length - 1) // 16 + 1)
            assert torch.allclose(details["attention"].sum(dim=1), torch.ones(2))
            assert torch.allclose(
                details["frame_times_seconds"], torch.arange(0, length, 16) / 100,
            )
            torch.nn.functional.cross_entropy(
                details["logits"], torch.tensor([0, 2]),
            ).backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, (variant, name)
                assert torch.isfinite(parameter.grad).all(), (variant, name)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = []
        generator = np.random.default_rng(7)
        for split in ("train", "val", "test"):
            for label in ("C3", "E3"):
                for repetition in range(2):
                    identifier = f"{split}_{label}_{repetition}"
                    voltage = generator.normal(size=256).astype(np.float32)
                    np.save(root / f"{identifier}.npy", voltage)
                    rows.append({
                        "path": f"{identifier}.npy", "label": label,
                        "batch_id": split, "trial_id": identifier, "split": split,
                    })
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        records, _ = read_manifest(manifest)
        expected_train = np.concatenate([
            load_voltage(record.path, 16)
            for record in records if record.split == "train"
        ])
        assert np.isclose(fit_scale(records, 16), np.sqrt(np.mean(expected_train ** 2)))
        for record in records:
            if record.split != "train":
                np.save(record.path, np.load(record.path) * 100)
        assert np.isclose(fit_scale(records, 16), np.sqrt(np.mean(expected_train ** 2)))
        leak_manifest = root / "leak.csv"
        rows[-1]["batch_id"] = "train"
        with leak_manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        try:
            read_manifest(leak_manifest)
        except ValueError as error:
            assert "数据泄漏" in str(error)
        else:
            raise AssertionError("未能阻止跨批次泄漏")
        output = root / "runs"
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("train.py")),
            "--manifest", str(manifest), "--sample-rate", "100",
            "--baseline-samples", "16", "--variant", "all",
            "--seeds", "7", "--epochs", "1", "--batch-size", "4",
            "--n-fft", "64", "--hop-length", "16", "--channels", "4",
            "--hidden-size", "4", "--fusion-size", "8", "--output", str(output),
        ], check=True)
        summary = json.loads((output / "summary.json").read_text())
        assert set(summary) == set(VARIANTS)
        checkpoint = torch.load(output / "full/seed_7/best.pt", weights_only=True)
        restored = PGDTAN(ModelConfig(**checkpoint["config"]))
        restored.load_state_dict(checkpoint["state_dict"])
        prediction = subprocess.run([
            sys.executable, str(Path(__file__).with_name("predict.py")),
            "--checkpoint", str(output / "full/seed_7/best.pt"),
            "--voltage", str(records[0].path),
        ], check=True, capture_output=True, text=True)
        result = json.loads(prediction.stdout)
        assert result["prediction"] in checkpoint["labels"]
        assert np.isclose(sum(result["probabilities"].values()), 1)
    print("通过：六组前向/反向、时间轴、批次泄漏、训练集尺度、训练与预测。")
    print("合成数据仅验证程序流程，不代表真实识别效果。")


if __name__ == "__main__":
    main()
