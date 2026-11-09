import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data import VoltageDataset, fit_scale, load_voltage, read_manifest
from model import ModelConfig, PGDTAN, VARIANTS


def evaluate(model, loader, device, num_classes):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    loss_sum = 0.0
    with torch.no_grad():
        for voltage, targets in loader:
            logits = model(voltage.to(device))
            loss_sum += nn.functional.cross_entropy(
                logits, targets.to(device), reduction="sum",
            ).item()
            predictions = logits.argmax(dim=1).cpu()
            counts = torch.bincount(
                targets * num_classes + predictions, minlength=num_classes ** 2,
            )
            confusion += counts.reshape(num_classes, num_classes)
    correct = confusion.diag().float()
    denominators = confusion.sum(dim=0) + confusion.sum(dim=1)
    f1 = 2 * correct / denominators.clamp_min(1)
    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": correct.sum().item() / len(loader.dataset),
        "macro_f1": f1.mean().item(),
        "confusion_matrix": confusion.tolist(),
        "support": confusion.sum(dim=1).tolist(),
    }


def run(args, records, labels, scale, variant, seed, output):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    config = ModelConfig(
        num_classes=len(labels), sample_rate=args.sample_rate, variant=variant,
        n_fft=args.n_fft, hop_length=args.hop_length, channels=args.channels,
        hidden_size=args.hidden_size, fusion_size=args.fusion_size,
        smoothing_kernel=args.smoothing_kernel, dropout=args.dropout,
    )
    device = torch.device(args.device)
    model = PGDTAN(config).to(device)
    datasets = {
        split: VoltageDataset(records, labels, split, args.baseline_samples, scale)
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=split == "train",
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    best_score = -1.0
    best_state = None
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for voltage, targets in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(voltage.to(device))
            loss = nn.functional.cross_entropy(logits, targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError("训练损失出现非有限值，请检查信号与训练参数")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(targets)
        validation = evaluate(model, loaders["val"], device, len(labels))
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(datasets["train"]),
            "validation": validation,
        })
        print(
            f"{variant} seed={seed} epoch={epoch}/{args.epochs} "
            f"val_macro_f1={validation['macro_f1']:.4f}", flush=True,
        )
        if validation["macro_f1"] > best_score:
            best_score = validation["macro_f1"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    validation = evaluate(model, loaders["val"], device, len(labels))
    test = evaluate(model, loaders["test"], device, len(labels))
    metadata = {
        "config": config.to_dict(), "labels": labels, "scale": scale,
        "baseline_samples": args.baseline_samples, "seed": seed,
        "num_samples": len(datasets["train"][0][0]),
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "torch_version": str(torch.__version__),
        "arguments": vars(args),
    }
    torch.save({**metadata, "state_dict": best_state}, output / "best.pt")
    result = {**metadata, "validation": validation, "test": test, "history": history}
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="PG-DTAN 训练与消融实验")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample-rate", type=float, required=True)
    parser.add_argument("--baseline-samples", type=int, required=True)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="full")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-fft", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=64)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--fusion-size", type=int, default=64)
    parser.add_argument("--smoothing-kernel", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="runs/experiment")
    args = parser.parse_args()
    if min(args.baseline_samples, args.epochs, args.batch_size) < 1:
        parser.error("基线点数、训练轮数与 batch size 必须为正整数")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("学习率必须为正数，权重衰减不能为负数")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("随机种子不能重复")
    return args


def main():
    args = parse_args()
    records, labels = read_manifest(args.manifest)
    lengths = {len(load_voltage(record.path, args.baseline_samples)) for record in records}
    if len(lengths) != 1 or min(lengths) < args.n_fft:
        raise ValueError("所有样本必须等长，且长度至少为 n_fft；请按统一刺激时间窗切片")
    for split in ("val", "test"):
        available = {record.label for record in records if record.split == split}
        if available != set(labels):
            raise ValueError(f"{split} 必须覆盖所有类别，避免宏平均指标失真")
    scale = fit_scale(records, args.baseline_samples)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    output = Path(args.output)
    if output.exists():
        raise ValueError("输出目录已存在，请换一个目录以保留已有结果")
    output.mkdir(parents=True)
    snapshot = [
        {**record.__dict__, "path": str(record.path)} for record in records
    ]
    (output / "manifest_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    summary = {}
    for variant in variants:
        results = []
        for seed in args.seeds:
            destination = output / variant / f"seed_{seed}"
            destination.mkdir(parents=True)
            results.append(run(args, records, labels, scale, variant, seed, destination))
        summary[variant] = {}
        for metric in ("accuracy", "macro_f1"):
            values = [result["test"][metric] for result in results]
            summary[variant][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "values": values,
            }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
