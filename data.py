import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    path: Path
    label: str
    batch_id: str
    trial_id: str
    split: str


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label", "batch_id", "trial_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"清单必须包含字段：{sorted(required)}")
        rows = list(reader)
    records = []
    ownership = {"batch": {}, "trial": {}}
    seen_paths = set()
    for row in rows:
        if any(not row.get(key, "").strip() for key in required):
            raise ValueError("清单字段不能为空")
        split = row["split"].strip()
        if split not in {"train", "val", "test"}:
            raise ValueError("split 只能是 train、val、test")
        source = (path.parent / row["path"].strip()).resolve()
        if source in seen_paths:
            raise ValueError(f"同一数据文件重复出现：{source}")
        if not source.is_file():
            raise ValueError(f"数据文件不存在：{source}")
        seen_paths.add(source)
        record = Record(
            source, row["label"].strip(), row["batch_id"].strip(),
            row["trial_id"].strip(), split,
        )
        for name, identifier in (("batch", record.batch_id), ("trial", record.trial_id)):
            previous = ownership[name].setdefault(identifier, split)
            if previous != split:
                raise ValueError(f"数据泄漏：同一 {name} {identifier} 跨越不同集合")
        records.append(record)
    for split in ("train", "val", "test"):
        if not any(record.split == split for record in records):
            raise ValueError(f"缺少 {split} 数据")
    labels = sorted({record.label for record in records if record.split == "train"})
    if len(labels) < 2:
        raise ValueError("训练集至少需要两个类别")
    if any(record.label not in labels for record in records):
        raise ValueError("验证集或测试集存在训练集未包含的类别")
    return records, labels


def load_voltage(path, baseline_samples):
    voltage = np.load(path, allow_pickle=False)
    if voltage.ndim != 1 or not np.issubdtype(voltage.dtype, np.number):
        raise ValueError(f"数据必须为一维实数电压数组：{path}")
    if np.iscomplexobj(voltage) or len(voltage) <= baseline_samples:
        raise ValueError(f"数据应为实数，且长度大于基线采样点数：{path}")
    voltage = voltage.astype(np.float64)
    if not np.isfinite(voltage).all():
        raise ValueError(f"电压包含 NaN 或无穷值：{path}")
    return voltage - voltage[:baseline_samples].mean()


def fit_scale(records, baseline_samples):
    squared_sum = 0.0
    count = 0
    for record in records:
        if record.split == "train":
            voltage = load_voltage(record.path, baseline_samples)
            squared_sum += float(np.dot(voltage, voltage))
            count += voltage.size
    scale = (squared_sum / count) ** 0.5
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("训练集电压尺度无效或所有信号均为零")
    return scale


class VoltageDataset(Dataset):
    def __init__(self, records, labels, split, baseline_samples, scale):
        self.records = [record for record in records if record.split == split]
        self.label_indices = {label: index for index, label in enumerate(labels)}
        self.baseline_samples = baseline_samples
        self.scale = scale

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        voltage = load_voltage(record.path, self.baseline_samples) / self.scale
        return torch.tensor(voltage, dtype=torch.float32), self.label_indices[record.label]
