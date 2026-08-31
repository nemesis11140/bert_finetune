"""数据集：加载 JSON 数据并包装成 PyTorch Dataset。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def load_data(path: str | Path) -> tuple[list[str], list[int]]:
    """读取 [{text, label}, ...] 格式的 JSON，返回 (texts, labels)。"""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    texts = [item["text"] for item in items]
    labels = [item["label"] for item in items]
    return texts, labels


class SentimentDataset(Dataset):
    """把文本 + 标签包装成 Trainer 能用的数据集，__getitem__ 时做 tokenize。"""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "token_type_ids": encoding["token_type_ids"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
