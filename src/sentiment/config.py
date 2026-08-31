"""配置加载：把 config.yaml 解析成带类型的 dataclass，方便 IDE 补全和取值校验。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    processed_path: str
    test_sentences_path: str
    num_per_class: int
    val_ratio: float
    seed: int


@dataclass(frozen=True)
class ModelConfig:
    base_model: str
    num_labels: int
    max_len: int


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    seed: int


@dataclass(frozen=True)
class RunConfig:
    runs_dir: str
    models_dir: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    run: RunConfig


def _check_types(section: str, obj: object, expected: dict[str, type]) -> None:
    """校验某个配置 section 里的字段类型，防止 YAML 把数值解析成字符串（如 2e-5）。"""
    for field, typ in expected.items():
        value = getattr(obj, field)
        if not isinstance(value, typ):
            raise ValueError(
                f"config.yaml 的 [{section}.{field}] 类型不对：期望 {typ.__name__}，"
                f"实际是 {type(value).__name__}（值: {value!r}）。"
                f"注意 YAML 里科学计数法要写 0.00002 或 2.0e-5，不要写 2e-5。"
            )


def load_config(path: str | Path) -> Config:
    """从 YAML 文件加载配置，并校验关键字段类型，配置写错会直接抛清晰异常。"""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    train = TrainConfig(**raw["train"])
    run = RunConfig(**raw["run"])

    _check_types("data", data, {"num_per_class": int, "val_ratio": float, "seed": int})
    _check_types("model", model, {"num_labels": int, "max_len": int})
    _check_types("train", train, {
        "batch_size": int, "epochs": int, "learning_rate": float,
        "weight_decay": float, "logging_steps": int, "seed": int,
    })
    return Config(data=data, model=model, train=train, run=run)
