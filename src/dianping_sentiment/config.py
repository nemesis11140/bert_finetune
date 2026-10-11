"""配置：读根目录的 config.yaml，解析成带类型校验的 dataclass。

训练和服务共用一份配置文件，服务段（``serving``）可以整个省略走默认值，
也可以用环境变量覆盖——容器部署时换个 ``DS_MODEL_DIR`` 就能切模型，不用重建镜像。
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

import yaml

from . import ID2LABEL

# 仓库根目录：src/dianping_sentiment/config.py → 上溯 3 层
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


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
class ServingConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    threads: int = 8        # WSGI 工作线程数
    model_dir: str = ""     # 留空 = 取 models_dir 下最新
    num_threads: int = 0    # torch 线程数，0 = 自动
    warmup: bool = True
    max_batch_size: int = 64
    max_text_length: int = 4096
    max_content_length: int = 1048576  # JSON 请求体上限（字节）
    log_level: str = "INFO"


@dataclass(frozen=True)
class Config:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    run: RunConfig
    serving: ServingConfig


def _check_types(section: str, obj: object, expected: dict[str, type]) -> None:
    """校验字段类型，防止 YAML 把数值解析成字符串（如把 2e-5 当成字符串）。"""
    for field, typ in expected.items():
        value = getattr(obj, field)
        if type(value) is not typ:
            raise ValueError(
                f"config.yaml 的 [{section}.{field}] 类型不对：期望 {typ.__name__}，"
                f"实际是 {type(value).__name__}（值: {value!r}）。"
                f"注意 YAML 里科学计数法要写 0.00002 或 2.0e-5，不要写 2e-5。"
            )


def _bool_env(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        raise ValueError(f"布尔环境变量值不合法: {value!r}")
    return normalized in ("1", "true", "yes", "on")


def _apply_env(cfg: ServingConfig) -> ServingConfig:
    """环境变量 > config.yaml > 默认值。"""
    env = os.environ
    for key, attr, cast in (
        ("DS_HOST", "host", str),
        ("DS_PORT", "port", int),
        ("DS_DEBUG", "debug", _bool_env),
        ("DS_MODEL_DIR", "model_dir", str),
        ("DS_NUM_THREADS", "num_threads", int),
        ("DS_LOG_LEVEL", "log_level", lambda v: v.upper()),
    ):
        if key in env:
            # Compose 未指定模型时会传空字符串，应保留 YAML 中的显式模型。
            if key == "DS_MODEL_DIR" and not env[key].strip():
                continue
            cfg = replace(cfg, **{attr: cast(env[key])})
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    """加载配置。path 为 None 时用仓库根目录的 config.yaml。"""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        data = DataConfig(**raw["data"])
        model = ModelConfig(**raw["model"])
        train = TrainConfig(**raw["train"])
        run = RunConfig(**raw["run"])
    except KeyError as exc:
        raise ValueError(f"{path} 缺少必需的配置段: {exc}") from None

    # serving 整段可省略，缺的项走默认值
    serving_raw = raw.get("serving") or {}
    unknown = set(serving_raw) - {f.name for f in fields(ServingConfig)}
    if unknown:
        raise ValueError(f"config.yaml 的 [serving] 里有不认识的配置项: {sorted(unknown)}")
    serving = _apply_env(ServingConfig(**serving_raw))

    _check_types("data", data, {"num_per_class": int, "val_ratio": float, "seed": int})
    _check_types("model", model, {"num_labels": int, "max_len": int})
    _check_types("train", train, {
        "batch_size": int, "epochs": int, "learning_rate": float,
        "weight_decay": float, "logging_steps": int, "seed": int,
    })
    if model.num_labels != len(ID2LABEL):
        raise ValueError(f"[model.num_labels] 只支持 {len(ID2LABEL)}，当前 {model.num_labels}")

    _check_types("serving", serving, {
        "host": str, "port": int, "debug": bool, "threads": int,
        "model_dir": str, "num_threads": int, "warmup": bool,
        "max_batch_size": int, "max_text_length": int,
        "max_content_length": int, "log_level": str,
    })
    for section, obj, names in (
        ("data", data, ("num_per_class",)),
        ("train", train, ("batch_size", "epochs", "logging_steps", "learning_rate")),
        ("serving", serving, ("threads", "max_batch_size", "max_text_length", "max_content_length")),
    ):
        for name in names:
            value = getattr(obj, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"[{section}.{name}] 必须大于 0 且是有限数值")
    if not 0 < data.val_ratio < 1:
        raise ValueError("[data.val_ratio] 必须在 0 和 1 之间")
    if not 2 <= model.max_len <= 512:
        raise ValueError("[model.max_len] 必须在 2 到 512 之间（BERT token 数）")
    if not math.isfinite(train.weight_decay) or train.weight_decay < 0:
        raise ValueError("[train.weight_decay] 必须是非负有限数值")
    if not 1 <= serving.port <= 65535 or serving.num_threads < 0:
        raise ValueError("serving.port 必须在 1 到 65535 之间，num_threads 必须非负")
    if serving.log_level.upper() not in logging.getLevelNamesMapping():
        raise ValueError(f"[serving.log_level] 不合法: {serving.log_level}")

    return Config(data=data, model=model, train=train, run=run, serving=serving)


def latest_model_dir(models_dir: str | Path) -> Path:
    """取模型根目录下最新的一个（时间戳目录名天然可排序）。

    只认含 config.json 的目录，models/ 下散落的日志/临时文件不会被误选。
    """
    models_root = Path(models_dir)
    candidates = sorted(p for p in models_root.glob("*")
                        if p.is_dir() and (p / "config.json").is_file())
    if not candidates:
        raise FileNotFoundError(
            f"{models_root} 下没有可用模型（需要含 config.json 的目录）。"
            f"请先训练（uv run python scripts/cli.py train），"
            f"或用 DS_MODEL_DIR 指定模型目录。"
        )
    return candidates[-1]


def resolve_model_dir(cfg: Config, root: str | Path = PROJECT_ROOT) -> Path:
    """算出服务要加载的模型目录：配置里写死了就用写死的，否则取最新的。

    上线时建议把 ``serving.model_dir`` 写死，避免多训了一版模型被悄悄切走。
    """
    root = Path(root)
    if cfg.serving.model_dir:
        path = Path(cfg.serving.model_dir)
        return path if path.is_absolute() else root / path
    return latest_model_dir(root / cfg.run.models_dir)
