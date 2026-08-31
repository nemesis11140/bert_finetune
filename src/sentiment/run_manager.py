"""run 生命周期管理：创建带时间戳的存档目录、写 manifest、把控制台输出存成日志。

一次训练 = 一个 run，目录结构：
runs/<时间戳>/
├── manifest.json      # 超参数 + 数据信息 + 环境版本 + git commit
├── metrics.json       # 训练指标时序（loss 每步 / 验证指标每 epoch）
├── console.log        # 完整控制台输出（含进度条、警告、报错）
├── training_curve.png
└── checkpoints/       # 中间 checkpoint
"""
from __future__ import annotations

import contextlib
import datetime
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def create_run_dir(runs_dir: str | Path) -> tuple[Path, str]:
    """创建 runs/<时间戳>/ 目录，返回 (run_dir, run_name)。"""
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "n/a"


def _package_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("torch", "transformers", "scikit-learn", "matplotlib", "pyyaml"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "n/a"
    return versions


def save_manifest(run_dir: str | Path, cfg, extra: dict | None = None) -> Path:
    """把本次运行的配置、数据信息、环境信息写进 manifest.json，保证可复现。"""
    manifest = {
        "run_name": extra.get("run_name") if extra else None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "device": extra.get("device") if extra else None,
        "data": {
            "num_per_class": cfg.data.num_per_class,
            "val_ratio": cfg.data.val_ratio,
            "seed": cfg.data.seed,
            "n_train": extra.get("n_train") if extra else None,
            "n_val": extra.get("n_val") if extra else None,
        },
        "model": {
            "base_model": cfg.model.base_model,
            "num_labels": cfg.model.num_labels,
            "max_len": cfg.model.max_len,
        },
        "train": {
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "learning_rate": cfg.train.learning_rate,
            "weight_decay": cfg.train.weight_decay,
            "logging_steps": cfg.train.logging_steps,
            "seed": cfg.train.seed,
        },
    }
    path = Path(run_dir) / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def save_metrics(run_dir: str | Path, log_history: list[dict]) -> Path:
    """把训练指标时序（log_history）落盘成 metrics.json。"""
    path = Path(run_dir) / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, ensure_ascii=False, indent=2, default=float)
    return path


class _Tee:
    """同一份输出同时写到控制台和日志文件（模拟 shell 的 tee）。

    必须透传 isatty/fileno/encoding 等属性，否则库代码（tqdm、transformers 等）
    在检测"是否终端"或写进度条时会报错。
    """

    def __init__(self, stream, file):
        self.stream = stream
        self.file = file
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data: str):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()


@contextlib.contextmanager
def tee_console(log_path: str | Path):
    """把运行期间的 stdout/stderr 完整记录到 log_path，同时正常显示在终端。"""
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, log)
    sys.stderr = _Tee(orig_err, log)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = orig_out, orig_err
        log.close()
