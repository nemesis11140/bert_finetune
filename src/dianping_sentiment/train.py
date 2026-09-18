"""训练：微调 BERT 情感分类模型，并完整存档这次运行。

一次训练 = 一个 run：
  runs/<时间戳>/     manifest.json（配置+环境）、metrics.json、console.log、曲线
  models/<时间戳>/   模型（权重 + 分词器 + 当时用的 config.yaml 副本）

**不落 checkpoint**（``save_strategy="no"``）。checkpoint 的用途是训练中断后续训，
但这里单次训练只要两分多钟，重跑一遍比留着断点便宜；而断点每个都带 optimizer
状态，3 个就是 3.5G，堆在项目目录里纯属占地方。代价是不能断点续训，对 demo 无所谓。

连带地也没有 ``load_best_model_at_end``（它靠磁盘断点还原最优轮），最终保存的是
**最后一轮**的权重。想改回"存 val 最优"，把 ``save_strategy`` 设成 ``"epoch"``
并加回 ``load_best_model_at_end=True`` 即可。

文件后半部分是这个流程自己用到的辅助代码：run 目录与存档、控制台日志 tee、训练曲线。
它们只服务于训练，没必要单独成模块。
"""
from __future__ import annotations

import contextlib
import datetime
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .config import DEFAULT_CONFIG, Config, load_config
from .data import SentimentDataset, load_data
from .model import get_device, load_for_training, load_tokenizer, run_inference


# ==================== 训练 ====================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
    }


def _train(cfg: Config, run_dir: Path, device):
    """核心训练流程，返回训练产物供存档。"""
    # 1. 数据加载与切分
    texts, labels = load_data(cfg.data.processed_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=cfg.data.val_ratio,
        random_state=cfg.data.seed,
        stratify=labels,
    )
    print(f"数据: 共 {len(texts)} 条 | 训练 {len(train_texts)} | 验证 {len(val_texts)}")

    # 2. 模型与数据集
    tokenizer = load_tokenizer(cfg.model.base_model)
    model = load_for_training(cfg.model.base_model, cfg.model.num_labels, device)
    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, cfg.model.max_len)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, cfg.model.max_len)

    # 3. 训练
    steps_per_epoch = max(1, len(train_ds) // cfg.train.batch_size)
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        eval_strategy="epoch",     # 每轮评一次验证集，指标进 metrics.json 和曲线图
        save_strategy="no",        # 不落 checkpoint，理由见模块 docstring
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        num_train_epochs=cfg.train.epochs,
        weight_decay=cfg.train.weight_decay,
        logging_steps=max(1, cfg.train.logging_steps),
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"🚀 开始训练（每 {max(1, cfg.train.logging_steps)} 步打印一次 loss，"
          f"共 {steps_per_epoch * cfg.train.epochs} 步）")
    trainer.train()

    return trainer, tokenizer, trainer.state.log_history, len(train_texts), len(val_texts)


def _print_summary(log_history: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("📈 训练过程总结")
    print("=" * 60)
    train_losses = [h["loss"] for h in log_history if "loss" in h]
    if train_losses:
        print(f"训练 loss 共 {len(train_losses)} 个采样点：{train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    eval_entries = [h for h in log_history if "eval_loss" in h]
    if eval_entries:
        print("\n验证集指标（每 epoch）:")
        for h in eval_entries:
            acc = h.get("eval_accuracy")
            f1 = h.get("eval_f1")
            acc_txt = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
            f1_txt = f"{f1:.4f}" if isinstance(f1, (int, float)) else "-"
            print(f"  epoch {h.get('epoch') or 0:.1f}: loss={h['eval_loss']:.4f}  "
                  f"accuracy={acc_txt}  f1={f1_txt}")
        best = min(eval_entries, key=lambda h: h["eval_loss"])
        print(f"最优: epoch {best.get('epoch') or 0:.1f}, eval_loss={best['eval_loss']:.4f}")
    print("=" * 60)


def _print_test_samples(cfg: Config, model, tokenizer, device) -> None:
    """训练结束后的现场抽查。

    走 ``model.run_inference`` 而不是另写一份推理：抽查看到的行为就是上线后的行为
    （同一套动态 padding / inference_mode 路径）。
    """
    samples = (
        "今天天气真好，心情特别愉快！",
        "这家餐厅的服务态度太差了，不会再来了。",
        "这电影真好看，看得我昏昏欲睡",
        "服务态度真是太好了，好到让我想投诉",
    )
    result = run_inference(model, tokenizer, list(samples), device, cfg.model.max_len)
    print(f"\n🧪 模型效果抽查（整批 {result.latency_ms:.0f} ms，单条均摊 {result.per_item_ms:.1f} ms）:")
    for text, pred in zip(samples, result.items):
        print(f"  {text}\n  → 预测: {pred.label} (置信度: {pred.confidence:.2%})")


# ==================== run 存档 ====================

def create_run_dir(runs_dir: str | Path) -> tuple[Path, str]:
    """创建 runs/<时间戳>/ 目录，返回 (run_dir, run_name)。"""
    run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
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


def save_manifest(run_dir: str | Path, cfg: Config, extra: dict) -> Path:
    """把本次运行的配置、数据信息、环境信息写进 manifest.json，保证可复现。"""
    manifest = {
        "run_name": extra.get("run_name"),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "device": extra.get("device"),
        "data": {
            "num_per_class": cfg.data.num_per_class,
            "val_ratio": cfg.data.val_ratio,
            "seed": cfg.data.seed,
            "n_train": extra.get("n_train"),
            "n_val": extra.get("n_val"),
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

    必须透传 isatty/fileno/encoding，否则库代码（tqdm、transformers 等）
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


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _flatten_progress(path: str | Path) -> None:
    """把进度条的控制序列压平，让 console.log 能用普通编辑器打开。

    tqdm 靠 ``\\r`` 回车 + ``ESC[A`` 上移光标在终端里**原地重画**进度条，这套控制
    序列原样落盘就没法看了（实测一次训练 507 个 \\r、70 个 ESC），编辑器打开满屏乱码。
    再加上 Windows 文本模式会把 ``\\n`` 写成 ``\\r\\n``，两种 ``\\r`` 还混在一起。

    处理顺序：剥掉 ANSI 转义 → 归一化 CRLF → 每一行只保留最后一次重画的结果
    （也就是进度条的最终状态）→ 收掉重画留下的行尾空白和连续空行。
    """
    p = Path(path)
    # 必须显式写 newline=""：文本模式默认会把 \r 也翻译成 \n，那样进度条的重画
    # 结构就被破坏了，压平逻辑失效（read_text() 在 3.12 上没法关掉这个翻译）。
    with open(p, encoding="utf-8", errors="replace", newline="") as f:
        text = f.read()
    text = _ANSI_RE.sub("", text)
    text = text.replace("\r\n", "\n")
    text = "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)


@contextlib.contextmanager
def tee_console(log_path: str | Path):
    """把运行期间的 stdout/stderr 完整记录到 log_path，同时正常显示在终端。

    收尾时会把 tqdm 的控制序列压平（见 ``_flatten_progress``），日志才读得下去。
    """
    # newline="" 让 \n 原样落盘，不在 Windows 上被悄悄改写成 \r\n
    log = open(log_path, "w", encoding="utf-8", errors="replace", newline="")
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
        # 压平只是让日志好读，失败了也不该影响训练结果（模型这时已经存好了）
        try:
            _flatten_progress(log_path)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ 日志压平失败（不影响训练）: {exc}")


# ==================== 训练曲线 ====================

# 参考调色板：系列色 + 墨色文字 + 灰色坐标轴
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
_SURFACE, _INK, _MUTED, _GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"


def save_training_curve(log_history: list[dict], save_path: str | Path) -> bool:
    """画训练 loss + 验证指标双图，存为 PNG。未装 matplotlib 时返回 False。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        return False

    # 中文字体：按可用性自动选择，避免缺字方框
    for name in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    steps = [h["step"] for h in log_history if "loss" in h]
    losses = [h["loss"] for h in log_history if "loss" in h]
    evals = [h for h in log_history if "eval_loss" in h]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(_SURFACE)

    # 左图：训练 loss 随 step 变化
    ax = axes[0]
    ax.set_facecolor(_SURFACE)
    ax.plot(steps, losses, color=_SERIES[0], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("step", color=_MUTED)
    ax.set_ylabel("train loss", color=_MUTED)
    ax.set_title("训练 Loss 曲线", color=_INK)
    ax.tick_params(colors=_MUTED)
    ax.grid(color=_GRID, linewidth=0.6)

    # 右图：每个 epoch 的验证集指标
    ax = axes[1]
    ax.set_facecolor(_SURFACE)
    if evals:
        epochs = [h.get("epoch") or i for i, h in enumerate(evals, start=1)]
        ax.plot(epochs, [h["eval_loss"] for h in evals], color=_SERIES[0],
                linewidth=2, marker="o", markersize=4, label="eval loss")
        if "eval_accuracy" in evals[0]:
            ax.plot(epochs, [h["eval_accuracy"] for h in evals], color=_SERIES[1],
                    linewidth=2, marker="s", markersize=4, label="accuracy")
        if "eval_f1" in evals[0]:
            ax.plot(epochs, [h["eval_f1"] for h in evals], color=_SERIES[2],
                    linewidth=2, marker="^", markersize=4, label="f1")
    ax.set_xlabel("epoch", color=_MUTED)
    ax.set_ylabel("指标值", color=_MUTED)
    ax.set_title("验证集指标", color=_INK)
    if evals:
        ax.legend(frameon=False)
    ax.tick_params(colors=_MUTED)
    ax.grid(color=_GRID, linewidth=0.6)

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(_MUTED)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def replot_from_metrics(metrics_path: str | Path) -> bool:
    """从已存档的 metrics.json 重新画曲线（改样式后无需重训）。"""
    metrics_path = Path(metrics_path)
    with open(metrics_path, encoding="utf-8") as f:
        log_history = json.load(f)
    out_png = metrics_path.parent / "training_curve.png"
    ok = save_training_curve(log_history, out_png)
    print(f"{'✅' if ok else '❌'} 已{'重新生成' if ok else '未生成'}曲线图: {out_png}")
    return ok


# ==================== 入口 ====================

def main(config_path: str | Path | None = None) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(config_path)
    device = get_device()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    print(f"使用设备: {device}")

    run_dir, run_name = create_run_dir(cfg.run.runs_dir)
    print(f"📂 本次运行目录: {run_dir}")

    with tee_console(run_dir / "console.log"):
        trainer, tokenizer, log_history, n_train, n_val = _train(cfg, run_dir, device)

        # 存档：manifest / metrics / 曲线图
        save_manifest(run_dir, cfg, extra={
            "run_name": run_name, "device": str(device), "n_train": n_train, "n_val": n_val,
        })
        save_metrics(run_dir, log_history)
        curve_path = run_dir / "training_curve.png"
        if save_training_curve(log_history, curve_path):
            print(f"📊 训练曲线图: {curve_path}")
        else:
            print("📊 未安装 matplotlib，跳过曲线图")

        # 产出：模型 → models/<run_name>/
        model_dir = Path(cfg.run.models_dir) / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        # 模型目录里附带一份当时用的配置，让模型"自描述"
        shutil.copy(config_path or DEFAULT_CONFIG, model_dir / "train.yaml")
        print(f"✅ 模型已保存: {model_dir}")

        _print_summary(log_history)
        _print_test_samples(cfg, trainer.model, tokenizer, device)

    print(f"\n✅ 全部完成。存档: {run_dir}")
