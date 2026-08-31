"""训练模块：读 config.yaml，训练 BERT 情感分类模型，并完整存档这次运行。

产出物：
- runs/<时间戳>/          训练存档（manifest / metrics / console.log / 曲线 / checkpoints）
- models/<时间戳>/        最优模型（权重 + 分词器 + 当时用的 config.yaml）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from .config import Config, load_config
from .dataset import SentimentDataset, load_data
from .model import get_device, load_model, load_tokenizer
from .plotting import save_training_curve
from .run_manager import create_run_dir, save_manifest, save_metrics, tee_console


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
    model = load_model(cfg.model.base_model, cfg.model.num_labels, device)
    train_ds = SentimentDataset(train_texts, train_labels, tokenizer, cfg.model.max_len)
    val_ds = SentimentDataset(val_texts, val_labels, tokenizer, cfg.model.max_len)

    # 3. 训练
    steps_per_epoch = max(1, len(train_ds) // cfg.train.batch_size)
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,          # 训练结束自动切回 val 最优的权重
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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

    print(f"🚀 开始训练（每 {max(1, cfg.train.logging_steps)} 步打印一次 loss，共 {steps_per_epoch * cfg.train.epochs} 步）")
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
            print(f"  epoch {h.get('epoch') or 0:.1f}: loss={h['eval_loss']:.4f}  accuracy={acc_txt}  f1={f1_txt}")
        best = min(eval_entries, key=lambda h: h["eval_loss"])
        print(f"最优: epoch {best.get('epoch') or 0:.1f}, eval_loss={best['eval_loss']:.4f}")
    print("=" * 60)


def _print_test_samples(cfg: Config, model, tokenizer, device) -> None:
    from .model import predict

    print("\n🧪 模型效果抽查:")
    for text in ("今天天气真好，心情特别愉快！", "这家餐厅的服务态度太差了，不会再来了。",
                 "这部电影剧情精彩，演员演技也很棒。", "这个产品质量很差，用了三天就坏了。"):
        label, confidence, _ = predict(text, model, tokenizer, device, cfg.model.max_len)
        print(f"  {text}\n  → 预测: {label} (置信度: {confidence:.2%})")


def main(config_path: str = "config.yaml") -> None:
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

        # 产出：最优模型 → models/<run_name>/
        model_dir = Path(cfg.run.models_dir) / run_name
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        # 模型目录里附带一份当时用的配置，让模型"自描述"
        import shutil
        shutil.copy(config_path, model_dir / "config.yaml")
        print(f"✅ 最优模型已保存: {model_dir}")

        _print_summary(log_history)
        _print_test_samples(cfg, trainer.model, tokenizer, device)

    print(f"\n✅ 全部完成。存档: {run_dir}")
