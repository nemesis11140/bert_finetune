"""评估模块：用已保存的模型跑批量测试集或交互式预测。

- 批量模式：读 data/test_sentences.json，计算 accuracy / f1 / 混淆矩阵，打印逐条结果和错例，
  并把结果存成 eval_results.json 放到模型目录下。
- 交互模式（--interactive）：命令行逐条输入，实时预测，替代原来的 test.py。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import Config, load_config
from .model import get_device, load_model, load_tokenizer, predict


def _latest_model_dir(models_dir: str) -> Path | None:
    """取 models/ 下最新的 run 目录（按目录名倒序）。"""
    candidates = sorted(Path(models_dir).glob("*"), key=lambda p: p.name)
    return candidates[-1] if candidates else None


def batch_evaluate(cfg: Config, model_dir: Path, device) -> None:
    model = load_model(str(model_dir), cfg.model.num_labels, device)
    tokenizer = load_tokenizer(str(model_dir))

    with open(cfg.data.test_sentences_path, encoding="utf-8") as f:
        items = json.load(f)

    print(f"测试集: {len(items)} 条 | 模型: {model_dir}")
    results = []
    for it in items:
        label, confidence, _ = predict(it["text"], model, tokenizer, device, cfg.model.max_len)
        gold = it.get("label")
        results.append({**it, "pred_label": label, "confidence": round(confidence, 4)})

    # 逐条打印
    print("\n逐条结果:")
    for r in results:
        mark = ""
        if r.get("label") is not None:
            mark = " ✅" if r["label"] == int(r["pred_label"] == "正面") else " ❌"
        print(f"  [{r.get('label', '?')}] → {r['pred_label']} ({r['confidence']:.1%})  {r['text']}{mark}")

    # 有 gold 标签时计算指标
    if all("label" in r for r in results):
        gold = np.array([r["label"] for r in results])
        pred = np.array([1 if r["pred_label"] == "正面" else 0 for r in results])
        acc = accuracy_score(gold, pred)
        f1 = f1_score(gold, pred, average="binary")
        cm = confusion_matrix(gold, pred)
        print("\n" + "=" * 40)
        print(f"准确率: {acc:.2%}")
        print(f"F1    : {f1:.2%}")
        print("混淆矩阵（行=真实, 列=预测）:")
        print(f"        预测负面  预测正面")
        print(f" 真实负面   {cm[0][0]:>5}     {cm[0][1]:>5}")
        print(f" 真实正面   {cm[1][0]:>5}     {cm[1][1]:>5}")
        errors = [r for r in results if r["label"] != (1 if r["pred_label"] == "正面" else 0)]
        print(f"\n错例 {len(errors)} 条:")
        for r in errors:
            print(f"  真实={r['label']} 预测={r['pred_label']} ({r['confidence']:.1%})  {r['text']}")
        print("=" * 40)

    # 存结果
    out_path = model_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存: {out_path}")


def interactive(cfg: Config, model_dir: Path, device) -> None:
    model = load_model(str(model_dir), cfg.model.num_labels, device)
    tokenizer = load_tokenizer(str(model_dir))
    print(f"🎯 模型已加载: {model_dir}\n请输入文本进行情感分类，输入 q 或 exit 退出\n")

    while True:
        text = input("请输入: ").strip()
        if text.lower() in ("q", "quit", "exit", "退出"):
            print("👋 再见")
            break
        if not text:
            continue
        label, confidence, probs = predict(text, model, tokenizer, device, cfg.model.max_len)
        print(f"  → 预测: {label} (置信度: {confidence:.2%})")
        for i, p in enumerate(probs):
            print(f"      P({model.config.id2label[i]}) = {p:.2%}")
        print()


def main(config_path: str = "config.yaml", model_dir: str | None = None, interactive_mode: bool = False) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(config_path)
    device = get_device()

    if model_dir is None:
        found = _latest_model_dir(cfg.run.models_dir)
        if found is None:
            print(f"❌ models/ 目录为空，请先用 scripts/train.py 训练，或 --model-dir 指定模型路径")
            sys.exit(1)
        model_dir = str(found)

    if interactive_mode:
        interactive(cfg, Path(model_dir), device)
    else:
        batch_evaluate(cfg, Path(model_dir), device)
