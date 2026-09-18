"""评估模块：用已保存的模型跑批量测试集或交互式预测。

- 批量模式：读 data/test_sentences.json，计算 accuracy / f1 / 混淆矩阵，打印逐条结果和错例，
  并把结果存成 eval_results.json 放到模型目录下。
- 交互模式（--interactive）：命令行逐条输入，实时预测。

推理统一走 ``serving/predictor.py``——和线上服务是同一套代码，所以这里的指标
就是线上行为的忠实反映。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import ID2LABEL
from .config import Config, latest_model_dir, load_config
from .model import SentimentPredictor


def batch_evaluate(cfg: Config, model_dir: Path) -> None:
    predictor = SentimentPredictor(model_dir, max_len=cfg.model.max_len)
    print(f"模型: {model_dir} | 设备: {predictor.device}")

    with open(cfg.data.test_sentences_path, encoding="utf-8") as f:
        items = json.load(f)

    print(f"测试集: {len(items)} 条")
    # 一次性整批推理（和线上批量接口同一条路径），比逐条循环快得多
    result = predictor.predict([it["text"] for it in items])
    print(f"推理耗时: 整批 {result.latency_ms:.0f} ms（单条均摊 {result.per_item_ms:.1f} ms）\n")

    results = []
    for it, pred in zip(items, result.items):
        results.append({
            "text": it["text"],
            "label": it.get("label"),
            "pred_label": pred.label,
            "confidence": round(pred.confidence, 4),
        })

    # 逐条打印
    print("逐条结果:")
    for r in results:
        mark = ""
        if r["label"] is not None:
            mark = " ✅" if r["label"] == int(r["pred_label"] == ID2LABEL[1]) else " ❌"
        print(f"  [{r['label'] if r['label'] is not None else '?'}] → {r['pred_label']} "
              f"({r['confidence']:.1%})  {r['text']}{mark}")

    # 有 gold 标签时计算指标
    if all(r["label"] is not None for r in results):
        gold = np.array([r["label"] for r in results])
        pred = np.array([1 if r["pred_label"] == ID2LABEL[1] else 0 for r in results])
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
        errors = [r for r in results if r["label"] != (1 if r["pred_label"] == ID2LABEL[1] else 0)]
        print(f"\n错例 {len(errors)} 条:")
        for r in errors:
            print(f"  真实={r['label']} 预测={r['pred_label']} ({r['confidence']:.1%})  {r['text']}")
        print("=" * 40)

    # 存结果
    out_path = model_dir / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存: {out_path}")


def interactive(cfg: Config, model_dir: Path) -> None:
    predictor = SentimentPredictor(model_dir, max_len=cfg.model.max_len)
    print(f"🎯 模型已加载: {model_dir}（设备: {predictor.device}）")
    print("请输入评价文本，输入 q 或 exit 退出\n")

    while True:
        text = input("请输入: ").strip()
        if text.lower() in ("q", "quit", "exit", "退出"):
            print("👋 再见")
            break
        if not text:
            continue
        result = predictor.predict_one(text)
        pred = result.items[0]
        print(f"  → 预测: {pred.label} (置信度: {pred.confidence:.2%}, 耗时: {result.latency_ms:.1f} ms)")
        for name, p in pred.probs.items():
            print(f"      P({name}) = {p:.2%}")
        print()


def main(config_path=None, model_dir: str | None = None, interactive_mode: bool = False) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    cfg = load_config(config_path)

    if model_dir is None:
        try:
            found = latest_model_dir(cfg.run.models_dir)
        except FileNotFoundError as exc:
            print(f"❌ {exc}")
            sys.exit(1)
        model_dir = str(found)
        print(f"未指定 --model-dir，使用最新模型: {model_dir}")

    if interactive_mode:
        interactive(cfg, Path(model_dir))
    else:
        batch_evaluate(cfg, Path(model_dir))
