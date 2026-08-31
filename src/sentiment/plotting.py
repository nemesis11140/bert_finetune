"""训练曲线绘图：把 metrics.json（log_history）画成 PNG。

中文字体处理：优先用系统可用的中文字体（SimHei / Microsoft YaHei 等），
找不到就提示但不报错，图片里中文可能显示为方框。
"""
from __future__ import annotations

import json
from pathlib import Path

# 经过 CVD 校验的参考调色板：系列色 1/2/3，文字用墨色，坐标轴用灰
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


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
            font_manager.findfont(
                font_manager.FontProperties(family=name), fallback_to_default=False
            )
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue

    steps = [h["step"] for h in log_history if "loss" in h]
    losses = [h["loss"] for h in log_history if "loss" in h]
    evals = [h for h in log_history if "eval_loss" in h]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)

    # 左图：训练 loss 随 step 变化
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.plot(steps, losses, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("step", color=MUTED)
    ax.set_ylabel("train loss", color=MUTED)
    ax.set_title("训练 Loss 曲线", color=INK)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)

    # 右图：每个 epoch 的验证集指标（画在同一轴上）
    ax = axes[1]
    ax.set_facecolor(SURFACE)
    if evals:
        epochs = [h.get("epoch") or i for i, h in enumerate(evals, start=1)]
        ax.plot(epochs, [h["eval_loss"] for h in evals], color=SERIES[0], linewidth=2,
                marker="o", markersize=4, label="eval loss")
        if "eval_accuracy" in evals[0]:
            ax.plot(epochs, [h["eval_accuracy"] for h in evals], color=SERIES[1], linewidth=2,
                    marker="s", markersize=4, label="accuracy")
        if "eval_f1" in evals[0]:
            ax.plot(epochs, [h["eval_f1"] for h in evals], color=SERIES[2], linewidth=2,
                    marker="^", markersize=4, label="f1")
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_ylabel("指标值", color=MUTED)
    ax.set_title("验证集指标", color=INK)
    if evals:
        ax.legend(frameon=False)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.6)

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(MUTED)

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
