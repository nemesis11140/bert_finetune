"""Streamlit 交互页面：输入中文 → 展示预测结果 + 各类别概率 + 推理耗时。

运行: uv run streamlit run scripts/app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from .config import Config, load_config
from .evaluate import _latest_model_dir
from .model import get_device, load_model, load_tokenizer, predict

# 用 cache_resource 缓存"模型加载"：整个应用只加载一次，避免每次交互都重新读盘
@st.cache_resource(show_spinner="正在加载模型...")
def _load_model(model_dir: str, num_labels: int):
    device = get_device()
    tokenizer = load_tokenizer(model_dir)
    model = load_model(model_dir, num_labels, device)
    return model, tokenizer, device


def _predict_with_latency(text, model, tokenizer, device, max_len):
    t0 = time.perf_counter()
    label, confidence, probs = predict(text, model, tokenizer, device, max_len)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return label, confidence, probs, elapsed_ms


def _list_models(models_dir: str) -> list[Path]:
    return sorted(Path(models_dir).glob("*"))


def _render_result(label, confidence, probs, elapsed_ms, device, model):
    st.markdown("---")
    left, right = st.columns(2)
    with left:
        emoji = "😄" if label == "正面" else "😞"
        color = "green" if label == "正面" else "red"
        st.markdown(f"### {emoji} **<span style='color:{color}'>{label}</span>**", unsafe_allow_html=True)
        st.markdown(f"置信度: **{confidence:.2%}**")
    with right:
        st.markdown(f"### ⏱️ **{elapsed_ms:.0f} ms**")
        st.caption(f"推理设备: {device}")

    st.markdown("各类别概率")
    cols = st.columns(2)
    for i, p in enumerate(probs):
        with cols[i]:
            st.markdown(f"**{model.config.id2label[i]}**  {p:.2%}")
            st.progress(p)


def main(config_path: str = "config.yaml") -> None:
    st.set_page_config(page_title="BERT 情感分类", page_icon="🎯", layout="centered")
    cfg: Config = load_config(config_path)

    st.title("🎯 BERT 中文情感分类")
    st.caption("输入一句中文，模型预测情感倾向并显示耗时（本地 bert-base-chinese 微调模型）")

    # 侧边栏：选择要用的模型（默认最新）
    model_dirs = _list_models(cfg.run.models_dir)
    if not model_dirs:
        st.error("models/ 目录为空，请先运行 `uv run python scripts/train.py` 训练模型")
        st.stop()

    st.sidebar.header("模型")
    model_dir = st.sidebar.selectbox(
        "选择模型",
        [str(p) for p in model_dirs],
        index=len(model_dirs) - 1,
        format_func=lambda p: Path(p).name,
    )
    model, tokenizer, device = _load_model(model_dir, cfg.model.num_labels)

    # 输入区
    text = st.text_input(
        "请输入要判断情感的文本",
        placeholder="例如：今天天气真好，心情特别愉快！",
        label_visibility="visible",
    )
    if st.button("预测", type="primary") and text.strip():
        label, confidence, probs, elapsed = _predict_with_latency(
            text.strip(), model, tokenizer, device, cfg.model.max_len
        )
        _render_result(label, confidence, probs, elapsed, device, model)


if __name__ == "__main__":
    main()
