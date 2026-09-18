"""模型加载与推理。

这是**唯一**的推理实现——离线评估（``evaluate.py``）、训练后的抽查（``train.py``）
和线上服务（``service.py``）都走这里，所以评估指标就是线上行为的忠实反映。

推理路径上的几个提速点（实测单条 119ms → 42ms，批量均摊 121ms → 21ms）：
1. **动态 padding**：不再把每条都补齐到 ``max_len``（128）。一句 20 字的评价实际只有
   20 来个 token，原来白算 6 倍算力。改成按 batch 内最长补齐，这是提速的主因。
2. **``torch.inference_mode()``** 替代 ``no_grad()``，少一层 autograd 记账。
3. **批量内按长度排序**：长短句混在一起时减少 padding 浪费，返回前还原顺序。
4. **启动预热**：把首次前向的冷启动开销挪到服务启动阶段。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from . import ID2LABEL

# 实测（12 逻辑核，单条 22 token）：1 线程 72ms / 2 线程 47ms / 4 线程 43ms / 8 线程 36ms。
# 8 线程往后收益很小（内存带宽瓶颈），所以自动档取 min(8, 核数)，
# 避免在核多的机器上一次推理就抢满整机 CPU。
_MAX_AUTO_THREADS = 8


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_num_threads(num_threads: int = 0) -> int:
    """把配置里的 torch 线程数解析成实际值；0 表示自动。"""
    if num_threads and num_threads > 0:
        return num_threads
    return max(1, min(_MAX_AUTO_THREADS, os.cpu_count() or 1))


def _require_local_dir(path: str | Path, what: str) -> str:
    """确认路径是个真实存在的目录，挡住 transformers 的静默联网下载。

    给 ``from_pretrained`` 传一个不存在的路径它不报错，而是把这串字符当成
    HuggingFace 上的 repo id 去下载——下到全局缓存 ``~/.cache/huggingface``，
    不是当前目录。本地目录一改名或删掉，就会变成"没人让它下、它自己下了整仓 1.6G"。
    所以在调用前先卡一道。
    """
    if not Path(path).is_dir():
        raise FileNotFoundError(
            f"找不到{what}: {path}\n"
            f"预训练权重见 README「准备预训练模型」；"
            f"训练好的模型请确认目录存在（或用 --model-dir / DS_MODEL_DIR 指定）。"
        )
    return str(path)


def load_tokenizer(path: str | Path) -> BertTokenizer:
    return BertTokenizer.from_pretrained(_require_local_dir(path, "分词器目录"))


def load_for_training(base_model: str, num_labels: int, device: torch.device):
    """从预训练权重造一个待微调的模型（分类头随机初始化）。

    把 id2label/label2id 写进 config，训练产物保存后自带标签语义。
    """
    return BertForSequenceClassification.from_pretrained(
        _require_local_dir(base_model, "预训练权重目录"),
        num_labels=num_labels, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()}
    ).to(device)


@dataclass(frozen=True)
class Prediction:
    """单条预测结果。"""

    label: str                  # "正面" / "负面"
    label_id: int
    confidence: float           # 被选中类别的概率
    probs: dict[str, float]     # 各类别概率：{"负面": .., "正面": ..}

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "confidence": round(self.confidence, 6),
            "probs": {k: round(v, 6) for k, v in self.probs.items()},
        }


@dataclass
class InferenceResult:
    """一次推理调用的结果。

    ``latency_ms`` 是整批的墙钟耗时（含分词）。批量场景下要看 ``per_item_ms``——
    那才是分摊后的单条成本，别把整批耗时当成"一条要多少毫秒"。
    """

    items: list[Prediction] = field(default_factory=list)
    latency_ms: float = 0.0
    batch_size: int = 0

    @property
    def per_item_ms(self) -> float:
        return self.latency_ms / self.batch_size if self.batch_size else 0.0


@torch.inference_mode()
def run_inference(model, tokenizer, texts: list[str], device: torch.device,
                  max_len: int = 128) -> InferenceResult:
    """对任意已加载的模型做一次批量推理。

    抽成独立函数是为了让"训练结束后拿内存里的模型抽查几句"也走这条路径
    （``train.py`` 就是这么用的），不用再维护一份只用于抽查的推理实现。
    """
    texts = [t for t in texts]
    if not texts:
        return InferenceResult(items=[], latency_ms=0.0, batch_size=0)

    t0 = time.perf_counter()

    # 长度排序 → 短句聚在一起，减少长句把整批 padding 撑大的浪费
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    encoded = tokenizer(
        [texts[i] for i in order],
        truncation=True,
        max_length=max_len,
        padding=True,            # 动态 padding：补齐到本批最长，而不是每条都补到 max_len
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    probs = torch.softmax(model(**encoded).logits, dim=-1).cpu()

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 还原调用方传入的顺序
    items: list[Prediction | None] = [None] * len(texts)
    for pos, original_index in enumerate(order):
        row = probs[pos]
        label_id = int(torch.argmax(row))
        items[original_index] = Prediction(
            label=ID2LABEL[label_id],
            label_id=label_id,
            confidence=float(row[label_id]),
            probs={ID2LABEL[i]: float(row[i]) for i in range(len(row))},
        )

    return InferenceResult(items=[x for x in items if x is not None],
                           latency_ms=elapsed_ms, batch_size=len(texts))


class SentimentPredictor:
    """加载一个微调好的模型，提供单条 / 批量推理。

    构造后模型常驻内存，``predict`` 系列方法不碰磁盘、不加锁，可被多个请求线程复用
    （PyTorch 前向是只读的）。
    """

    def __init__(self, model_dir: str | Path, max_len: int = 128,
                 num_threads: int = 0, device: torch.device | None = None) -> None:
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"{self.model_dir} 下没有 config.json，不是有效的模型目录")

        torch.set_num_threads(resolve_num_threads(num_threads))
        self.num_threads = torch.get_num_threads()
        self.device = device or get_device()
        self.max_len = max_len

        self.tokenizer = load_tokenizer(self.model_dir)
        self.model = BertForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str]) -> InferenceResult:
        """批量预测，返回顺序与传入的 texts 一致。"""
        return run_inference(self.model, self.tokenizer, texts, self.device, self.max_len)

    def predict_one(self, text: str) -> InferenceResult:
        """单条预测。返回 InferenceResult 是为了和批量共用一套响应结构。"""
        return self.predict([text])

    def warmup(self) -> float:
        """空跑一次，把首次前向的冷启动开销挪到服务启动阶段。"""
        t0 = time.perf_counter()
        self.predict(["预热"])
        return (time.perf_counter() - t0) * 1000

    def info(self) -> dict:
        """给 /api/meta 和启动日志用的自描述信息。"""
        return {
            "model_dir": str(self.model_dir),
            "model_name": self.model_dir.name,
            "device": str(self.device),
            "max_len": self.max_len,
            "num_threads": self.num_threads,
        }
