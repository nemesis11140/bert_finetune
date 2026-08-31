"""模型与分词器加载，标签映射统一收口在这里。"""
from __future__ import annotations

import torch
from transformers import BertForSequenceClassification, BertTokenizer

# 标签映射：所有地方都从这儿取，避免散落各处不一致
ID2LABEL = {0: "负面", 1: "正面"}
LABEL2ID = {"负面": 0, "正面": 1}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(base_model: str) -> BertTokenizer:
    return BertTokenizer.from_pretrained(base_model)


def load_model(base_model: str, num_labels: int, device: torch.device) -> BertForSequenceClassification:
    """加载预训练模型 + 随机初始化分类头。"""
    return BertForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)


def predict(text: str, model, tokenizer, device: torch.device, max_len: int = 128):
    """单条文本预测，返回 (label, confidence, probs)。"""
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = torch.argmax(probs).item()
    return ID2LABEL[pred_id], probs[pred_id].item(), probs.tolist()
