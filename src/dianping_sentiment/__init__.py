"""大众点评客户评价情感分析。

中文用户评价的正负面判别：本地 bert-base-chinese 微调 + REST API + Web 页面。

模块划分（够用就好，不搞分层）：
- ``config``    读根目录 config.yaml
- ``data``      合成训练数据 + Dataset
- ``model``     模型加载 + 推理（训练评估和线上服务共用同一套推理代码）
- ``train``     训练 + run 存档 + 训练曲线
- ``evaluate``  批量评估 / 交互式预测
- ``service``   Flask 服务：REST API + 页面
"""

__version__ = "0.2.0"

# 标签映射：训练、评估、服务三处共用，避免同一个映射散落多份对不上
ID2LABEL: dict[int, str] = {0: "负面", 1: "正面"}
LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}
POSITIVE_ID = LABEL2ID["正面"]
NEGATIVE_ID = LABEL2ID["负面"]
