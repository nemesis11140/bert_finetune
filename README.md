# bert-finetune

BERT 中文情感二分类微调（轻量级工程化项目）。数据由模板合成，模型用本地 `bert-base-chinese` 微调。

## 模型准备（必读）

本仓库**不包含任何模型权重**——预训练权重和训练产出都已在 `.gitignore` 中排除（体积太大）。运行前需要自己准备两个模型目录：

### 1. 预训练模型 `bert-base-chinese/`（需手动下载）

- 训练、评估、Web 页面首次运行时都从 `config.yaml → model.base_model` 指定的本地路径加载预训练权重。**缺少这个目录，脚本会直接报错，跑不起来。**
- 下载后放到**项目根目录** `bert-base-chinese/`（约 400MB），至少需要这几个文件：

  | 文件 | 作用 |
  |---|---|
  | `config.json` | 模型结构配置 |
  | `pytorch_model.bin` 或 `model.safetensors` | 预训练权重（任选其一） |
  | `vocab.txt` | 词表 |
  | `tokenizer_config.json` / `tokenizer.json` | 分词器配置 |

- 下载地址：[HuggingFace: bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)，或用命令行：

  ```bash
  uv run huggingface-cli download google-bert/bert-base-chinese --local-dir bert-base-chinese
  ```

### 2. 训练产出模型 `models/`（训练后自动生成）

- `scripts/train.py` 每次训练结束会把**最优模型**保存到 `models/<时间戳>/`（权重 + 分词器 + 当时用的 `config.yaml`），同样不进 git。
- 评估和 Web 页面默认从 `models/` 读取：`scripts/evaluate.py --model-dir models/<时间戳>`，Streamlit 侧边栏可选择 `models/` 下的任一模型。
- 仓库已带一个从旧版迁移过来的模型 `models/20260827-legacy`，想直接体验推理可跳过训练，用它即可。

## 项目架构

```
bert-finetune/
├── config.yaml                  # 唯一配置入口：数据/模型/训练/输出，改参数不用动代码
├── data/
│   ├── processed/               # 生成的训练数据（gitignore，可再生成）
│   └── test_sentences.json      # 评估用测试集（带 gold 标签，入 git）
├── src/sentiment/               # 核心逻辑包
│   ├── config.py                # 从 config.yaml 加载配置（dataclass 校验）
│   ├── data_generation.py       # 模板合成数据（句式/词库都在这里扩充）
│   ├── dataset.py               # 数据集类 + 数据加载
│   ├── model.py                 # 模型/分词器加载 + 单条预测 + 标签映射
│   ├── run_manager.py           # run 生命周期：存档目录/manifest/控制台日志
│   ├── plotting.py              # 训练曲线绘图（含中文字体处理）
│   ├── train.py                 # 训练 + 每 run 完整存档 + 产出最优模型
│   └── evaluate.py              # 批量评估（指标/混淆/错例）+ 交互式预测
├── scripts/                     # 命令行入口（薄封装，含 src 路径引导）
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── app.py                   # Web 交互页面（Streamlit）
├── src/sentiment/webapp.py      # Web 页面逻辑：输入 → 预测结果 + 概率 + 耗时
├── runs/                        # 每次训练一个存档目录（gitignore）
│   └── <时间戳>/
│       ├── manifest.json        # 超参数 + 数据信息 + 环境版本 + git commit
│       ├── metrics.json         # 训练指标时序（loss 每步 / 验证指标每 epoch）
│       ├── console.log          # 完整控制台输出（进度条/警告/报错）
│       ├── training_curve.png
│       └── checkpoints/         # 中间 checkpoint
├── models/                      # 正式产出的最优模型（gitignore）
│   └── <时间戳>/                # 权重 + 分词器 + 当时用的 config.yaml
└── tests/                       # 轻量冒烟测试（无需 pytest）
```

## 常用命令

```bash
# 1. 生成训练数据（数量由 config.yaml 的 data.num_per_class 控制）
uv run python scripts/generate_data.py

# 2. 训练（每次运行自动在 runs/ 存档、在 models/ 产出最优模型）
uv run python scripts/train.py

# 3. 评估最新模型（跑 data/test_sentences.json，算 acc/f1/混淆矩阵/错例）
uv run python scripts/evaluate.py

# 指定模型评估 / 交互式预测
uv run python scripts/evaluate.py --model-dir models/<时间戳>
uv run python scripts/evaluate.py --interactive

# 重绘某个 run 的曲线（改样式后不用重新训练）
uv run python scripts/train.py --replot runs/<时间戳>

# 跑冒烟测试
uv run python tests/test_data_generation.py

# Web 交互页面（浏览器打开 http://localhost:8501）
uv run streamlit run scripts/app.py
```

## Web 交互页面

`uv run streamlit run scripts/app.py` 启动一个本地页面：输入一句中文 → 点击「预测」→ 展示情感标签、置信度、各类别概率和推理耗时（毫秒）。

- 侧边栏可选择用 `models/` 下的哪个模型，默认取最新训练的那个。
- 模型首次加载会缓存，之后每次推理都是纯推理耗时（CPU 上单条约 100–250ms）。
- 当前 `models/20260827-legacy` 是从旧版训练结果迁移过来的真实模型，可直接用。

## 设计要点

- **一切参数进 `config.yaml`**：超参数、数据量、输出目录都集中管理，训练时把当时配置记入 `manifest.json`，保证可复现。
- **一次训练 = 一个 run**：`runs/<时间戳>/` 完整存档（超参/指标/控制台日志/曲线），历史 run 不会被覆盖；最优模型单独落到 `models/<时间戳>/`。
- **代码分模块**：数据生成、数据集、模型、训练、评估各自独立，改一处不影响其他。
- **best checkpoint**：`load_best_model_at_end` 自动按 val loss 挑最优权重，保存的不再是"最后一步"。

## 数据说明

当前训练数据由模板合成，特点是量大、标签干净、正负均衡，但句式单一。因此**验证集/测试集高分不代表真实场景泛化**——`data/test_sentences.json` 里专门混了否定、口语、反讽类句子用于摸底。想要更真实的效果，应接入真实语料（微博/电商评论等）。
