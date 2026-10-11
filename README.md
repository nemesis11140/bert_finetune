# 大众点评客户评价情感分析

基于中文 BERT 微调的用户评价情感分析系统，支持单条与批量评价的正负面判别，提供 Web 页面和 REST API，可通过 Docker 部署。

## 业务功能

- 单条分析：输出情感标签、置信度及两类概率。
- 批量分析：按输入顺序返回结果，汇总正负面数量和占比。
- 模型管理：训练产物按运行时间保存，部署时通过配置指定模型版本。

模型为单标签二分类：`0=负面`、`1=正面`，不包含中性标签。默认输入截断为 128 个 token；置信度是模型输出得分，不代表实际准确率。

## 项目架构

技术栈：Python 3.12、PyTorch CPU、Transformers、Flask、Waitress、原生 JavaScript。

```text
bert_finetune/
├── src/dianping_sentiment/
│   ├── data.py                  # 数据生成、校验与 Dataset
│   ├── train.py                 # 模型训练、指标与运行存档
│   ├── evaluate.py              # 模型评估与错例分析
│   ├── model.py                 # 模型加载与统一推理
│   ├── service.py               # Web 页面、API 与请求校验
│   ├── config.py                # 配置加载与模型版本解析
│   ├── templates/              # 页面模板
│   └── static/                 # JavaScript / CSS
├── scripts/cli.py              # 生成、训练、评估、服务命令入口
├── config.yaml                 # 项目配置
├── pyproject.toml / uv.lock     # 依赖声明与锁文件
├── Dockerfile                  # 镜像构建
├── docker-compose.yml          # 本地构建与运行
├── docker-compose.deploy.yml   # 云端使用已有镜像
├── .env.example                # 部署环境变量示例
├── data/                       # 训练数据与测试集
├── tests/                      # 自动化测试
├── bert-base-chinese/          # 预训练权重，仅训练需要
├── models/<运行名>/             # 训练模型及分词器
└── runs/<运行名>/               # 配置、日志、指标与训练曲线
```

训练流程：数据准备 → 分层切分 → BERT 微调 → 保存模型及运行记录。

服务流程：页面或 API 请求 → 参数校验 → 分词与批量推理 → 返回结果。模型启动时加载并预热，训练后评估和线上服务共用推理代码；推理在服务部署机器执行。

## 本地运行

安装 Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在项目根目录执行：

```bash
uv sync --locked

# 首次训练前下载预训练权重
uv run hf download google-bert/bert-base-chinese config.json model.safetensors vocab.txt tokenizer.json tokenizer_config.json --local-dir bert-base-chinese

uv run python scripts/cli.py generate
uv run python scripts/cli.py train
uv run python scripts/cli.py evaluate
uv run python scripts/cli.py serve
```

访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)。已有训练模型时可直接启动服务，无需下载预训练权重或重新训练。训练保存最后一轮模型，不保存中间 checkpoint。

## 配置

`config.yaml` 管理数据、模型、训练和服务参数。服务支持 `DS_HOST`、`DS_PORT`、`DS_DEBUG`、`DS_MODEL_DIR`、`DS_NUM_THREADS`、`DS_LOG_LEVEL` 环境变量覆盖。

- 模型选择：优先使用非空 `DS_MODEL_DIR`，其次使用 `serving.model_dir`，均未指定时按目录名选择最新模型；切换模型后需重启或重新创建容器。
- 输入限制：默认每批 64 条、每条 4096 字符、请求体 1 MiB，可在 `serving` 中调整。
- 部署参数：Compose 通过同目录 `.env` 读取镜像名、模型目录、宿主端口和绑定地址，参考 `.env.example`。

## Docker 部署

### 本地构建

准备好 `models/` 下的训练模型，复制 `.env.example` 为 `.env` 并填写实际模型目录：

```bash
docker compose build api
docker compose up -d --no-build api
docker compose ps
docker compose logs --tail 50 api
```

镜像包含代码、配置和 Python 依赖；模型不进入镜像，通过只读卷挂载。API 使用非 root 用户运行。

### 云服务器部署

服务器安装 Docker 和 Compose，拉取或导入已发布镜像，并准备以下文件：

```text
部署目录/
├── docker-compose.deploy.yml
├── config.yaml
├── .env
└── models/20260917-192434/       # 完整模型目录：权重、标签配置及分词器
```

`.env` 示例，将镜像名和模型版本替换为实际值：

```dotenv
DS_IMAGE=registry.example.com/your-project/dianping-sentiment:版本
DS_MODEL_DIR=/app/models/20260917-192434
DS_PORT=8000
DS_BIND_ADDRESS=127.0.0.1
```

在部署目录执行：

```bash
docker compose -f docker-compose.deploy.yml pull api
docker compose -f docker-compose.deploy.yml up -d --no-build api
docker compose -f docker-compose.deploy.yml ps
curl http://127.0.0.1:8000/api/health
```

通过 `docker load` 导入镜像时跳过 `pull`。服务器不需要源码、Python、uv 或 `bert-base-chinese/`；挂载的模型和配置须允许容器用户读取。

默认仅本机可访问，适合反向代理转发；需要外部直连时设 `DS_BIND_ADDRESS=0.0.0.0`，并放行对应端口。对外提供服务时应配置 HTTPS、认证及限流。

### 日常维护

云端命令统一使用 `docker compose -f docker-compose.deploy.yml`：

```bash
docker compose -f docker-compose.deploy.yml stop api
docker compose -f docker-compose.deploy.yml start api
docker compose -f docker-compose.deploy.yml logs -f api
docker compose -f docker-compose.deploy.yml restart api
```

修改挂载的 `config.yaml` 后重启即可；修改 `.env` 或镜像版本后执行 `up -d --no-build`，并通过 `/api/meta` 确认生效版本。代码或依赖变化需要重建镜像，模型更新无需重建镜像。健康状态 `healthy` 表示存活探针通过。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | Web 页面 |
| GET | `/api/health` | 存活检查 |
| GET | `/api/meta` | 模型版本、设备、标签与输入限制 |
| POST | `/api/predict` | 单条预测：`{"text":"菜很好吃"}` |
| POST | `/api/predict_batch` | 批量预测：`{"texts":["菜很好吃","服务太差"]}` |

单条结果包含 `label`、`label_id`、`confidence`、`probs` 和 `latency_ms`；批量结果另包含逐条结果、情感分布与单条均摊耗时。批量输入也支持换行字符串，空行会被过滤。无效输入返回 400，请求体超限返回 413。

## 测试与效果

```bash
uv run python tests/test_all.py
uv run python scripts/cli.py evaluate --model-dir models/20260917-192434
```

测试覆盖标签映射、数据与配置校验、接口、训练及模型推理；缺少训练模型时真实推理用例会跳过。

当前训练数据由模板合成，默认正负各 200 条。已有模型在 22 条人工标注测试句上准确率为 95.45%、正类 F1 为 94.74%；该结果仅用于小样本回归验证，真实业务效果需用独立领域数据评估。
