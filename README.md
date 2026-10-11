# 中文评价情感二分类 Demo

基于 `bert-base-chinese` 微调的中文情感分类项目，提供数据生成、训练、评估、REST API 和 Web 页面，演示从模型开发到 Docker 部署的完整流程。应用场景以餐饮点评为例，当前语料也包含商品、电影等通用评价。

项目使用模板合成数据，未使用真实大众点评业务数据，也不代表与大众点评平台有任何关联。适合学习、演示和内部试用；真实业务接入前需要补充领域数据、重新评估效果及部署容量。

## 任务定义与能力边界

本项目是**单标签二分类**，每条文本只输出一个标签：

| 标签 ID | 标签 | 含义 |
|---|---|---|
| `0` | 负面 | 整体评价偏负面 |
| `1` | 正面 | 整体评价偏正面 |

标签映射在 `src/dianping_sentiment/__init__.py` 定义，训练时保存到模型 `config.json`。服务启动时检查模型类别数和标签映射，拒绝三分类、标签反向或映射不明的模型。

- 没有“中性”“混合情感”或“未知”标签。中性、褒贬混合和无关文本仍会被分到正面或负面。
- `confidence` 是两个类别的 softmax 得分，未经概率校准，不等于模型准确率；高分也可能判断错误。
- 默认最多处理 128 个 token（包含特殊 token），超出的内容截断。token 数不等于字符数，长评价后半段可能无法参与判断。
- 反讽、否定、转折示例用于展示能力边界，不能保证所有相似句式都正确。
- 增加中性标签需要重新定义标注体系、准备三类数据、调整代码并重新训练；只修改 `num_labels` 无法扩展当前模型。

## 实现与目录

服务使用 Flask + Waitress。模型在进程启动时从本地文件加载一次，设置为推理模式并按配置预热。评估和 API 共用 `model.py` 的推理逻辑，使用动态 padding 和 `torch.inference_mode()`；批量结果保持输入顺序。

当前依赖锁定为 CPU 版 PyTorch，默认部署不需要 GPU。训练和推理使用本地权重，模型加载禁用 Hub 下载。浏览器向 API 发送文本，推理在服务部署机器执行；云端部署时，文本会传到云服务器。

```text
bert_finetune/
├── pyproject.toml / uv.lock           # 依赖声明与锁文件
├── config.yaml                       # 训练、评估及服务配置
├── Dockerfile / .dockerignore         # 镜像构建
├── docker-compose.yml                # 从源码构建、服务及容器训练
├── docker-compose.deploy.yml         # 云端仅推理，使用现成镜像
├── .env.example                      # Compose 环境变量示例
├── scripts/cli.py                    # generate / train / evaluate / serve
├── src/dianping_sentiment/
│   ├── config.py                     # 配置加载、校验和模型目录解析
│   ├── data.py                       # 模板语料、数据校验和 Dataset
│   ├── train.py                      # BERT 微调、运行存档和曲线
│   ├── evaluate.py                   # 指标、混淆矩阵及错例
│   ├── model.py                      # 模型加载和统一推理
│   ├── service.py                    # 请求校验、路由和服务日志
│   ├── templates/                    # Jinja 页面
│   └── static/                       # JavaScript / CSS
├── tests/test_all.py                  # 自动化测试，无需 pytest
├── data/test_sentences.json           # 22 条人工标注示例
├── docs/                             # 页面截图
├── bert-base-chinese/                # 自备预训练权重，仅训练需要
├── data/processed/                   # 生成的训练数据，不入库
├── models/<运行名>/                   # 训练模型及分词器，不入库
└── runs/<运行名>/                     # 配置、指标、日志及曲线，不入库
```

## 本地开发与训练

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)（本次验证版本为 0.13.0）。`pyproject.toml` 将 Python 范围限制为 `>=3.12,<3.13`。以下命令均在项目根目录执行；仅运行容器时，宿主机不需要 Python 或 uv。

### 1. 安装依赖

```bash
uv sync --locked
```

通过锁文件安装一致版本。PyTorch CPU wheel 来自官方源，首次安装需要联网且下载量较大。

### 2. 下载预训练权重（训练时需要）

```bash
uv run hf download google-bert/bert-base-chinese config.json model.safetensors vocab.txt tokenizer.json tokenizer_config.json --local-dir bert-base-chinese
```

模型来源：[google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)。权重与分词文件需完整放在 `bert-base-chinese/`；仓库和镜像均不包含它们。若已有训练模型且仅提供 API，可跳过下载和训练。

### 3. 生成数据、训练和评估

```bash
uv run python scripts/cli.py generate
uv run python scripts/cli.py train
uv run python scripts/cli.py evaluate
```

默认生成正负各 200 条模板样本，按类别分层切出 20% 验证集，训练 5 轮。训练数据格式如下，标签只允许整数 `0`、`1`：

```json
[
  {"text": "菜品好吃，服务也很周到", "label": 1},
  {"text": "等了很久，上来的菜还是凉的", "label": 0}
]
```

训练保存最后一轮权重，不保存 checkpoint，也不自动还原验证集最优轮。输出包括：

- `models/<运行名>/`：`model.safetensors`、`config.json`、分词器文件及 `train.yaml` 配置副本。
- `runs/<运行名>/`：`manifest.json`、`metrics.json`、`console.log`、`training_curve.png`。
- 评估将逐条预测保存到所选模型目录的 `eval_results.json`，因此评估时该目录需要可写。

指定模型评估、交互预测或重绘曲线：

```bash
uv run python scripts/cli.py evaluate --model-dir models/20260917-192434
uv run python scripts/cli.py evaluate --interactive
uv run python scripts/cli.py train --replot runs/20260917-192434
```

`--config` 是全局参数，应放在子命令之前，例如 `uv run python scripts/cli.py --config config.yaml serve`。

### 4. 启动服务

```bash
uv run python scripts/cli.py serve
```

浏览器访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)。没有可用模型时启动失败；项目不会自动替你下载权重或训练。

![单条分析](docs/screenshot-single.png)

![批量分析](docs/screenshot-batch.png)

截图为历史版本示例，具体页面文案以当前运行版本为准。

## 配置与模型选择

`config.yaml` 包含 `data`、`model`、`train`、`run`、`serving` 五个配置段。支持的服务环境变量如下，环境变量优先于 YAML：

| 环境变量 | 对应配置 | 用途 |
|---|---|---|
| `DS_HOST` | `serving.host` | 容器内部监听地址 |
| `DS_PORT` | `serving.port` | 服务端口；Compose 中另用于宿主端口映射 |
| `DS_DEBUG` | `serving.debug` | 开发模式，部署时保持关闭 |
| `DS_MODEL_DIR` | `serving.model_dir` | 训练模型目录 |
| `DS_NUM_THREADS` | `serving.num_threads` | PyTorch CPU 线程数，`0` 表示自动 |
| `DS_LOG_LEVEL` | `serving.log_level` | 日志级别 |

`threads`、`warmup`、`max_batch_size`、`max_text_length`、`max_content_length` 通过 YAML 配置。默认批量上限 64 条、每条最多 4096 字符、请求体最多 1 MiB。字符限制用于输入校验，模型截断上限仍是 `model.max_len` 个 token。

`DS_MODEL_DIR` 非空时优先使用它；未指定或为空时保留 YAML 的 `serving.model_dir`。两者都未指定时，从 `models/` 下含 `config.json` 的子目录按名称排序选择最后一个。只检查候选目录，不代表完整模型有效；最终加载仍会校验配置和权重。

模型在启动时选定，不会随目录变化热更新。部署建议明确指定已验证版本，便于回滚；只有一个模型目录时也可以使用自动选择。`evaluate --model-dir` 优先于服务配置，未传该参数时与服务使用同一套模型选择规则。

## Docker 部署

### 本机从源码构建

需要 Docker Engine / Docker Desktop 和 Docker Compose v2 或兼容版本；Docker Desktop 使用 Linux 容器。模型通过目录挂载，不进入镜像。

```bash
docker compose build api
docker compose up -d --no-build api
docker compose ps
docker compose logs --tail 50 api
```

默认端口只绑定宿主机 `127.0.0.1:8000`。模型、配置必须存在于 Compose 文件所在目录：

```text
部署目录/
├── docker-compose.yml
├── config.yaml
└── models/
    └── 20260917-192434/   # 上传整个模型目录，包含权重、标签配置及分词器
```

可复制 `.env.example` 为 `.env` 固定模型和端口。Compose 中容器端口始终为 `8000`，`.env` 的 `DS_PORT` 控制宿主映射端口；例如 `DS_PORT=9000` 时访问宿主机 `9000`。

### 云服务器使用已有镜像

服务器只需 Docker 和 Compose，无需安装 Python、uv，也无需下载预训练权重或重新训练。镜像需与服务器 CPU 架构匹配。

1. 发布构建好的镜像到你的镜像仓库，或通过 `docker save` / `docker load` 传输。
2. 上传 `docker-compose.deploy.yml`、`config.yaml` 和完整训练模型目录 `models/<运行名>/`。
3. 在部署目录创建 `.env`，将 `DS_IMAGE` 替换成已发布的实际镜像名和版本：

```dotenv
DS_IMAGE=你的镜像仓库/你的镜像名:版本
DS_MODEL_DIR=/app/models/20260917-192434
DS_PORT=8000
DS_BIND_ADDRESS=127.0.0.1
DS_NUM_THREADS=0
DS_LOG_LEVEL=INFO
```

服务器执行：

```bash
docker compose -f docker-compose.deploy.yml pull api
docker compose -f docker-compose.deploy.yml up -d --no-build api
docker compose -f docker-compose.deploy.yml ps
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:8000/api/meta
```

若通过 `docker load` 导入镜像，保证 `DS_IMAGE` 与导入的名称和标签一致，跳过 `pull`。云端 Compose 没有 `build`，不会要求服务器提供源码或 Dockerfile。

默认只允许服务器本机访问，适合由反向代理提供 HTTPS。若需要外部直连，将 `DS_BIND_ADDRESS` 改为 `0.0.0.0` 并配置云安全组/防火墙。当前没有认证和限流，不宜直接作为不受保护的公开 API。容器的非 root 用户 UID 为 `10001`，挂载的模型和配置需要可读，目录需要可遍历。

### 停止、更新与回滚

以下命令为本地 Compose；云端请在 `docker compose` 后加 `-f docker-compose.deploy.yml`。

```bash
docker compose stop api                  # 停止但保留容器
docker compose start api                 # 使用同一容器启动，无需构建
docker compose down                      # 移除容器与项目网络，挂载文件仍保留
docker compose up -d --no-build api       # 从已有镜像重新创建并启动
docker compose restart api               # 修改挂载的 config.yaml 后重启
docker compose up -d --no-build --force-recreate api  # 修改 .env 后重新创建
```

修改源码或依赖后需要重新构建并发布镜像；修改模型或挂载配置不需要重建镜像。切换模型时上传新目录、修改 `.env` 的 `DS_MODEL_DIR`、重建容器，再通过 `/api/meta` 确认版本和预测接口做冒烟测试。回滚时指定旧镜像版本和旧模型目录。

健康检查访问 `/api/health`，不执行模型前向。模型加载失败会阻止服务启动；`healthy` 不证明预测效果或高负载下的可用性。`restart: unless-stopped` 用于进程退出后的重启，不会因为 `unhealthy` 状态自动重启。

### 容器内训练（可选）

使用源码仓库的 `docker-compose.yml`，另外准备 `bert-base-chinese/`。训练服务通过 profile 与 API 分离：

```bash
docker compose build train
docker compose run --rm train generate
docker compose run --rm train
docker compose run --rm train evaluate
```

产物写回宿主机的 `data/`、`runs/`、`models/`。Linux 上可通过 `.env` 的 `DS_TRAIN_UID`、`DS_TRAIN_GID` 指定宿主用户身份，并提前创建可写目录；未指定时训练容器使用 root，API 始终使用非 root 用户。

## API

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | Web 页面 |
| GET | `/api/health` | 存活状态和运行时长 |
| GET | `/api/meta` | 模型版本、设备、标签及输入限制 |
| POST | `/api/predict` | 单条预测，`{"text":"评价"}` |
| POST | `/api/predict_batch` | 批量预测，`{"texts":["评价1","评价2"]}` |

批量接口也接受 `texts` 为换行分隔字符串。字符串会去掉首尾空白，空行会被过滤；数组中的非字符串元素返回 400。

```bash
curl -X POST http://127.0.0.1:8000/api/predict -H 'Content-Type: application/json' -d '{"text":"菜很好吃，下次还来"}'
```

返回结构示例（分数和耗时仅用于说明格式）：

```json
{
  "text": "菜很好吃，下次还来",
  "prediction": {
    "label": "正面",
    "label_id": 1,
    "confidence": 0.98,
    "probs": {"负面": 0.02, "正面": 0.98}
  },
  "latency_ms": 40.0
}
```

批量返回包含 `texts`、`items`、`summary`、`latency_ms` 和 `per_item_ms`；`items[].index` 从 `0` 开始，与过滤空白后的 `texts` 对应。`latency_ms` 是分词与模型推理耗时，不含网络传输、排队及完整请求处理；批量的 `per_item_ms` 是总推理耗时除以条数。

输入错误返回 400 和 `{"error":"..."}`，请求体超限返回 413，内部错误返回 500。Flask 的错误响应使用 JSON；被 WSGI 服务器提前拒绝的超大请求可能返回文本响应。

## 评估与验证

已有模型 `20260917-192434` 在仓库 22 条人工标注示例上的历史结果：

| 指标 | 结果 |
|---|---|
| Accuracy | 21/22，95.45% |
| F1（正面为正类） | 94.74% |
| 混淆矩阵（行真实，列预测，顺序负面/正面） | `[[12, 0], [1, 9]]` |

该模型权重不在 Git 中；重新训练结果可能不同。这 22 条混合领域示例用于回归检查，样本量小，部分褒贬混合句标注有主观性，不能代表真实业务准确率。合成训练/验证集共享模板与词库，高验证分数不能证明跨模板或真实评论泛化。

延迟受 CPU、线程数、文本长度、批量大小和容器资源影响，因此不提供统一延迟或吞吐承诺。默认最多 8 个 PyTorch CPU 线程及 8 个 WSGI 请求线程；并发请求可能争抢 CPU 和内存，部署前应使用真实批量及并发进行容量测试。

运行自动化测试：

```bash
uv run python tests/test_all.py
```

覆盖数据与配置校验、二分类标签约束、HTTP 输入与响应、训练参数、评估边界和真实推理。真实模型用例在 `models/` 为空时跳过；纯单测通过不等于权重加载已验证。

## 后续业务化工作

- 使用取得合法使用权限的领域评论，定义中性/混合评价的标注规则，按来源、时间和模板隔离验证/测试集，并检查重复样本。
- 在独立测试集上报告各类别 precision/recall/F1 和错例；需要拒判或人工复核时，先评估阈值与概率校准。
- 加入模型版本管理、可复现的发布流程、HTTPS、认证/限流、指标监控、并发容量与内存评估。
- 长评论需要明确切片和聚合规则；三分类、方面级情感分析等需求需要调整任务与训练数据。
