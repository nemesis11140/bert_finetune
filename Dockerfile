# 大众点评客户评价情感分析 —— 服务镜像
#
# 模型权重不进镜像（400MB 且随训练变化），运行时挂载进来：
#   docker build -t dianping-sentiment .
#   docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" dianping-sentiment
# 想固定模型版本就显式指定：
#   docker run --rm -p 8000:8000 -v "$PWD/models:/app/models:ro" \
#     -e DS_MODEL_DIR=/app/models/20260917-192434 dianping-sentiment

FROM python:3.12-slim

# uv：装依赖用它，和本地开发用的是同一份 uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    DS_HOST=0.0.0.0 \
    DS_PORT=8000

WORKDIR /app

# 先只拷依赖清单并装依赖：改业务代码时这一层能命中缓存，不用重装 torch
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY config.yaml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
RUN uv sync --frozen

EXPOSE 8000

# 存活探针打 /api/health：只证明进程在，不碰模型（不会因为一次探针就做一次推理）
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).status == 200 else 1)"

CMD ["python", "scripts/cli.py", "serve"]
