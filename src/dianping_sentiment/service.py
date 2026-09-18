"""Flask 服务：REST API + 页面。

设计取舍：
- **页面不提供模型选择**。服务固定用配置指定的那一个模型（config.yaml 的
  ``serving.model_dir``，或环境变量 ``DS_MODEL_DIR``），模型版本由部署方控制。
- **模型在进程启动时加载一次并预热**，请求路径上没有磁盘 IO。
- 页面和 API 共用同一套推理，其他系统可以直接调 ``/api/predict``、``/api/predict_batch``。

请求校验（``_extract_*``）是纯函数、不碰 Flask，方便单测。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from . import ID2LABEL, POSITIVE_ID, __version__
from .config import Config, PROJECT_ROOT, load_config, resolve_model_dir
from .model import SentimentPredictor, InferenceResult

APP_TITLE = "大众点评客户评价情感分析"

# 页面上的示例句：故意挑了转折、反讽这类容易踩坑的，方便一眼看出模型能力边界
SAMPLES = [
    "这家店环境一般，不过菜是真好吃，强烈推荐",
    "这电影真好看，看得我昏昏欲睡",
    "服务态度真是太好了，好到让我想投诉",
    "份量足价格实惠，老板还送了小菜，下次还来",
]

logger = logging.getLogger("dianping_sentiment")


# ==================== 请求校验 ====================

class ValidationError(ValueError):
    """请求参数不合法，路由层转成 400。"""


def _extract_single(payload: object) -> str:
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValidationError("字段 text 不能为空")
    return text.strip()


def _extract_batch(payload: object, max_batch_size: int) -> list[str]:
    """允许两种形式：``{"texts": [...]}`` 或 ``{"texts": "一行一条的字符串"}``
    （页面上的批量输入框就是多行 textarea，直接传字符串更省事）。"""
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象")
    raw = payload.get("texts")

    if isinstance(raw, str):
        candidates = raw.splitlines()
    elif isinstance(raw, list):
        candidates = raw
    else:
        raise ValidationError("字段 texts 必须是数组或字符串")

    texts = [str(x).strip() for x in candidates]
    texts = [t for t in texts if t]
    if not texts:
        raise ValidationError("texts 里没有有效内容")
    if len(texts) > max_batch_size:
        raise ValidationError(f"一次最多分析 {max_batch_size} 条，当前 {len(texts)} 条")
    return texts


# ==================== 响应构造 ====================

def _summarize(items: list) -> dict:
    """情绪分布汇总，给前端画占比条用。"""
    total = len(items)
    positive = sum(1 for x in items if x.label_id == POSITIVE_ID)
    negative = total - positive
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_ratio": round(positive / total, 4) if total else 0.0,
        "negative_ratio": round(negative / total, 4) if total else 0.0,
    }


def _single_response(result: InferenceResult) -> dict:
    return {"prediction": result.items[0].to_dict(), "latency_ms": round(result.latency_ms, 2)}


def _batch_response(result: InferenceResult) -> dict:
    """批量响应：逐条结果 + 汇总。

    ``latency_ms`` 是整批耗时，``per_item_ms`` 才是分摊后的单条成本。
    """
    return {
        "items": [{"index": i, **p.to_dict()} for i, p in enumerate(result.items)],
        "summary": _summarize(result.items),
        "latency_ms": round(result.latency_ms, 2),
        "per_item_ms": round(result.per_item_ms, 2),
    }


# ==================== 应用 ====================

def _setup_logging(level: str) -> None:
    """配置根 logger。服务进程只配一次，重复调用不会叠加 handler。"""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(handler)
    root.setLevel(level.upper())


def create_app(config: Config | None = None, predictor: SentimentPredictor | None = None) -> Flask:
    """应用工厂。``predictor`` 可注入，测试里能塞假模型，不用真加载 400MB 权重。"""
    cfg = config or load_config()
    _setup_logging(cfg.serving.log_level)

    app = Flask(__name__)

    if predictor is None:
        model_dir = resolve_model_dir(cfg)
        logger.info("加载模型: %s", model_dir)
        predictor = SentimentPredictor(model_dir, max_len=cfg.model.max_len,
                                       num_threads=cfg.serving.num_threads)
        if cfg.serving.warmup:
            logger.info("预热完成，耗时 %.1f ms", predictor.warmup())

    app.extensions["predictor"] = predictor
    app.extensions["started_at"] = time.time()
    info = predictor.info()
    logger.info("服务就绪 | 模型=%s | 设备=%s | 线程=%s | max_len=%s",
                info["model_name"], info["device"], info["num_threads"], info["max_len"])

    # ---------- 请求日志 ----------

    @app.before_request
    def _start_timer() -> None:
        request._start = time.perf_counter()  # type: ignore[attr-defined]

    @app.after_request
    def _log_request(response):
        started = getattr(request, "_start", None)
        cost = (time.perf_counter() - started) * 1000 if started else -1
        logger.info("%s %s -> %s (%.1f ms)", request.method, request.path,
                    response.status_code, cost)
        return response

    # ---------- 错误处理 ----------

    @app.errorhandler(ValidationError)
    def _handle_validation_error(exc: ValidationError):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(404)
    def _handle_404(_):
        if request.path.startswith("/api/"):
            return jsonify({"error": f"接口不存在: {request.path}"}), 404
        return render_template("index.html", app_title=APP_TITLE,
                               labels=list(ID2LABEL.values()), samples=SAMPLES), 404

    @app.errorhandler(500)
    def _handle_500(exc):
        logger.exception("请求处理失败: %s", exc)
        return jsonify({"error": "服务内部错误，请查看服务日志"}), 500

    # ---------- 页面 ----------

    @app.get("/")
    def index():
        return render_template("index.html", app_title=APP_TITLE,
                               labels=list(ID2LABEL.values()), samples=SAMPLES)

    # ---------- API ----------

    @app.get("/api/health")
    def health():
        """存活探针：只证明进程还在，不碰模型。"""
        return jsonify({"status": "ok",
                        "uptime_seconds": round(time.time() - app.extensions["started_at"], 1)})

    @app.get("/api/meta")
    def meta():
        """当前生效的模型/运行环境，页面右上角展示的就是它。"""
        return jsonify({"app": APP_TITLE, "version": __version__, **predictor.info()})

    @app.post("/api/predict")
    def predict_single():
        text = _extract_single(request.get_json(silent=True))
        result = predictor.predict_one(text)
        logger.info("单条分析: %d 字 -> %s (%.1f ms)",
                    len(text), result.items[0].label, result.latency_ms)
        return jsonify({"text": text, **_single_response(result)})

    @app.post("/api/predict_batch")
    def predict_batch():
        texts = _extract_batch(request.get_json(silent=True), cfg.serving.max_batch_size)
        result = predictor.predict(texts)
        logger.info("批量分析: %d 条 -> 整批 %.1f ms / 单条 %.1f ms",
                    len(texts), result.latency_ms, result.per_item_ms)
        return jsonify({"texts": texts, **_batch_response(result)})

    return app


def serve(cfg: Config) -> None:
    """按配置启动服务：默认 waitress，debug 时用 Flask 自带服务器（有自动重载）。"""
    app = create_app(config=cfg)
    host, port = cfg.serving.host, cfg.serving.port

    if cfg.serving.debug:
        app.run(host=host, port=port, debug=True, use_reloader=True)
        return

    # Flask 自带的开发服务器是单线程、无连接管理的，不适合对外提供服务。
    # waitress 是纯 Python WSGI，Windows 上也能跑（gunicorn 不支持 Windows）。
    from waitress import serve as waitress_serve

    waitress_serve(app, host=host, port=port, threads=cfg.serving.threads)
