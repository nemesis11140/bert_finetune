"""冒烟测试（不依赖 pytest，直接运行: uv run python tests/test_all.py）。

覆盖两块：
1. 数据生成——数量、类别、去重、占位符完整性、转折句式是否真的进了生成流程。
2. 服务层——配置加载与环境变量覆盖、请求校验、响应结构，
   以及**动态 padding 是否改变了预测**（提速的核心风险点，用测试集逐条比对）。
"""
import os
import re
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dianping_sentiment import data as dg


def test_generate_small():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "data.json")
        stats = dg.generate(out, num_per_class=20, seed=42)
        assert stats["total"] == 40, stats
        assert stats["positive"] == 20
        assert stats["negative"] == 20
        assert stats["unique"] == 40, "生成的句子不应重复"

        import json

        with open(out, encoding="utf-8") as f:
            items = json.load(f)
        labels = {it["label"] for it in items}
        assert labels == {0, 1}


def test_no_leftover_placeholders():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "data.json")
        dg.generate(out, num_per_class=50, seed=42)

        import json

        with open(out, encoding="utf-8") as f:
            items = json.load(f)
        leftover = [it["text"] for it in items if re.search(r"\{[a-z_]+\}", it["text"])]
        assert not leftover, f"存在未替换的占位符: {leftover}"


def test_all_placeholders_have_word_pool():
    """每个模板的占位符都要在词库里有对应键，否则会漏出字面量 {xxx}。

    比随机抽样更可靠：新加模板写错 key 时立刻报错，不用等生成时碰运气。
    """
    groups = [
        (dg.positive_templates + dg.positive_negation_templates + dg.positive_contrast_templates,
         dg.positive_words),
        (dg.negative_templates + dg.negative_negation_templates
         + dg.negative_contrast_templates + dg.negative_sarcasm_templates,
         dg.negative_words),
    ]
    for templates, pool in groups:
        for t in templates:
            for key in re.findall(r"\{([a-z_]+)\}", t):
                assert key in pool, f"模板 {t!r} 的占位符 {{{key}}} 在词库里没有对应项"


def test_contrast_templates_reachable():
    """转折句式必须真的进了生成流程（否则模型又会退化成"见褒义即正面"）。"""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "data.json")
        dg.generate(out, num_per_class=500, seed=7)

        import json

        with open(out, encoding="utf-8") as f:
            items = json.load(f)
        contrast = [it for it in items if any(w in it["text"] for w in ("但", "却", "不过", "可惜", "好在"))]
        assert len(contrast) >= 20, f"转折句式样本过少: {len(contrast)} 条"
        # 转折句必须两类都有，不能只教模型往一边倒
        assert {it["label"] for it in contrast} == {0, 1}, "转折句式只覆盖了单一类别"


def test_no_empty_text():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "data.json")
        dg.generate(out, num_per_class=50, seed=42)

        import json

        with open(out, encoding="utf-8") as f:
            items = json.load(f)
        assert all(it["text"].strip() for it in items)


# ==================== 服务层 ====================

from dianping_sentiment import config as config_mod  # noqa: E402
from dianping_sentiment import service  # noqa: E402
from dianping_sentiment.config import (  # noqa: E402
    Config, DataConfig, ModelConfig, RunConfig, ServingConfig, TrainConfig,
)
from dianping_sentiment.model import InferenceResult, Prediction  # noqa: E402

CONFIG_PATH = os.path.join(ROOT, "config.yaml")
TEST_SENTENCES = os.path.join(ROOT, "data", "test_sentences.json")


def _fake_config(models_dir="models", serving_model_dir=""):
    """造一个只用于测路径解析的 Config，不读磁盘。"""
    return Config(
        data=DataConfig(processed_path="x", test_sentences_path="y",
                        num_per_class=1, val_ratio=0.2, seed=42),
        model=ModelConfig(base_model="m", num_labels=2, max_len=128),
        train=TrainConfig(batch_size=1, epochs=1, learning_rate=1e-5,
                          weight_decay=0.0, logging_steps=1, seed=42),
        run=RunConfig(runs_dir="runs", models_dir=models_dir),
        serving=ServingConfig(model_dir=serving_model_dir),
    )


def test_config_loads_and_env_overrides_win():
    cfg = config_mod.load_config(CONFIG_PATH)
    assert cfg.serving.port == 8000
    assert cfg.serving.max_batch_size == 64
    assert cfg.model.max_len == 128

    # 环境变量优先级最高（容器部署靠它切模型）
    os.environ["DS_PORT"] = "9001"
    os.environ["DS_MODEL_DIR"] = "models/whatever"
    os.environ["DS_LOG_LEVEL"] = "debug"
    try:
        override = config_mod.load_config(CONFIG_PATH)
        assert override.serving.port == 9001, override.serving
        assert override.serving.model_dir == "models/whatever"
        assert override.serving.log_level == "DEBUG", override.serving
    finally:
        for k in ("DS_PORT", "DS_MODEL_DIR", "DS_LOG_LEVEL"):
            os.environ.pop(k, None)


def test_config_tolerates_partial_serving_section():
    """serving 段可以只写一半，缺的项走默认值。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "data: {processed_path: a, test_sentences_path: b, num_per_class: 1,"
                " val_ratio: 0.2, seed: 1}\n"
                "model: {base_model: m, num_labels: 2, max_len: 128}\n"
                "train: {batch_size: 1, epochs: 1, learning_rate: 0.00001,"
                " weight_decay: 0.0, logging_steps: 1, seed: 1}\n"
                "run: {runs_dir: runs, models_dir: models}\n"
                "serving: {port: 8123}\n"
            )
        cfg = config_mod.load_config(path)
        assert cfg.serving.port == 8123
        assert cfg.serving.host == "127.0.0.1"
        assert cfg.serving.max_batch_size == 64


def test_config_rejects_missing_section_and_typo():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        # 少一整段 → 报错要指出缺哪段，而不是抛裸 KeyError
        with open(path, "w", encoding="utf-8") as f:
            f.write("data: {processed_path: a, test_sentences_path: b, num_per_class: 1,"
                    " val_ratio: 0.2, seed: 1}\n")
        try:
            config_mod.load_config(path)
        except ValueError as exc:
            assert "缺少必需的配置段" in str(exc), exc
        else:
            raise AssertionError("缺配置段时应该报错")

        # serving 段里写错字段名 → 明确报错，而不是被静默忽略
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "data: {processed_path: a, test_sentences_path: b, num_per_class: 1,"
                " val_ratio: 0.2, seed: 1}\n"
                "model: {base_model: m, num_labels: 2, max_len: 128}\n"
                "train: {batch_size: 1, epochs: 1, learning_rate: 0.00001,"
                " weight_decay: 0.0, logging_steps: 1, seed: 1}\n"
                "run: {runs_dir: runs, models_dir: models}\n"
                "serving: {prot: 8000}\n"
            )
        try:
            config_mod.load_config(path)
        except ValueError as exc:
            assert "不认识的配置项" in str(exc), exc
        else:
            raise AssertionError("serving 段写错字段名时应该报错")


def test_resolve_model_dir_picks_latest_with_config_json():
    with tempfile.TemporaryDirectory() as tmp:
        models = os.path.join(tmp, "models")
        for name in ("20260101-000000", "20260202-000000"):
            os.makedirs(os.path.join(models, name))
            open(os.path.join(models, name, "config.json"), "w").close()
        # 混一个没有 config.json 的目录进来，不应该被选中
        os.makedirs(os.path.join(models, "logs"))

        picked = config_mod.resolve_model_dir(_fake_config(), root=tmp)
        assert picked.name == "20260202-000000", picked

        # 配置里写死了就优先用写死的（上线固定版本靠这个）
        explicit = config_mod.resolve_model_dir(
            _fake_config(serving_model_dir="models/20260101-000000"), root=tmp)
        assert explicit.name == "20260101-000000", explicit


def test_resolve_model_dir_raises_when_no_model():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "models"))
        try:
            config_mod.resolve_model_dir(_fake_config(), root=tmp)
        except FileNotFoundError as exc:
            assert "没有可用模型" in str(exc)
        else:
            raise AssertionError("模型目录为空时应该报错")


def test_extract_single_validation():
    assert service._extract_single({"text": " 好吃 "}) == "好吃"
    for bad in ({}, {"text": ""}, {"text": "   "}, {"text": 123}, None, "字符串"):
        try:
            service._extract_single(bad)
        except service.ValidationError:
            pass
        else:
            raise AssertionError(f"应当拒绝: {bad!r}")


def test_extract_batch_accepts_list_and_multiline_string():
    assert service._extract_batch({"texts": ["a", " b ", ""]}, 10) == ["a", "b"]
    assert service._extract_batch({"texts": "a\n\nb\n  c  "}, 10) == ["a", "b", "c"]

    for bad in ({}, {"texts": []}, {"texts": "\n\n"}, {"texts": 5}):
        try:
            service._extract_batch(bad, 10)
        except service.ValidationError:
            pass
        else:
            raise AssertionError(f"应当拒绝: {bad!r}")

    # 超过上限要拒绝，而不是静默截断
    try:
        service._extract_batch({"texts": ["a"] * 11}, 10)
    except service.ValidationError as exc:
        assert "最多" in str(exc)
    else:
        raise AssertionError("超出 max_batch_size 应当报错")


def test_response_shape_and_summary():
    items = [
        Prediction(label="正面", label_id=1, confidence=0.9, probs={"负面": 0.1, "正面": 0.9}),
        Prediction(label="负面", label_id=0, confidence=0.8, probs={"负面": 0.8, "正面": 0.2}),
    ]
    result = InferenceResult(items=items, latency_ms=40.0, batch_size=2)

    single = service._single_response(result)
    assert single["prediction"]["label"] == "正面"
    assert single["latency_ms"] == 40.0

    batch = service._batch_response(result)
    assert batch["summary"] == {"total": 2, "positive": 1, "negative": 1,
                                "positive_ratio": 0.5, "negative_ratio": 0.5}
    assert batch["per_item_ms"] == 20.0
    assert [x["index"] for x in batch["items"]] == [0, 1]


def test_http_routes_with_injected_fake_predictor():
    """用假模型跑一遍路由：不加载 400MB 权重，也能确认页面里没有模型选择控件。"""

    class FakePredictor:
        def info(self):
            return {"model_dir": "fake", "model_name": "fake", "device": "cpu",
                    "max_len": 128, "num_threads": 1}

        def predict(self, texts):
            return InferenceResult(
                items=[Prediction(label="正面", label_id=1, confidence=0.9,
                                  probs={"负面": 0.1, "正面": 0.9}) for _ in texts],
                latency_ms=1.0, batch_size=len(texts))

        def predict_one(self, text):
            return self.predict([text])

        def warmup(self):
            return 0.0

    app = service.create_app(config=config_mod.load_config(CONFIG_PATH),
                             predictor=FakePredictor())
    c = app.test_client()

    html = c.get("/").get_data(as_text=True)
    assert "大众点评客户评价情感分析" in html
    assert "select" not in html.lower(), "页面不应提供模型选择控件"

    assert c.get("/api/health").status_code == 200
    assert c.get("/api/meta").get_json()["model_name"] == "fake"
    assert c.get("/api/nope").status_code == 404
    assert c.get("/nope").status_code == 404, "非 API 的 404 也要能正常渲染页面"

    # 参数不合法一律 400，且带可读的 error
    for body in ({}, {"text": ""}):
        resp = c.post("/api/predict", json=body)
        assert resp.status_code == 400, resp
        assert resp.get_json()["error"]
    assert c.post("/api/predict_batch", json={"texts": ["x"] * 65}).status_code == 400

    ok = c.post("/api/predict", json={"text": "好吃"})
    assert ok.status_code == 200
    assert ok.get_json()["prediction"]["label"] == "正面"

    batch = c.post("/api/predict_batch", json={"texts": "好吃\n难吃"})
    assert batch.status_code == 200
    assert batch.get_json()["summary"]["positive"] == 2
    assert batch.get_json()["texts"] == ["好吃", "难吃"]


# ---------- 训练日志的可读性 ----------

def test_flatten_progress_strips_tqdm_control_sequences():
    """console.log 得能用普通编辑器打开——tqdm 的重画控制序列必须压平。

    tqdm 靠 ``\\r`` 回车 + ``ESC[A`` 上移光标在终端里原地重画进度条。这套东西
    原样落盘就是满屏控制字符（实测一次训练 507 个 \\r、70 个 ESC），加上 Windows
    文本模式还会把 \\n 写成 \\r\\n，两种 \\r 混在一起，打开就是乱码。
    """
    from dianping_sentiment.train import _flatten_progress

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False,
                                     encoding="utf-8", newline="") as f:
        path = f.name
        f.write("数据: 共 400 条\n")
        # 同一行反复重画，末尾跟一个光标上移 + 回车（tqdm 的典型写法）
        f.write("\r  0%|          | 0/100"
                "\r 50%|#####     | 50/100"
                "\r100%|##########| 100/100\x1b[A\r\n")
        f.write("{'loss': '0.59'}\n")
    try:
        _flatten_progress(path)
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
    finally:
        os.unlink(path)

    assert "\x1b" not in text, "ANSI 转义没清干净"
    assert "\r" not in text, "回车符没清干净"
    assert "100%|##########| 100/100" in text, "进度条的最终状态被误删"
    assert "数据: 共 400 条" in text and "{'loss': '0.59'}" in text, "正文内容丢失"


# ---------- 模型加载的守卫 ----------

def test_loaders_refuse_missing_dir_instead_of_downloading():
    """路径不存在时必须报错，不能变成静默联网下载。

    历史坑：``base_model`` 写的是裸名字 ``bert-base-chinese``，而这恰好也是
    HuggingFace 上真实存在的 repo id。目录一旦不在，transformers 不报错，
    而是去 Hub 下载，且下到全局缓存 ``~/.cache/huggingface`` 而不是当前目录——
    表现就是"没人让它下、它自己下了 1.6G"。
    """
    import torch

    from dianping_sentiment.model import load_for_training, load_tokenizer

    missing = os.path.join(ROOT, "bert-base-chinese-这个目录不存在")
    calls = {
        "load_tokenizer": lambda: load_tokenizer(missing),
        "load_for_training": lambda: load_for_training(missing, 2, torch.device("cpu")),
    }
    for name, call in calls.items():
        try:
            call()
        except FileNotFoundError:
            continue
        raise AssertionError(f"{name}() 在路径不存在时没有报错——守卫失效，会静默联网下载")


# ---------- 推理（需要真实模型，没有就跳过） ----------

def _load_predictor_and_sentences():
    try:
        model_dir = config_mod.latest_model_dir(os.path.join(ROOT, "models"))
    except FileNotFoundError:
        return None, None
    import json

    from dianping_sentiment.model import SentimentPredictor

    with open(TEST_SENTENCES, encoding="utf-8") as f:
        sentences = [it["text"] for it in json.load(f)]
    return SentimentPredictor(model_dir, max_len=128), sentences


def test_dynamic_padding_does_not_change_predictions():
    """提速的核心风险：动态 padding 是否会改变预测。

    用测试集把"补齐到 max_len=128"和"动态 padding"两种方式跑一遍，
    标签必须完全一致，置信度允许浮点级误差。
    """
    predictor, sentences = _load_predictor_and_sentences()
    if predictor is None:
        print("    （跳过：models/ 下没有模型，先跑 cli.py train）")
        return

    import torch

    dynamic = predictor.predict(sentences)

    padded = []
    for text in sentences:
        encoded = predictor.tokenizer(text, truncation=True, padding="max_length",
                                      max_length=128, return_tensors="pt")
        encoded = {k: v.to(predictor.device) for k, v in encoded.items()}
        with torch.inference_mode():
            probs = torch.softmax(predictor.model(**encoded).logits, dim=-1)[0]
        label_id = int(torch.argmax(probs))
        padded.append((label_id, float(probs[label_id])))

    for text, dy, (padded_id, padded_conf) in zip(sentences, dynamic.items, padded):
        assert dy.label_id == padded_id, f"标签变了：{text} 动态={dy.label} 补齐={padded_id}"
        assert abs(dy.confidence - padded_conf) < 1e-3, \
            f"置信度偏差过大：{text} {dy.confidence} vs {padded_conf}"


def test_batch_order_preserved_and_matches_single():
    """长度排序是内部优化，不能影响返回顺序。"""
    predictor, sentences = _load_predictor_and_sentences()
    if predictor is None:
        print("    （跳过：models/ 下没有模型，先跑 cli.py train）")
        return

    # 故意把长短句混在一起，最容易暴露排序后没还原的 bug
    mixed = [sentences[0], sentences[1], sentences[2], sentences[0][:6], sentences[3]]
    batch = predictor.predict(mixed)
    assert len(batch.items) == len(mixed)

    for text, pred in zip(mixed, batch.items):
        single = predictor.predict_one(text).items[0]
        assert pred.label == single.label, f"批量与单条结果不一致：{text}"
        assert abs(pred.confidence - single.confidence) < 1e-3, text


def test_empty_input_returns_empty_result():
    predictor, _ = _load_predictor_and_sentences()
    if predictor is None:
        print("    （跳过：models/ 下没有模型，先跑 cli.py train）")
        return

    result = predictor.predict([])
    assert result.items == []
    assert result.batch_size == 0
    assert result.per_item_ms == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✅ {name}")
    print("全部通过")
