"""轻量冒烟测试（不依赖 pytest，直接运行: uv run python tests/test_data_generation.py）。

验证数据生成逻辑：数量、类别、去重、占位符替换是否完整。
"""
import os
import re
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sentiment import data_generation as dg


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


def test_no_empty_text():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "data.json")
        dg.generate(out, num_per_class=50, seed=42)

        import json

        with open(out, encoding="utf-8") as f:
            items = json.load(f)
        assert all(it["text"].strip() for it in items)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✅ {name}")
    print("全部通过")
