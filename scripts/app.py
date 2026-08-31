"""Web 交互页面入口。运行: uv run streamlit run scripts/app.py

启动后在浏览器打开 http://localhost:8501
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sentiment.webapp import main  # noqa: E402

main(os.path.join(ROOT, "config.yaml"))
