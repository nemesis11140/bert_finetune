"""评估入口。用法:
    uv run python scripts/evaluate.py                 # 批量评估最新模型
    uv run python scripts/evaluate.py --model-dir models/<run>   # 指定模型
    uv run python scripts/evaluate.py --interactive   # 交互式逐条预测
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="评估已训练的模型")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--model-dir", default=None, help="模型目录，默认取 models/ 下最新的")
    parser.add_argument("--interactive", action="store_true", help="交互式逐条预测")
    args = parser.parse_args()

    from sentiment.evaluate import main as eval_main

    eval_main(args.config, args.model_dir, args.interactive)


if __name__ == "__main__":
    main()
