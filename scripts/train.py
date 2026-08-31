"""训练入口。用法: uv run python scripts/train.py [--config config.yaml] [--replot runs/<run>]

--replot 只重绘某个 run 的曲线（读它的 metrics.json），不训练、几秒钟完成。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="训练 BERT 情感分类模型")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--replot", metavar="RUN_DIR", default=None,
                        help="只重绘该 run 的曲线，不训练")
    args = parser.parse_args()

    if args.replot:
        from sentiment.plotting import replot_from_metrics

        replot_from_metrics(os.path.join(args.replot, "metrics.json"))
        sys.exit(0)

    from sentiment.train import main as train_main

    train_main(args.config)


if __name__ == "__main__":
    main()
