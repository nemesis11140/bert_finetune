"""生成训练数据。用法: uv run python scripts/generate_data.py [--config config.yaml] [--num N] [--seed S]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="生成情感分类训练数据")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--num", type=int, default=None, help="每类样本数，覆盖 config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，覆盖 config.yaml")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    from sentiment.config import load_config
    from sentiment.data_generation import generate

    cfg = load_config(args.config)
    num = args.num if args.num is not None else cfg.data.num_per_class
    seed = args.seed if args.seed is not None else cfg.data.seed

    stats = generate(cfg.data.processed_path, num, seed)
    print("✅ 数据生成完成！")
    print(f"   总条数: {stats['total']}（正 {stats['positive']} / 负 {stats['negative']}）")
    print(f"   去重后唯一: {stats['unique']} 条")
    print(f"   已保存: {stats['path']}")


if __name__ == "__main__":
    main()
