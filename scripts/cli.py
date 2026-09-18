"""命令行入口：生成数据 / 训练 / 评估 / 启动服务。

用法:
    uv run python scripts/cli.py generate                     # 生成训练数据
    uv run python scripts/cli.py train                        # 训练（自动存档 + 产出模型）
    uv run python scripts/cli.py train --replot runs/<时间戳>  # 只重绘曲线，不训练
    uv run python scripts/cli.py evaluate                     # 用测试集评估最新模型
    uv run python scripts/cli.py evaluate --interactive       # 交互式逐条预测
    uv run python scripts/cli.py serve                        # 启动服务
    uv run python scripts/cli.py serve --host 0.0.0.0 --port 8000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def cmd_generate(args) -> None:
    from dianping_sentiment.config import load_config
    from dianping_sentiment.data import generate

    cfg = load_config(args.config)
    num = args.num if args.num is not None else cfg.data.num_per_class
    seed = args.seed if args.seed is not None else cfg.data.seed
    stats = generate(cfg.data.processed_path, num, seed)
    print("✅ 数据生成完成！")
    print(f"   总条数: {stats['total']}（正 {stats['positive']} / 负 {stats['negative']}）")
    print(f"   去重后唯一: {stats['unique']} 条")
    print(f"   已保存: {stats['path']}")


def cmd_train(args) -> None:
    if args.replot:
        from dianping_sentiment.train import replot_from_metrics

        replot_from_metrics(os.path.join(args.replot, "metrics.json"))
        return
    from dianping_sentiment.train import main

    main(args.config)


def cmd_evaluate(args) -> None:
    from dianping_sentiment.evaluate import main

    main(args.config, args.model_dir, args.interactive)


def cmd_serve(args) -> None:
    # 命令行参数优先级最高，写进环境变量后由 load_config 统一处理
    if args.host:
        os.environ["DS_HOST"] = args.host
    if args.port:
        os.environ["DS_PORT"] = str(args.port)

    from dianping_sentiment.config import load_config
    from dianping_sentiment.service import serve

    serve(load_config(args.config))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        prog="cli.py", description="大众点评客户评价情感分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  uv run python scripts/cli.py generate\n"
            "  uv run python scripts/cli.py train\n"
            "  uv run python scripts/cli.py evaluate --interactive\n"
            "  uv run python scripts/cli.py serve --port 8000\n"
        ),
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认用仓库根目录的 config.yaml）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="生成训练数据")
    p.add_argument("--num", type=int, default=None, help="每类样本数，覆盖配置")
    p.add_argument("--seed", type=int, default=None, help="随机种子，覆盖配置")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("train", help="训练模型")
    p.add_argument("--replot", metavar="RUN_DIR", default=None, help="只重绘该 run 的曲线，不训练")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="评估模型")
    p.add_argument("--model-dir", default=None, help="模型目录，默认取 models/ 下最新的")
    p.add_argument("--interactive", action="store_true", help="交互式逐条预测")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("serve", help="启动推理服务")
    p.add_argument("--host", default=None, help="监听地址，覆盖配置")
    p.add_argument("--port", type=int, default=None, help="监听端口，覆盖配置")
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
