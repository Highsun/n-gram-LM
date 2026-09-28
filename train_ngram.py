#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练入口（自研 numpy n-gram 实现）。

示例::

    # 词级 3-gram + 插值 Kneser-Ney（默认）
    python train_ngram.py --train corpus/word.train.txt --dev corpus/word.dev.txt \\
        --order 3 --smoothing kn --out models/word.kn3.npz

    # 5-gram + Stupid Backoff（alpha=0.4），只看前 3 万句
    python train_ngram.py --train corpus/word.train.txt --order 5 --smoothing sb \\
        --alpha 0.4 --limit-lines 30000 --out models/word.sb5.part.npz

    # 字级 4-gram，并做频次剪枝（min-count=2，类似 SRILM -gtmin / KenLM --prune）
    python train_ngram.py --train corpus/char.train.txt --dev corpus/char.dev.txt \\
        --order 4 --min-count 2 --out models/char.kn4.prune2.npz

输出的 ``xxx.npz`` + ``xxx.json`` 可以被 ``generate.py`` 直接加载。
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import ngram_lm as N


def peak_mem_mb() -> float:
    """进程峰值内存（macOS/Linux 的 ru_maxrss 单位不同，这里统一成 MB）。"""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(rss / (1024 * 1024), 1)
    return round(rss / 1024, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="训练自研 numpy n-gram 语言模型")
    p.add_argument("--train", required=True, help="训练语料（每行一句，空格分词）")
    p.add_argument("--dev", default="", help="验证语料（可选，用于算困惑度）")
    p.add_argument("--out", default="", help="模型输出路径（.npz）")
    p.add_argument("--order", type=int, default=3, help="n-gram 阶数（1~8）")
    p.add_argument("--smoothing", choices=["kn", "sb"], default="kn",
                   help="kn=插值 Kneser-Ney（默认），sb=Stupid Backoff")
    p.add_argument("--alpha", type=float, default=0.4, help="Stupid Backoff 回退系数")
    p.add_argument("--min-count", type=int, default=1,
                   help="3 阶及以上 n-gram 的最小保留计数（简写，低阶不剪）")
    p.add_argument("--prune", default="",
                   help="按阶剪枝，如 \"1 1 2 2\" 表示 2/3/4/5 阶的最小保留计数"
                        "（对应 KenLM --prune 0 0 1 1 / SRILM -gt4min 2 -gt5min 2）")
    p.add_argument("--min-word-count", type=int, default=1, help="词表最小频次")
    p.add_argument("--max-types", type=int, default=N.MAX_WORD_TYPES, help="词表上限")
    p.add_argument("--limit-lines", type=int, default=0, help="只用训练集前 N 句（0=全部）")
    p.add_argument("--ppl-limit", type=int, default=200000,
                   help="算困惑度时最多用多少个 token（0=不限）")
    p.add_argument("--no-ppl", action="store_true", help="跳过困惑度计算")
    p.add_argument("--runs-jsonl", default="", help="把本次运行的统计追加到该 JSONL")
    p.add_argument("--tag", default="", help="给本次运行打个标签（写进统计）")
    p.add_argument("--encoding", default="utf-8")
    return p


def main() -> int:
    args = build_parser().parse_args()
    N.log(f"=== 训练 {args.order}-gram / 平滑 {args.smoothing} / "
          f"min_count={args.min_count} / 语料 {args.train}"
          + (f"（前 {args.limit_lines} 句）" if args.limit_lines else "") + " ===")

    t0 = time.time()
    lm, stats = N.train(
        train_path=args.train, dev_path="", order=args.order, smoothing=args.smoothing,
        alpha=args.alpha, min_count=args.min_count,
        prune=[int(x) for x in args.prune.split()] if args.prune else None,
        max_types=args.max_types,
        min_word_count=args.min_word_count, limit_lines=args.limit_lines,
        verbose=True, encoding=args.encoding,
    )
    N.log(f"训练完成，用时 {stats['train_seconds']}s，峰值内存 {peak_mem_mb()} MB")

    if args.dev and not args.no_ppl:
        N.log("在验证集上计算困惑度 …")
        t = time.time()
        word2id = {w: i for i, w in enumerate(lm.vocab)}
        dev_sents = N.encode_file(args.dev, word2id, 0, args.encoding)
        ppl = lm.perplexity(dev_sents, max_tokens=args.ppl_limit)
        ppl["seconds"] = round(time.time() - t, 2)
        stats["dev"] = ppl
        N.log(f"验证集：{ppl['tokens']} token，OOV {ppl['oov_rate']:.2%}，"
              f"困惑度 PPL = {ppl['ppl']}（{ppl['seconds']}s）")

    stats["tag"] = args.tag
    stats["train_file"] = args.train
    stats["dev_file"] = args.dev
    stats["limit_lines"] = args.limit_lines
    stats["peak_mem_mb"] = peak_mem_mb()
    stats["wall_seconds"] = round(time.time() - t0, 2)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        lm.save(args.out, stats)
        size_mb = round(os.path.getsize(args.out) / 1024 / 1024, 1)
        stats["model_size_mb"] = size_mb
        N.log(f"模型写入 {args.out}（{size_mb} MB）+ {os.path.splitext(args.out)[0]}.json")
        with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)

    if args.runs_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.runs_jsonl)), exist_ok=True)
        with open(args.runs_jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(stats, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
