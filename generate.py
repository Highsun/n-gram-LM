#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用训练好的 n-gram 模型做文本续写。

因为词级模型是按"词"建模的，中文前缀需要先分词；这里用一个简单的
**前向最大匹配**（用模型词表当词典，最长 7 字）把前缀切开，这也是最常用的
词级 n-gram 生成做法。字级模型则天然一字一 token，不需要分词。

示例::

    # 词级 5-gram，采样续写 40 个词
    python generate.py --model models/word.kn5.npz \\
        --prefix "在阳光明媚的五月，我们学校胜利召开了" --max-words 40 \\
        --temperature 1.0 --top-k 0 --seed 2024

    # 贪心 / beam 解码
    python generate.py --model models/word.kn3.npz --prefix "在阳光明媚的五月，我们学校胜利召开了" \\
        --mode greedy --max-words 40
    python generate.py --model models/word.kn3.npz --prefix "..." --mode beam --beam 5 --max-words 40

    # 看看前缀后面概率最高的 10 个词
    python generate.py --model models/word.kn3.npz --prefix "在阳光明媚的五月，我们学校胜利召开了" --show-top 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import ngram_lm as N


def segment_maxmatch(text: str, vocab: set[str], max_len: int = 7) -> list[str]:
    """前向最大匹配分词；匹配不到就退化成单字。"""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        for j in range(min(max_len, n - i), 0, -1):
            piece = text[i:i + j]
            if piece in vocab:
                out.append(piece)
                i += j
                break
        else:
            out.append(text[i])
            i += 1
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="n-gram 语言模型文本生成")
    p.add_argument("--model", required=True, help="模型 .npz（train_ngram.py 的输出）")
    p.add_argument("--prefix", default="", help="中文前缀（会自动分词）")
    p.add_argument("--prefix-tokens", default="", help="已经分好词的前缀，空格分隔")
    p.add_argument("--max-words", type=int, default=40, help="最多续写多少个 token")
    p.add_argument("--min-words", type=int, default=10,
                   help="最少续写多少个 token（此前不允许出现 </s>，避免立刻断句）")
    p.add_argument("--mode", choices=["sample", "greedy", "beam"], default="sample")
    p.add_argument("--beam", type=int, default=5, help="beam search 宽度")
    p.add_argument("--temperature", type=float, default=1.0, help="采样温度（1.0=原始分布）")
    p.add_argument("--top-k", type=int, default=0, help="只在前 k 个候选里采样（0=不限）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--join", choices=["none", "space"], default="none",
                   help="输出拼接方式：none=中文连写（默认），space=保留空格")
    p.add_argument("--show-top", type=int, default=0, help="只打印前缀后的 top-N 候选词")
    p.add_argument("--tag", default="", help="结果表里显示的参数标签")
    p.add_argument("--json-out", default="", help="把结果追加写入该 JSONL")
    return p


def main() -> int:
    args = build_parser().parse_args()
    lm, meta = Ngram_load(args.model)
    vocab_set = set(lm.vocab[3:])          # 去掉 <unk>/<s>/</s>

    if args.prefix_tokens:
        toks = args.prefix_tokens.split()
    elif args.prefix:
        toks = segment_maxmatch(args.prefix, vocab_set, 7)
    else:
        toks = []

    word2id = {w: i for i, w in enumerate(lm.vocab)}
    unk = word2id[N.UNK]
    prefix_ids = [word2id.get(t, unk) for t in toks]
    n_oov = sum(1 for t in toks if t not in word2id)

    ctx = [N.ID_BOS] * (lm.order - 1) + prefix_ids

    if args.show_top:
        ids, probs = lm.distribution(ctx, temperature=1.0, top_k=0)
        order = np.argsort(-probs)[: args.show_top]
        print(f"模型：{args.model}（{meta['order']}-gram / {meta['smoothing']}）")
        print(f"前缀分词：{' '.join(toks)}")
        print(f"前缀中未登录词：{n_oov}")
        print("下一个词 top-%d：" % args.show_top)
        for i in order:
            print(f"  {lm.vocab[int(ids[i])]:>8s}  {probs[i]:.6f}")
        return 0

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    new_ids = lm.generate(prefix_ids, args.max_words, rng, temperature=args.temperature,
                          top_k=args.top_k, mode=args.mode, beam=args.beam,
                          min_words=args.min_words)
    dt = time.time() - t0

    sep = "" if args.join == "none" else " "
    out_words = [lm.vocab[i] for i in new_ids]
    text = sep.join(toks + out_words)
    print(f"模型    : {args.model}  ({meta['order']}-gram, {meta['smoothing']}"
          + (f", alpha={meta['alpha']}" if meta["smoothing"] == "sb" else "") + ")")
    print(f"参数    : mode={args.mode} T={args.temperature} top_k={args.top_k} "
          f"seed={args.seed} max_words={args.max_words}")
    print(f"前缀分词: {' '.join(toks)}   （未登录词 {n_oov}）")
    print(f"续写    : {sep.join(out_words)}")
    print(f"全文    : {text}")
    print(f"耗时    : {dt * 1000:.1f} ms，{len(new_ids)} 个 token")

    if args.json_out:
        rec = {
            "tag": args.tag or f"{meta['order']}gram-{meta['smoothing']}-T{args.temperature}-k{args.top_k}",
            "model": args.model,
            "order": meta["order"],
            "smoothing": meta["smoothing"],
            "alpha": meta.get("alpha"),
            "mode": args.mode,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "prefix": args.prefix or args.prefix_tokens,
            "prefix_tokens": toks,
            "continuation": "".join(out_words) if args.join == "none" else " ".join(out_words),
            "text": text,
            "new_tokens": len(new_ids),
            "seconds": round(dt, 3),
        }
        with open(args.json_out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


def Ngram_load(path: str):
    """包一层，方便以后换别的模型格式。"""
    return N.NgramLM.load(path)


if __name__ == "__main__":
    sys.exit(main())
