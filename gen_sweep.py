#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成参数的量化扫描：温度 × top-k × 随机种子，统计"续写质量"的可量化指标。

光看几条样例不够直观，这里对每个 (temperature, top_k) 组合采样多个种子，记录：

* mean_len       平均续写长度（token）
* eos_rate       在 max_words 之前自然结束（生成到 </s>）的比例
* logprob/token  生成文本在**同一个模型**下的平均对数概率（越高越"像训练语料"）
* ttr            类符/形符比 = 去重后的 token 数 / 总 token 数（越高越多样）
* rep_rate       重复率 = 1 - ttr
* unk_rate       生成里 <unk> 的比例（正常情况下应为 0，因为生成时禁用了 <unk>）

    python gen_sweep.py --model models/word.kn5.npz --seeds 10

输出：results/generation_sweep.jsonl（逐条）与 results/generation_sweep_summary.md（汇总）
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np

import generate as G
import ngram_lm as N

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

PREFIX = "在阳光明媚的五月，我们学校胜利召开了"


def sample_metrics(lm, ids: list[int], prefix: list[int]) -> dict:
    """统计一条续写的指标：长度、命中句末、模型对数概率、多样性、重复率。"""
    ctx = ([N.ID_BOS] * (lm.order - 1) + prefix)[-(lm.order - 1):]
    lps = []
    for w in ids + ([N.ID_EOS] if ids else []):
        lps.append(lm.logprob(int(w), ctx))
        ctx = (ctx + [int(w)])[-(lm.order - 1):]
    n = max(len(lps), 1)
    toks = [lm.vocab[i] for i in ids]
    # 只看实词/字，避免标点把多样性算高
    content = [t for t in toks if t not in "，。、；：（）“”《》—…！？"]
    ttr = (len(set(content)) / len(content)) if content else 1.0
    return {
        "length": len(ids),
        "logprob_per_token": round(sum(lps) / n, 4),
        "ttr": round(ttr, 4),
        "rep_rate": round(1 - ttr, 4),
        "unk_rate": round(sum(1 for i in ids if i == N.ID_UNK) / max(len(ids), 1), 4),
        "text": "".join(toks),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="生成参数（温度/top-k）量化扫描")
    p.add_argument("--model", default=os.path.join(HERE, "models", "word.kn5.npz"))
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--max-words", type=int, default=45)
    p.add_argument("--min-words", type=int, default=15)
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=[0.5, 0.7, 0.85, 1.0, 1.2, 1.5])
    p.add_argument("--top-ks", type=int, nargs="+", default=[0, 5, 20])
    p.add_argument("--seeds", type=int, default=10, help="每个组合采样的种子数（0..seeds-1）")
    p.add_argument("--out", default=os.path.join(RESULTS, "generation_sweep.jsonl"))
    p.add_argument("--out-md", default=os.path.join(RESULTS, "generation_sweep_summary.md"))
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    lm, meta = N.NgramLM.load(args.model)
    vocab_set = set(lm.vocab[3:])
    toks = G.segment_maxmatch(args.prefix, vocab_set, 7)
    word2id = {w: i for i, w in enumerate(lm.vocab)}
    prefix_ids = [word2id.get(t, word2id[N.UNK]) for t in toks]
    N.log(f"模型 {os.path.basename(args.model)}（{meta['order']}-gram/{meta['smoothing']}），"
          f"前缀分词：{' '.join(toks)}")

    records, summary = [], {}
    t0 = time.time()
    for T in args.temperatures:
        for k in args.top_ks:
            metrics = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                ids = lm.generate(prefix_ids, args.max_words, rng, temperature=T,
                                  top_k=k, min_words=args.min_words)
                m = sample_metrics(lm, ids, prefix_ids)
                m.update({"temperature": T, "top_k": k, "seed": seed})
                records.append(m)
                metrics.append(m)
            agg = {
                "temperature": T, "top_k": k, "n": len(metrics),
                "mean_len": round(st.mean(m["length"] for m in metrics), 2),
                "eos_rate": round(sum(1 for m in metrics if m["length"] < args.max_words)
                                  / len(metrics), 3),
                "logprob_per_token": round(st.mean(m["logprob_per_token"] for m in metrics), 4),
                "ttr": round(st.mean(m["ttr"] for m in metrics), 4),
                "rep_rate": round(st.mean(m["rep_rate"] for m in metrics), 4),
                "unk_rate": round(st.mean(m["unk_rate"] for m in metrics), 4),
            }
            summary[(T, k)] = agg
            N.log(f"  T={T:<4} top-k={k:<3} 平均长度 {agg['mean_len']:>5}  "
                  f"logprob/token {agg['logprob_per_token']:>8}  TTR {agg['ttr']:.3f}")

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write("# 生成参数扫描（温度 × top-k）\n\n")
        fh.write(f"- 模型：`{os.path.basename(args.model)}`（{meta['order']}-gram，"
                 f"{meta['smoothing']}）\n")
        fh.write(f"- 前缀：{args.prefix}（分词后 {' '.join(toks)}）\n")
        fh.write(f"- 每个组合采样 {args.seeds} 个种子，最多 {args.max_words} 个 token，"
                 f"前 {args.min_words} 个 token 禁止句末，禁止 `<unk>`\n\n")
        fh.write("| 温度 | top-k | 平均长度 | 提前结束比例 | logprob/token | TTR（多样性） | 重复率 |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for T in args.temperatures:
            for k in args.top_ks:
                a = summary[(T, k)]
                fh.write("| {} | {} | {} | {} | {} | {} | {} |\n".format(
                    T, k if k else "不限", a["mean_len"], a["eos_rate"],
                    a["logprob_per_token"], a["ttr"], a["rep_rate"]))
        fh.write("\n## 每个组合的一条样例\n\n")
        for r in records:
            if r["seed"] == 0:
                fh.write(f"- T={r['temperature']}, top-k={r['top_k'] or '不限'}：{r['text']}\n")
    N.log(f"写入 {args.out} 与 {args.out_md}（{len(records)} 条，{time.time() - t0:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
