#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 models/ 下所有训练好的模型，在**同一份完整验证集**上重算一遍困惑度，
输出统一的指标表（markdown + json），供画图和写实验报告直接引用。

为什么单独做这一步：`run_experiments.py` 里的 PPL 受 `--ppl-limit` 限制
（字级验证集有 243,485 个 token，会被截到 20 万），不同模型之间口径可能不一致。
这里统一：完整验证集、不做截断、统一含 OOV 与句末 </s>。

    python ppl_table.py                      # 全部模型
    python ppl_table.py --tag word.kn5       # 只算匹配 tag 的模型

输出：
    results/ppl_full.json   原始记录（每个模型一条）
    results/ppl_full.md     markdown 表（可直接贴进报告）
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

import ngram_lm as N

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")
CORPUS = os.path.join(HERE, "corpus")


def chars_per_token() -> dict[str, float]:
    """语料里"每 token 多少个字"，用于把 bits/token 换算成 bits/字。"""
    out = {}
    for unit in ("word", "char"):
        path = os.path.join(CORPUS, f"{unit}.train.txt")
        n_char = n_tok = 0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    toks = line.split()
                    n_tok += len(toks)
                    n_char += sum(len(t) for t in toks)
        out[unit] = (n_char / n_tok) if n_tok else 1.0
    return out


def unit_of(meta: dict) -> str:
    return "char" if "char" in os.path.basename(meta.get("train_file", "")) else "word"


def main() -> int:
    p = argparse.ArgumentParser(description="在完整验证集上重算各模型困惑度")
    p.add_argument("--tag", default="", help="只处理 tag 含该子串的模型")
    p.add_argument("--out-json", default=os.path.join(RESULTS, "ppl_full.json"))
    p.add_argument("--out-md", default=os.path.join(RESULTS, "ppl_full.md"))
    args = p.parse_args()

    cpt = chars_per_token()
    models = sorted(glob.glob(os.path.join(MODELS, "*.npz")))
    models = [m for m in models if not os.path.basename(m).startswith(("cmp.", "tmp"))]
    rows: list[dict] = []
    for path in models:
        tag = os.path.splitext(os.path.basename(path))[0]
        if args.tag and args.tag not in tag:
            continue
        meta_path = os.path.splitext(path)[0] + ".json"
        if not os.path.exists(meta_path):
            continue
        meta = json.load(open(meta_path, encoding="utf-8"))
        unit = unit_of(meta)
        lm, _ = N.NgramLM.load(path)
        word2id = {w: i for i, w in enumerate(lm.vocab)}
        dev = N.encode_file(os.path.join(CORPUS, f"{unit}.dev.txt"), word2id)
        t0 = time.time()
        r = lm.perplexity(dev, max_tokens=0)          # 0 = 不截断
        dt = round(time.time() - t0, 2)
        # 各阶 n-gram 的 singleton 比例（剪枝前统计，直接存在 npz 里）
        raw = np.load(path, allow_pickle=False)
        singletons = {}
        for k in range(2, lm.order + 1):
            key = f"s{k}"
            if key in raw:
                n1, _n2 = int(raw[key][0]), int(raw[key][1])
                size = int(raw[f"c{k}"].shape[0])
                singletons[str(k)] = {"size": size, "singleton": n1,
                                      "ratio": round(n1 / max(size, 1), 4)}
        bits = -r["logprob"] / math.log(2)
        row = {
            "tag": tag, "unit": unit, "order": lm.order, "smoothing": lm.smoothing,
            "prune": meta.get("prune"),
            "vocab_size": len(lm.vocab),
            "tokens": r["tokens"], "oov": r["oov"], "oov_rate": r["oov_rate"],
            "ppl": r["ppl"], "logprob_per_token": r["logprob"],
            "bits_per_token": round(bits, 3),
            "bits_per_char": round(bits / cpt[unit], 3),
            "chars_per_token": round(cpt[unit], 3),
            "model_mb": round(os.path.getsize(path) / 1048576, 1),
            "train_seconds": meta.get("train_seconds"),
            "peak_mem_mb": meta.get("peak_mem_mb"),
            "ngram_types": meta.get("ngram_types"),
            "timing": meta.get("timing"),
            "ppl_seconds": dt,
            "singletons": singletons,
        }
        rows.append(row)
        N.log(f"{tag:34s} {unit} {lm.order}-gram  PPL={r['ppl']:>9.2f}  "
              f"bits/字={row['bits_per_char']:.3f}  OOV={r['oov']:>5} ({r['oov_rate']:.2%})  "
              f"{dt}s")

    rows.sort(key=lambda r: (r["unit"], r["order"], r["tag"]))
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)

    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write("# 各模型在完整验证集上的困惑度\n\n")
        fh.write("统一口径：在完整验证集上评测、不做 token 截断、含 OOV 与句末 `</s>`。"
                 f"`bits/字` = `bits/token ÷ 每 token 字数`（语料实测：词级 {cpt['word']:.3f} 字/词，"
                 f"字级 {cpt['char']:.3f} 字/字），这是近似换算，未严格对齐 EOS 分母和 OOV 信息损失，不能据此判断跨粒度优劣。\n\n")
        fh.write("| 模型 | 粒度 | 阶数 | 平滑 | 分阶剪枝 | 词表 | 验证 token | OOV | "
                 "PPL | logprob/token | bits/token | bits/字 | 模型 MB | 训练 s | 峰值内存 MB |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write("| `{}` | {} | {} | {} | {} | {} | {} | {} ({:.2%}) | {:.2f} | {:.3f} | {:.3f} | "
                     "{:.3f} | {} | {} | {} |\n".format(
                         r["tag"], r["unit"], r["order"], r["smoothing"],
                         "/".join(str(x) for x in (r["prune"] or [1])),
                         r["vocab_size"], r["tokens"], r["oov"], r["oov_rate"],
                         r["ppl"], r["logprob_per_token"], r["bits_per_token"],
                         r["bits_per_char"], r["model_mb"], r["train_seconds"],
                         r["peak_mem_mb"]))
    N.log(f"写入 {args.out_json} 与 {args.out_md}（共 {len(rows)} 个模型）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
