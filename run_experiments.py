#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键跑完作业里的两组实验，并把结果整理成 markdown 放到 ``results/``。

    python run_experiments.py --stage train   # 训练时间 / 规模 / 剪枝 对比
    python run_experiments.py --stage gen     # 不同参数下的文本续写
    python run_experiments.py --stage all

训练阶段：对每个配置调用一次 ``train_ngram.py`` 子进程（这样测到的耗时和峰值内存
是"这个配置单独跑"的真实值），日志存到 ``results/logs/``，汇总到
``results/training_time.md`` 和 ``results/train_runs.jsonl``。

生成阶段：加载训练好的模型，用统一前缀做续写，结果写到
``results/generation_samples.md`` 和 ``results/generation_runs.jsonl``。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import generate as G
import ngram_lm as N

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(RESULTS, "logs")
MODELS = os.path.join(HERE, "models")

PREFIX = "在阳光明媚的五月，我们学校胜利召开了"


# --------------------------------------------------------------------------- #
# 训练配置
# --------------------------------------------------------------------------- #

def train_configs() -> list[dict]:
    cfg: list[dict] = []
    # 1) 阶数对比（词级、KN、全量语料）
    for order in (1, 2, 3, 4, 5):
        cfg.append(dict(unit="word", order=order, smoothing="kn", tag=f"word.kn{order}"))
    # 2) 平滑方法对比（Stupid Backoff 的 alpha）
    cfg.append(dict(unit="word", order=3, smoothing="sb", alpha=0.4, tag="word.sb3.a04"))
    cfg.append(dict(unit="word", order=3, smoothing="sb", alpha=1.0, tag="word.sb3.a10"))
    cfg.append(dict(unit="word", order=5, smoothing="sb", alpha=0.4, tag="word.sb5.a04"))
    # 3) 分阶剪枝（≈ KenLM --prune 0 0 1 1 / SRILM -gt4min 2 -gt5min 2）
    cfg.append(dict(unit="word", order=3, smoothing="kn", prune="1 2", tag="word.kn3.prune2"))
    cfg.append(dict(unit="word", order=3, smoothing="kn", prune="1 3", tag="word.kn3.prune3"))
    cfg.append(dict(unit="word", order=5, smoothing="kn", prune="1 1 2 2",
                    tag="word.kn5.prune22"))
    # 连二元一起剪（更激进的剪枝，模型更小、更依赖回退）
    cfg.append(dict(unit="word", order=3, smoothing="kn", prune="2 2",
                    tag="word.kn3.prune_all"))
    # 词表大小对比（其余参数相同）
    cfg.append(dict(unit="word", order=3, smoothing="kn", max_types=20000,
                    tag="word.kn3.v20k"))
    cfg.append(dict(unit="word", order=3, smoothing="kn", max_types=40000,
                    tag="word.kn3.v40k"))
    # 4) 语料规模对比（10% / 25% / 50%）
    for pct, lines in ((10, 28144), (25, 70362), (50, 140724)):
        cfg.append(dict(unit="word", order=3, smoothing="kn", limit_lines=lines,
                        tag=f"word.kn3.{pct}pct"))
    # 5) 字级 vs 词级
    cfg.append(dict(unit="char", order=3, smoothing="kn", tag="char.kn3"))
    cfg.append(dict(unit="char", order=5, smoothing="kn", tag="char.kn5"))
    return cfg


def run_training(cfg: dict, force: bool = False) -> dict:
    unit, tag = cfg["unit"], cfg["tag"]
    out = os.path.join(MODELS, f"{tag}.npz")
    log_path = os.path.join(LOGS, f"{tag}.log")
    if os.path.exists(out) and not force:
        meta = json.load(open(os.path.splitext(out)[0] + ".json", encoding="utf-8"))
        N.log(f"[训练] {tag} 已存在，跳过（训练 {meta.get('train_seconds')}s）")
        return meta

    cmd = [sys.executable, os.path.join(HERE, "train_ngram.py"),
           "--train", os.path.join(HERE, "corpus", f"{unit}.train.txt"),
           "--dev", os.path.join(HERE, "corpus", f"{unit}.dev.txt"),
           "--order", str(cfg["order"]),
           "--smoothing", cfg.get("smoothing", "kn"),
           "--alpha", str(cfg.get("alpha", 0.4)),
           "--min-count", str(cfg.get("min_count", 1)),
           "--max-types", str(cfg.get("max_types", 65533)),
           "--out", out,
           "--tag", tag,
           "--runs-jsonl", os.path.join(RESULTS, "train_runs.jsonl")]
    if cfg.get("limit_lines"):
        cmd += ["--limit-lines", str(cfg["limit_lines"])]
    if cfg.get("prune"):
        cmd += ["--prune", cfg["prune"]]

    N.log(f"[训练] {tag} …")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        N.log(f"[训练] {tag} 失败（returncode={proc.returncode}），见 {log_path}")
        raise SystemExit(1)
    meta = json.load(open(os.path.splitext(out)[0] + ".json", encoding="utf-8"))
    N.log(f"[训练] {tag} 完成：{time.time() - t0:.1f}s 墙钟 / "
          f"{meta['train_seconds']}s 训练 / 峰值 {meta['peak_mem_mb']}MB / "
          f"PPL {meta.get('dev', {}).get('ppl', 'N/A')}")
    return meta


# --------------------------------------------------------------------------- #
# 生成配置
# --------------------------------------------------------------------------- #

def gen_configs() -> list[dict]:
    cfg: list[dict] = []
    # 阶数对比（固定 T=1.0, 无 top-k）
    for order in (2, 3, 4, 5):
        cfg.append(dict(model=f"word.kn{order}.npz", max_words=45, temperature=1.0, top_k=0,
                        seed=2024, tag=f"词级{order}-gram/KN/T=1.0"))
    # 平滑方法对比
    cfg.append(dict(model="word.sb3.a04.npz", max_words=45, temperature=1.0, top_k=0,
                    seed=2024, tag="词级3-gram/StupidBackoff(α=0.4)/T=1.0"))
    cfg.append(dict(model="word.sb3.a10.npz", max_words=45, temperature=1.0, top_k=0,
                    seed=2024, tag="词级3-gram/StupidBackoff(α=1.0)/T=1.0"))
    # 温度 / top-k 对比（5-gram）
    for T, k in ((1.0, 0), (0.8, 0), (0.7, 5), (1.0, 20), (1.3, 0)):
        cfg.append(dict(model="word.kn5.npz", max_words=45, temperature=T, top_k=k,
                        seed=7, tag=f"词级5-gram/KN/T={T}/top-k={k}"))
    # 剪枝模型
    cfg.append(dict(model="word.kn3.prune2.npz", max_words=45, temperature=0.8, top_k=5,
                    seed=7, tag="词级3-gram/KN/剪枝min_count=2/T=0.8/top-k=5"))
    # 字级
    cfg.append(dict(model="char.kn3.npz", max_words=60, temperature=1.0, top_k=0,
                    seed=7, tag="字级3-gram/KN/T=1.0"))
    cfg.append(dict(model="char.kn5.npz", max_words=60, temperature=0.8, top_k=5,
                    seed=7, tag="字级5-gram/KN/T=0.8/top-k=5"))
    # 解码方式
    cfg.append(dict(model="word.kn5.npz", max_words=45, temperature=1.0, top_k=0,
                    seed=7, mode="greedy", tag="词级5-gram/KN/贪心解码"))
    cfg.append(dict(model="word.kn5.npz", max_words=45, temperature=1.0, top_k=0,
                    seed=7, mode="beam", beam=5, tag="词级5-gram/KN/beam=5"))
    return cfg


def run_generation(cfgs: list[dict], jsonl: str, md_path: str) -> None:
    cache: dict[str, tuple] = {}
    rows = []
    for cfg in cfgs:
        path = os.path.join(MODELS, cfg["model"])
        if not os.path.exists(path):
            N.log(f"[生成] 缺模型 {cfg['model']}，跳过")
            continue
        if cfg["model"] not in cache:
            cache[cfg["model"]] = N.NgramLM.load(path)
        lm, meta = cache[cfg["model"]]
        vocab_set = set(lm.vocab[3:])
        toks = G.segment_maxmatch(PREFIX, vocab_set, 7)
        word2id = {w: i for i, w in enumerate(lm.vocab)}
        prefix_ids = [word2id.get(t, word2id[N.UNK]) for t in toks]

        rng = np.random.default_rng(cfg.get("seed", 42))
        t0 = time.time()
        new_ids = lm.generate(prefix_ids, cfg.get("max_words", 45), rng,
                              temperature=cfg.get("temperature", 1.0),
                              top_k=cfg.get("top_k", 0),
                              mode=cfg.get("mode", "sample"),
                              beam=cfg.get("beam", 5),
                              min_words=cfg.get("min_words", 15))
        dt = time.time() - t0
        cont = "".join(lm.vocab[i] for i in new_ids)
        rec = {
            "tag": cfg["tag"],
            "model": cfg["model"],
            "order": meta["order"],
            "smoothing": meta["smoothing"],
            "alpha": meta.get("alpha"),
            "min_count": meta.get("min_count"),
            "mode": cfg.get("mode", "sample"),
            "temperature": cfg.get("temperature", 1.0),
            "top_k": cfg.get("top_k", 0),
            "seed": cfg.get("seed", 42),
            "prefix": PREFIX,
            "prefix_tokens": toks,
            "oov_in_prefix": sum(1 for t in toks if t not in word2id),
            "continuation": cont,
            "text": "".join(toks) + cont,
            "new_tokens": len(new_ids),
            "seconds": round(dt, 3),
        }
        rows.append(rec)
        N.log(f"[生成] {cfg['tag']}：{rec['text'][:60]}…")

    with open(jsonl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# n-gram 文本续写结果\n\n")
        fh.write(f"统一前缀：**{PREFIX}**（按各模型词表用前向最大匹配切分，"
                 f"每个模型的分词结果见下表）\n\n")
        for r in rows:
            fh.write(f"## {r['tag']}\n\n")
            fh.write(f"- 模型：`{r['model']}`（{r['order']}-gram，{r['smoothing']}"
                     + (f"，α={r['alpha']}" if r["smoothing"] == "sb" else "")
                     + (f"，min_count={r['min_count']}" if r.get("min_count", 1) > 1 else "")
                     + "）\n")
            fh.write(f"- 解码：mode={r['mode']}，T={r['temperature']}，"
                     f"top-k={r['top_k']}，seed={r['seed']}，"
                     f"最长 {r['new_tokens']} token，耗时 {r['seconds'] * 1000:.1f} ms\n")
            fh.write(f"- 前缀分词：{' / '.join(r['prefix_tokens'])}"
                     f"（未登录词 {r['oov_in_prefix']} 个）\n")
            fh.write(f"- **续写**：{r['continuation']}\n")
            fh.write(f"- 全文：{r['text']}\n\n")
    N.log(f"生成结果写入 {md_path}")


# --------------------------------------------------------------------------- #
# 汇总成 markdown 表格
# --------------------------------------------------------------------------- #

def render_training_table(jsonl: str, out_md: str) -> None:
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")] if os.path.exists(jsonl) else []
    seen: dict[str, dict] = {}
    for r in rows:                       # 同一 tag 只保留最后一次
        seen[r["tag"]] = r
    rows = list(seen.values())

    def unit_of(r):
        return "字级" if r["train_file"].endswith("char.train.txt") else "词级"

    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("# 训练时间 / 模型规模 / 困惑度 汇总\n\n")
        fh.write("全部为本机（macOS, CPU, numpy 单进程）实测值。"
                 "训练耗时含词表统计、编码、计数、平滑构建；PPL 在按文章切出的 2% 验证集上计算"
                 "（最多 20 万 token）。\n\n")
        fh.write("| 配置 | 粒度 | 阶数 | 平滑 | 分阶剪枝(2..5阶最小计数) | 训练句数 | 训练 token | "
                 "2/3/4/5-gram 类型数 | 训练耗时 s | 其中平滑 s | 峰值内存 MB | 模型 MB | dev PPL |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            types = r.get("ngram_types", {})
            tstr = " / ".join(f"{types.get(str(k), '-')}" for k in (2, 3, 4, 5))
            timing = r.get("timing", {})
            prune = r.get("prune") or [1]
            fh.write("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                r["tag"], unit_of(r), r["order"], r["smoothing"],
                "/".join(str(x) for x in prune),
                r.get("train_sentences", "-"), r.get("train_tokens", "-"), tstr,
                r.get("train_seconds", "-"), timing.get("smooth", "-"),
                r.get("peak_mem_mb", "-"), r.get("model_size_mb", "-"),
                r.get("dev", {}).get("ppl", "-"),
            ))
        fh.write("\n## 各阶段耗时明细（秒）\n\n")
        fh.write("| 配置 | 词表 | 编码 | 滑窗 | 2-gram 计数 | 3-gram 计数 | 4-gram 计数 | "
                 "5-gram 计数 | 平滑构建 | 总计 |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            t = r.get("timing", {})
            fh.write("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                r["tag"], t.get("vocab", "-"), t.get("encode", "-"), t.get("window", "-"),
                t.get("count2", "-"), t.get("count3", "-"), t.get("count4", "-"),
                t.get("count5", "-"), t.get("smooth", "-"), r.get("train_seconds", "-")))
    N.log(f"训练汇总写入 {out_md}")


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description="跑 n-gram 全部实验")
    p.add_argument("--stage", choices=["train", "gen", "all"], default="all")
    p.add_argument("--force", action="store_true", help="已有模型也重新训练")
    p.add_argument("--only", default="", help="只跑 tag 里包含该子串的配置")
    args = p.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    if args.stage in ("train", "all"):
        for cfg in train_configs():
            if args.only and args.only not in cfg["tag"]:
                continue
            run_training(cfg, force=args.force)
        render_training_table(os.path.join(RESULTS, "train_runs.jsonl"),
                              os.path.join(RESULTS, "training_time.md"))

    if args.stage in ("gen", "all"):
        run_generation(gen_configs(),
                       os.path.join(RESULTS, "generation_runs.jsonl"),
                       os.path.join(RESULTS, "generation_samples.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
