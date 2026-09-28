#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三方对比：自研 ngram_lm.py / KenLM / SRILM，用同一份训练语料、同一个验证集，
各自用自己惯常的方式处理词表与 `<unk>`，比较训练时间、模型体积与困惑度。

读表前必须知道的一件事：**三方的 `<unk>` 处理方式不同，PPL 不能只按数字大小比**。

* 自研实现把词表截到 65533（16-bit id 上限），其余词型映射成 `<unk>`，
  `<unk>` 因此是个高频词（约占训练 token 的 1.2%），猜中 OOV 的代价较小；
* KenLM 默认保留全部词型，并**插值一元**，把 `<unk>` 概率压得很小
  （它的 `--interpolate_unigrams 0` 才是 SRILM/SRI 那种"给 `<unk>` 大质量"的行为）；
* SRILM 默认同样保留全部词型，`<unk>` 的质量介于两者之间；
* 词表大小和未知词概率共同改变预测事件；OOV 更少不保证困惑度更低。
  所以本脚本同时记录每个工具的 OOV 数，并额外跑一个 KenLM 的 `--interpolate_unigrams 0`
  变体作为受控对照。

    KENLM=~/ngram-toolchain/src/kenlm SRILM=~/ngram-toolchain/src/srilm \
        python compare_toolchains.py --order 3 5

输出：results/toolchain_comparison.md + results/toolchain_runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time

import ngram_lm as N

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")
LOGS = os.path.join(RESULTS, "logs")


def peak_mem_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024), 1) if sys.platform == "darwin" else round(rss / 1024, 1)


def run_cmd(cmd: list[str], log_path: str, merge_output: bool = False) -> tuple[int, float]:
    """执行外部命令并计时，输出落盘。

    merge_output=False：stderr 落盘（lmplz / ngram-count 的训练进度走 stderr），stdout 丢弃；
    merge_output=True ：stdout+stderr 都落盘（SRILM `ngram -ppl` 的结果走 stdout）。
    """
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        if merge_output:
            proc = subprocess.run(cmd, stdout=log, stderr=log, text=True)
        else:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log, text=True)
    return proc.returncode, round(time.time() - t0, 2)


def corpus_stats(train_path: str, dev_path: str) -> dict:
    def stat(path: str) -> dict:
        n_tok = n_lines = n_types = 0
        types: set[str] = set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                toks = line.split()
                if not toks:
                    continue
                n_tok += len(toks)
                n_lines += 1
                types.update(toks)
        return {"tokens": n_tok, "lines": n_lines, "types": len(types)}

    return {"train": stat(train_path), "dev": stat(dev_path)}


def mine(train_path: str, dev_path: str, order: int, tag: str,
         prune: list[int] | None = None) -> dict:
    """自研实现：训练 + 在固定词表 dev 上算困惑度（不限 token 数）。"""
    t0 = time.time()
    lm, stats = N.train(train_path, order=order, smoothing="kn", prune=prune, verbose=False)
    train_s = round(time.time() - t0, 2)
    word2id = {w: i for i, w in enumerate(lm.vocab)}
    dev = N.encode_file(dev_path, word2id)
    ppl = lm.perplexity(dev, max_tokens=0)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        size_mb = None
        if stats.get("ngram_types"):
            path = os.path.join(tmp, "m.npz")
            lm.save(path, stats)
            size_mb = round(os.path.getsize(path) / 1048576, 1)
    return {
        "tool": "自研 numpy", "tag": tag, "order": order,
        "smoothing": "插值 KN（单折扣）" + ("＋剪枝" if prune else ""),
        "train_seconds": train_s, "peak_mem_mb": peak_mem_mb(), "model_mb": size_mb,
        "ppl": ppl["ppl"], "ppl_excl_oov": None, "oov": ppl["oov"], "tokens": ppl["tokens"],
        "ngram_types": stats["ngram_types"],
    }


def kenlm(kenlm_dir: str, train_path: str, dev_path: str, order: int, tag: str,
          interpolate_unigrams: bool = True, prune: str = "", quantize: bool = False) -> dict:
    bin_dir = os.path.join(kenlm_dir, "build", "bin")
    arpa = os.path.join(MODELS, f"cmp.kenlm.{tag}.arpa")
    binary = os.path.join(MODELS, f"cmp.kenlm.{tag}.trie")
    cmd = [os.path.join(bin_dir, "lmplz"), "-o", str(order), "-S", "4G",
           "--text", train_path, "--arpa", arpa]
    if prune:
        cmd += ["--prune"] + prune.split()
    if not interpolate_unigrams:
        cmd += ["--interpolate_unigrams", "0"]
    rc, train_s = run_cmd(cmd, os.path.join(LOGS, f"cmp.kenlm.{tag}.train.log"))
    if rc != 0:
        raise SystemExit(f"lmplz 失败：{tag}")
    build_cmd = [os.path.join(bin_dir, "build_binary")]
    if quantize:
        build_cmd += ["-q", "8", "-b", "8"]
    build_cmd += ["trie", arpa, binary]
    rc, build_s = run_cmd(build_cmd, os.path.join(LOGS, f"cmp.kenlm.{tag}.build.log"))
    if rc != 0:
        raise SystemExit(f"build_binary 失败：{tag}")
    qlog = os.path.join(LOGS, f"cmp.kenlm.{tag}.query.log")
    with open(dev_path, encoding="utf-8") as fin, open(qlog, "w") as errlog:
        t0 = time.time()
        proc = subprocess.run([os.path.join(bin_dir, "query"), "-v", "summary", binary],
                              stdin=fin, stdout=subprocess.PIPE, stderr=errlog, text=True)
        ppl_s = round(time.time() - t0, 2)
    out = proc.stdout
    return {
        "tool": "KenLM", "tag": tag, "order": order,
        "smoothing": ("修改版 KN（插值一元）" if interpolate_unigrams else "修改版 KN（SRI 式一元）")
                    + (f"，剪枝 {prune}" if prune else "") + ("，8-bit 量化" if quantize else ""),
        "train_seconds": train_s, "build_seconds": build_s,
        "model_mb": round(os.path.getsize(binary) / 1048576, 1),
        "arpa_mb": round(os.path.getsize(arpa) / 1048576, 1),
        "ppl": float(re.search(r"Perplexity including OOVs:\s*([0-9.]+)", out).group(1)),
        "ppl_excl_oov": float(re.search(r"Perplexity excluding OOVs:\s*([0-9.]+)", out).group(1)),
        # 注意：必须用 ^OOVs: 行首锚定，否则会匹配到
        # "Perplexity including OOVs: 254.0…" 里的那个数字
        "oov": int(re.search(r"^OOVs:\s*(\d+)", out, re.M).group(1)),
        "tokens": int(re.search(r"^Tokens:\s*(\d+)", out, re.M).group(1)),
        "ppl_seconds": ppl_s,
    }


def srilm(srilm_dir: str, train_path: str, dev_path: str, order: int, tag: str,
          discount: str = "-kndiscount -interpolate", prune: str = "") -> dict:
    ng_count = os.path.join(srilm_dir, "bin", "ngram-count")
    ng = os.path.join(srilm_dir, "bin", "ngram")
    arpa = os.path.join(MODELS, f"cmp.srilm.{tag}.arpa")
    cmd = [ng_count, "-order", str(order), "-text", train_path] + discount.split()
    if prune:
        cmd += ["-prune", prune]
    cmd += ["-gtmin", "1", "-lm", arpa]
    rc, train_s = run_cmd(cmd, os.path.join(LOGS, f"cmp.srilm.{tag}.train.log"))
    if rc != 0:
        raise SystemExit(f"ngram-count 失败：{tag}")
    rc, ppl_s = run_cmd([ng, "-order", str(order), "-lm", arpa, "-ppl", dev_path],
                        os.path.join(LOGS, f"cmp.srilm.{tag}.ppl.log"), merge_output=True)
    if rc != 0:
        raise SystemExit(f"ngram -ppl 失败：{tag}")
    text = open(os.path.join(LOGS, f"cmp.srilm.{tag}.ppl.log"), encoding="utf-8").read()
    m = re.search(r"(\d+) sentences, (\d+) words, (\d+) OOVs", text)
    logprob = float(re.search(r"logprob=\s*(-?[0-9.]+)", text).group(1))     # 以 10 为底
    tool_ppl = float(re.search(r"ppl=\s*([0-9.]+)", text).group(1))          # SRILM 自报值
    words, sents = int(m.group(2)), int(m.group(1))
    # 仅重算分母为"词数 + 句末 </s> 数"；闭词表评测跳过零概率 OOV，不能补回其 logprob。
    # （SRILM 自报的 ppl 在存在 OOV 时会把 OOV 从分母里扣掉，故与这里不同，一并保留。）
    tokens_common = words + sents
    ppl_common = round(10 ** (-logprob / tokens_common), 2)
    return {
        "tool": "SRILM", "tag": tag, "order": order,
        "smoothing": ("修改版 KN" if "-kndiscount" in discount else "原始 KN（单折扣）")
                    + (f"，熵剪枝 -prune {prune}" if prune else ""),
        "train_seconds": train_s,
        "model_mb": round(os.path.getsize(arpa) / 1048576, 1),
        "ppl": ppl_common, "tool_ppl": tool_ppl, "ppl_excl_oov": None,
        "oov": int(m.group(3)), "tokens": tokens_common, "sentences": sents, "words": words,
        "ppl_seconds": ppl_s,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="三方（自研/KenLM/SRILM）同语料对比（词表与 OOV 策略不同）")
    p.add_argument("--vocab-model", default=os.path.join(MODELS, "word.kn5.npz"))
    p.add_argument("--order", type=int, nargs="+", default=[3, 5])
    p.add_argument("--kenlm", default=os.environ.get("KENLM", ""))
    p.add_argument("--srilm", default=os.environ.get("SRILM", ""))
    p.add_argument("--skip-kenlm", action="store_true")
    p.add_argument("--skip-srilm", action="store_true")
    p.add_argument("--with-pruning", action="store_true",
                   help="额外跑剪枝/量化变体（自研 prune、KenLM --prune/-q、SRILM 熵剪枝）")
    p.add_argument("--out", default=os.path.join(RESULTS, "toolchain_comparison.md"))
    args = p.parse_args()

    os.makedirs(LOGS, exist_ok=True)
    lm, _meta = N.NgramLM.load(args.vocab_model)
    train_path = os.path.join(CORPUS, "word.train.txt")
    dev_path = os.path.join(CORPUS, "word.dev.txt")
    vs = corpus_stats(train_path, dev_path)
    N.log(f"语料：{os.path.basename(train_path)} / {os.path.basename(dev_path)}")
    N.log(f"  训练 {vs['train']['lines']} 句 / {vs['train']['tokens']} token / "
          f"{vs['train']['types']} 词型；验证 {vs['dev']['tokens']} token")
    N.log(f"  自研实现的建模词表（截断后）：{len(lm.vocab)}")

    rows: list[dict] = []
    for order in args.order:
        tag = f"word.o{order}"
        N.log(f"=== {order}-gram：自研 numpy ===")
        rows.append(mine(train_path, dev_path, order, tag))
        if args.kenlm and not args.skip_kenlm:
            N.log(f"=== {order}-gram：KenLM（默认插值一元）===")
            rows.append(kenlm(args.kenlm, train_path, dev_path, order, f"{tag}.kenlm.interp"))
            N.log(f"=== {order}-gram：KenLM（SRI 式一元，<unk> 质量更大）===")
            rows.append(kenlm(args.kenlm, train_path, dev_path, order, f"{tag}.kenlm.sriunk",
                              interpolate_unigrams=False))
        if args.srilm and not args.skip_srilm:
            N.log(f"=== {order}-gram：SRILM（修改版 KN）===")
            rows.append(srilm(args.srilm, train_path, dev_path, order, f"{tag}.srilm.kn"))
            N.log(f"=== {order}-gram：SRILM（原始 KN，单折扣）===")
            rows.append(srilm(args.srilm, train_path, dev_path, order, f"{tag}.srilm.ukn",
                              discount="-ukndiscount -interpolate"))

    if args.with_pruning:
        order = args.order[-1]
        tag = f"word.o{order}"
        # 三家的"剪掉最高两阶的 singleton"写法：自研用最小保留计数，KenLM 用 --prune 阈值
        if order >= 4:
            mine_vec = [1] * (order - 3) + [2, 2]
            kenlm_vec = " ".join(["0"] * (order - 3) + ["1", "1"])
        else:
            mine_vec = [1, 2]
            kenlm_vec = "0 1"
        N.log(f"=== {order}-gram：剪枝/量化变体 ===")
        N.log(f"  [自研] prune={mine_vec} …")
        rows.append(mine(train_path, dev_path, order, f"{tag}.mine.prune", prune=mine_vec))
        if args.kenlm and not args.skip_kenlm:
            N.log(f"  [KenLM] --prune {kenlm_vec} …")
            rows.append(kenlm(args.kenlm, train_path, dev_path, order, f"{tag}.kenlm.prune",
                              prune=kenlm_vec))
            N.log("  [KenLM] 8-bit 量化（不剪枝）…")
            rows.append(kenlm(args.kenlm, train_path, dev_path, order, f"{tag}.kenlm.q8",
                              quantize=True))
        if args.srilm and not args.skip_srilm:
            N.log("  [SRILM] 熵剪枝 -prune 1e-7 / 1e-6 …")
            rows.append(srilm(args.srilm, train_path, dev_path, order, f"{tag}.srilm.ep7",
                              prune="1e-7"))
            rows.append(srilm(args.srilm, train_path, dev_path, order, f"{tag}.srilm.ep6",
                              prune="1e-6"))

    with open(os.path.join(RESULTS, "toolchain_runs.jsonl"), "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# 自研实现 / KenLM / SRILM 三方对比\n\n")
        fh.write(f"三方使用同一份语料（`{os.path.basename(train_path)}`，"
                 f"{vs['train']['lines']} 句 / {vs['train']['tokens']} token / {vs['train']['types']} 词型）"
                 f"与同一份验证集（`{os.path.basename(dev_path)}`，{vs['dev']['tokens']} token）。"
                 f"自研实现受 16-bit id 限制，建模词表截断为 {len(lm.vocab)} 个词型，其余词型映射为 `<unk>`；"
                 f"KenLM/SRILM 保留全部词型。**因此三方验证集 OOV 数不同，PPL 需结合 OOV 列一起看**："
                 f"词表大小与未知词概率都会改变预测事件；OOV 更少不保证 PPL 更低。\n\n")
        fh.write("评测口径：自研/KenLM 含 OOV 与 EOS；SRILM 仅按词数 + 句末数重算分母，"
                 "没有补回零概率 OOV 的代价，尚不可严格横比。\n\n")
        fh.write("| 工具 | 模型 | 平滑 | 训练耗时 s | 模型体积 MB | 验证 token | OOV | "
                 "PPL / SRILM 分母重算值 | 工具自报 PPL |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            tool_ppl = r.get("tool_ppl")
            fh.write("| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
                r["tool"], f"{r['order']}-gram", r["smoothing"], r["train_seconds"],
                r["model_mb"] if r["model_mb"] else "-", r.get("tokens", "-"), r["oov"],
                f"{r['ppl']:.2f}", f"{tool_ppl:.2f}" if tool_ppl else "-"))
    N.log(f"对比表写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
