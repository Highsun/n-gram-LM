#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 results/ 里的实验记录画成图（PNG，默认 150 dpi，输出到 results/figures/）。

    python plot_results.py                 # 画全部能找到数据的图
    python plot_results.py --only 3 5      # 只画第 3、5 张

依赖：matplotlib（可选 seaborn 美化），无需其他第三方库。
数据来源（缺哪个就跳过对应的图）：
    results/ppl_full.json            由 ppl_table.py 生成（各模型在完整验证集上的 PPL）
    results/train_runs.jsonl         由 run_experiments.py 生成（训练耗时/内存/体积/类型数）
    results/toolchain_runs.jsonl     由 compare_toolchains.py 生成（三方对比）
    results/generation_sweep.jsonl   由 gen_sweep.py 生成（温度/top-k 扫描）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# matplotlib 默认会在 ~/.matplotlib 建缓存；若该目录不可写（沙箱/容器里常见）就用临时目录
_cache = os.path.join(os.path.expanduser("~"), ".matplotlib")
if not os.access(os.path.dirname(_cache) or "/", os.W_OK) or (
        os.path.exists(_cache) and not os.access(_cache, os.W_OK)):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGDIR = os.path.join(RESULTS, "figures")

WORD, CHAR = "word", "char"
ORDER_CN = {1: "1-gram", 2: "2-gram", 3: "3-gram", 4: "4-gram", 5: "5-gram"}


def setup_style() -> str:
    """挑一个系统里可用的中文字体，避免图里出现方框。"""
    prefer = ["PingFang SC", "Heiti TC", "Heiti SC", "Songti SC", "STHeiti",
              "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in prefer if f in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.sans-serif": [chosen, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", rc={"font.sans-serif": [chosen, "DejaVu Sans"],
                                             "axes.unicode_minus": False})
    except Exception:
        pass
    return chosen


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_json(path: str):
    if not os.path.exists(path):
        return None
    return json.load(open(path, encoding="utf-8"))


def save(fig, name: str) -> str:
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  写出 {os.path.relpath(path, HERE)}")
    return path


def pick(rows: list[dict], unit: str, tag: str) -> dict | None:
    return next((r for r in rows if r.get("unit") == unit and r.get("tag") == tag), None)


# --------------------------------------------------------------------------- #
# 图 1：阶数 vs 困惑度，以及高阶 n-gram 的稀疏程度
# --------------------------------------------------------------------------- #
def fig1_order_ppl(ppl: list[dict]) -> None:
    if not ppl:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for unit, color, label in ((WORD, "#4C72B0", "词级"), (CHAR, "#DD8452", "字级")):
        rs = [r for r in ppl if r["unit"] == unit and r["tag"] == f"{unit}.kn{r['order']}"]
        rs.sort(key=lambda r: r["order"])
        if rs:
            axes[0].plot([r["order"] for r in rs], [r["ppl"] for r in rs],
                         "o-", color=color, label=f"{label}（{unit}）")
            for r in rs:
                axes[0].annotate(f"{r['ppl']:.0f}", (r["order"], r["ppl"]),
                                 textcoords="offset points", xytext=(0, 7), ha="center",
                                 fontsize=9, color=color)
    axes[0].set_yscale("log")
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[0].set_xticklabels([ORDER_CN[i] for i in (1, 2, 3, 4, 5)])
    axes[0].set_xlabel("n-gram 阶数")
    axes[0].set_ylabel("验证集困惑度 PPL（对数刻度）")
    axes[0].set_title("阶数越高困惑度越低，但收益递减")
    axes[0].legend()
    axes[0].text(0.02, 0.06, "注意：词级与字级的 token 粒度不同，PPL 不可直接比较\n"
                             "（换算到每字的 bits 见图 7）",
                 transform=axes[0].transAxes, fontsize=8, color="#666666")

    for unit, color in ((WORD, "#4C72B0"), (CHAR, "#DD8452")):
        rs = [r for r in ppl if r["unit"] == unit and r["tag"] == f"{unit}.kn{r['order']}"]
        rs.sort(key=lambda r: r["order"])
        xs, ys = [], []
        for r in rs:
            for k, s in (r.get("singletons") or {}).items():
                if int(k) == r["order"]:
                    xs.append(r["order"])
                    ys.append(s["ratio"] * 100)
        if xs:
            axes[1].plot(xs, ys, "s--", color=color, label=f"{'词' if unit==WORD else '字'}级 n-gram")
    axes[1].set_xticks([2, 3, 4, 5])
    axes[1].set_xticklabels([ORDER_CN[i] for i in (2, 3, 4, 5)])
    axes[1].set_xlabel("n-gram 阶数")
    axes[1].set_ylabel("只出现一次的 n-gram 占比（%）")
    axes[1].set_title("高阶 n-gram 十分稀疏（token 数不变，类型数快速增长）")
    axes[1].legend()
    save(fig, "fig1_order_vs_ppl_and_sparsity.png")


# --------------------------------------------------------------------------- #
# 图 2：训练耗时分解
# --------------------------------------------------------------------------- #
def fig2_stage_timing(runs: list[dict]) -> None:
    rs = [r for r in runs if r.get("tag", "").startswith("word.kn") and r.get("timing")]
    rs = [r for r in rs if r["tag"] in {f"word.kn{o}" for o in (1, 2, 3, 4, 5)}]
    if not rs:
        return
    rs.sort(key=lambda r: r["order"])
    stages = [("vocab", "词表统计"), ("encode", "语料编码"), ("window", "滑窗"),
              ("count2", "2-gram 计数"), ("count3", "3-gram 计数"), ("count4", "4-gram 计数"),
              ("count5", "5-gram 计数"), ("smooth", "KN 平滑构建")]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bottoms = [0.0] * len(rs)
    xs = [ORDER_CN[r["order"]] for r in rs]
    cmap = plt.get_cmap("tab20")
    for i, (key, label) in enumerate(stages):
        vals = [float(r["timing"].get(key, 0) or 0) for r in rs]
        if max(vals) <= 0:
            continue
        ax.bar(xs, vals, bottom=bottoms, label=label, color=cmap(i % 20))
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, b, r in zip(xs, bottoms, rs):
        ax.text(x, b + 0.3, f"{b:.1f}s", ha="center", fontsize=9)
    ax.set_xlabel("模型阶数（词级，全量语料）")
    ax.set_ylabel("训练耗时（秒）")
    ax.set_title("训练耗时分解：计数阶段恒定，平滑构建随阶数增长")
    ax.legend(ncol=3, fontsize=9)
    save(fig, "fig2_training_time_breakdown.png")


# --------------------------------------------------------------------------- #
# 图 3：内存与模型体积
# --------------------------------------------------------------------------- #
def fig3_resource(ppl: list[dict], runs: list[dict]) -> None:
    rs = []
    for unit in (WORD, CHAR):
        for o in (1, 2, 3, 4, 5):
            row = pick(ppl, unit, f"{unit}.kn{o}")
            if row:
                rs.append(row)
    if not rs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for unit, color, label in ((WORD, "#4C72B0", "词级"), (CHAR, "#DD8452", "字级")):
        sub = sorted([r for r in rs if r["unit"] == unit], key=lambda r: r["order"])
        if not sub:
            continue
        xs = [r["order"] for r in sub]
        axes[0].plot(xs, [r["peak_mem_mb"] / 1024 for r in sub], "o-", color=color, label=label)
        axes[1].plot(xs, [r["model_mb"] for r in sub], "o-", color=color, label=label)
    axes[0].set_xlabel("n-gram 阶数")
    axes[0].set_ylabel("训练峰值内存（GB）")
    axes[0].set_title("峰值内存随阶数增长")
    axes[0].set_xticks([1, 2, 3, 4, 5])
    axes[1].set_xlabel("n-gram 阶数")
    axes[1].set_ylabel("模型文件体积（MB，npz 压缩后）")
    axes[1].set_title("模型体积随阶数增长（近似平方）")
    axes[1].set_xticks([1, 2, 3, 4, 5])
    for ax in axes:
        ax.legend()
    save(fig, "fig3_memory_and_model_size.png")


# --------------------------------------------------------------------------- #
# 图 4：语料规模 vs 困惑度 / 训练时间
# --------------------------------------------------------------------------- #
def fig4_scaling(runs: list[dict]) -> None:
    base = next((r for r in runs if r.get("tag") == "word.kn3"), None)
    parts = [r for r in runs if r.get("tag", "").startswith("word.kn3.") and r["tag"].endswith("pct")]
    if not base or not parts:
        return
    rows = sorted(parts + [base], key=lambda r: r.get("train_tokens") or 0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [r["train_tokens"] / 1e6 for r in rows]
    ax.plot(x, [r["dev"]["ppl"] for r in rows], "o-", color="#4C72B0", label="困惑度 PPL")
    ax.set_xlabel("训练语料规模（百万 token）")
    ax.set_ylabel("验证集困惑度 PPL", color="#4C72B0")
    ax.tick_params(axis="y", labelcolor="#4C72B0")
    for xi, r in zip(x, rows):
        ax.annotate(f"{r['dev']['ppl']:.0f}", (xi, r["dev"]["ppl"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9,
                    color="#4C72B0")
    ax2 = ax.twinx()
    ax2.plot(x, [r["train_seconds"] for r in rows], "s--", color="#DD8452", label="训练耗时")
    ax2.set_ylabel("训练耗时（秒）", color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    ax2.grid(False)
    ax.set_title("语料规模与困惑度、训练时间的关系（3-gram KN）")
    save(fig, "fig4_corpus_scaling.png")


# --------------------------------------------------------------------------- #
# 图 5：体积-困惑度权衡（剪枝 / 量化）
# --------------------------------------------------------------------------- #
def fig5_pareto(ppl: list[dict], tools: list[dict]) -> None:
    pts = []
    for r in ppl:
        if r["unit"] == WORD and r["smoothing"] != "sb":     # SB 的分数不是归一化概率，不参与权衡图
            pruned = bool(r.get("prune") and any(x > 1 for x in r["prune"]))
            pts.append((r["model_mb"], r["ppl"], r["tag"],
                        "自研-剪枝" if pruned else "自研", r["order"]))
    for r in tools:
        if not r.get("model_mb"):
            continue
        if r["tool"] == "KenLM" and "量化" in r.get("smoothing", ""):
            group = "KenLM-量化"
        elif r["tool"] == "KenLM" and "剪枝" in r.get("smoothing", ""):
            group = "KenLM-剪枝"
        elif r["tool"] == "SRILM" and "熵剪枝" in r.get("smoothing", ""):
            group = "SRILM-熵剪枝"
        elif r["tool"] == "SRILM":
            group = "SRILM"
        else:
            group = "KenLM"
        pts.append((r["model_mb"], r["ppl"], r.get("tag", ""), group, r["order"]))
    if not pts:
        return
    styles = {
        "自研": ("#4C72B0", "o"), "自研-剪枝": ("#7AA6DC", "o"),
        "KenLM": ("#55A868", "s"), "KenLM-剪枝": ("#9BD3A6", "s"), "KenLM-量化": ("#2F6B3F", "s"),
        "SRILM": ("#C44E52", "^"), "SRILM-熵剪枝": ("#E59AA0", "^"),
    }
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for group, (color, marker) in styles.items():
        sub = [p for p in pts if p[3] == group]
        if not sub:
            continue
        ax.scatter([p[0] for p in sub], [p[1] for p in sub], s=70, marker=marker,
                   color=color, label=group, edgecolor="white", linewidth=0.8)
    # 只标注 5-gram 的关键点，并错开偏移，避免文字互相压住
    anno = {"word.kn5": (0, 10), "word.kn5.prune22": (0, -14),
            "word.o5.kenlm.interp": (-6, 9), "word.o5.kenlm.sriunk": (-16, -15),
            "word.o5.kenlm.prune": (4, 8), "word.o5.kenlm.q8": (6, -14),
            "word.o5.srilm.kn": (-10, 9), "word.o5.srilm.ep7": (10, -6),
            "word.o5.srilm.ep6": (10, -6)}
    for mb, p, tag, group, order in pts:
        if tag not in anno:
            continue
        ax.annotate(tag.replace("word.", "").replace(".npz", ""), (mb, p), fontsize=7.5,
                    textcoords="offset points", xytext=anno[tag], color="#444444")
    ax.set_xlabel("模型体积（MB，越小越好）")
    ax.set_ylabel("验证集 PPL（越低越好）")
    ax.set_yscale("log")
    ax.set_yticks([150, 200, 300, 400, 600, 1000, 2000])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(140, 1000)
    ax.set_title("模型体积与困惑度的权衡（标注词级 5-gram 的关键配置）")
    ax.legend(fontsize=9)
    save(fig, "fig5_size_ppl_tradeoff.png")


# --------------------------------------------------------------------------- #
# 图 6：三方工具对比
# --------------------------------------------------------------------------- #
def fig6_tools(tools: list[dict]) -> None:
    if not tools:
        return
    rows = [r for r in tools if r["order"] in (3, 5)]
    # 显式的"配置 -> 记录"映射，避免模糊匹配出错
    def find(order: int, pred) -> dict | None:
        return next((x for x in rows if x["order"] == order and pred(x)), None)

    configs = [
        ("自研(单折扣KN)", lambda x: x["tool"] == "自研 numpy", "#4C72B0"),
        ("KenLM(默认插值一元)", lambda x: x["tool"] == "KenLM" and "SRI" not in x.get("smoothing", ""),
         "#55A868"),
        ("KenLM(SRI式一元)", lambda x: x["tool"] == "KenLM" and "SRI" in x.get("smoothing", ""),
         "#9BD3A6"),
        ("SRILM(修改版KN)", lambda x: x["tool"] == "SRILM" and "原始" not in x.get("smoothing", ""),
         "#C44E52"),
        ("SRILM(原始KN)", lambda x: x["tool"] == "SRILM" and "原始" in x.get("smoothing", ""),
         "#E59AA0"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    metrics = [("ppl", "验证集 PPL（统一口径，含 OOV）", axes[0]),
               ("train_seconds", "训练耗时（秒，含 lmplz / ngram-count 自身统计）", axes[1]),
               ("model_mb", "模型体积（MB：npz / trie / ARPA）", axes[2])]
    for key, title, ax in metrics:
        vals, names, colors = [], [], []
        for order in (3, 5):
            for name, pred, color in configs:
                r = find(order, pred)
                if r is None or r.get(key) is None:
                    continue
                vals.append(r[key])
                names.append(f"{order}-gram\n{name}")
                colors.append(color)
        ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7.5)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}" if v >= 10 else f"{v:.2f}", ha="center",
                    va="bottom", fontsize=8)
    fig.suptitle("自研实现 / KenLM / SRILM：PPL、训练耗时、模型体积（词级 3-gram 与 5-gram）",
                 y=1.02)
    save(fig, "fig6_three_tool_comparison.png")


# --------------------------------------------------------------------------- #
# 图 7：跨粒度可比的信息量（bits/字）
# --------------------------------------------------------------------------- #
def fig7_bits(ppl: list[dict]) -> None:
    if not ppl:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for unit, color, label in ((WORD, "#4C72B0", "词级"), (CHAR, "#DD8452", "字级")):
        rs = sorted([r for r in ppl if r["unit"] == unit and r["tag"] == f"{unit}.kn{r['order']}"],
                    key=lambda r: r["order"])
        if not rs:
            continue
        axes[0].plot([r["order"] for r in rs], [r["bits_per_token"] for r in rs],
                     "o-", color=color, label=label)
        axes[1].plot([r["order"] for r in rs], [r["bits_per_char"] for r in rs],
                     "o-", color=color, label=label)
    axes[0].set_title("每个 token 的信息量（bits/token）")
    axes[1].set_title("换算到每个汉字（bits/字）——跨粒度可比")
    for ax, ylabel in zip(axes, ("bits / token", "bits / 汉字")):
        ax.set_xlabel("n-gram 阶数")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend()
    save(fig, "fig7_bits_per_unit.png")


# --------------------------------------------------------------------------- #
# 图 8：生成参数扫描
# --------------------------------------------------------------------------- #
def fig8_gen_sweep(sweep: list[dict]) -> None:
    if not sweep:
        return
    temps = sorted({r["temperature"] for r in sweep})
    ks = sorted({r["top_k"] for r in sweep})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for k in ks:
        for ax, key, title in ((axes[0], "logprob_per_token", "续写的模型对数概率/ token（越高越像语料）"),
                               (axes[1], "ttr", "去重后的类符/形符比（越高越多样）"),
                               (axes[2], "length", "平均续写长度（token）")):
            ys = []
            for T in temps:
                vals = [r[key] for r in sweep if r["temperature"] == T and r["top_k"] == k]
                ys.append(sum(vals) / len(vals) if vals else float("nan"))
            ax.plot(temps, ys, "o-", label=f"top-k = {k if k else '不限'}")
    for ax, title in zip(axes, ("续写质量：logprob/token", "多样性：类符/形符比（TTR）",
                                "平均长度（token）")):
        ax.set_xlabel("采样温度 temperature")
        ax.set_title(title)
        ax.legend(fontsize=9)
    save(fig, "fig8_generation_sweep.png")


# --------------------------------------------------------------------------- #
# 图 9：<unk> 口径对 PPL 的影响（跨工具比较的前提）
# --------------------------------------------------------------------------- #
def fig9_unk_effect(tools: list[dict]) -> None:
    rows = [r for r in tools if r["order"] in (3, 5)]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels, vals, colors = [], [], []
    color_of = {"自研 numpy": "#4C72B0", "KenLM": "#55A868", "SRILM": "#C44E52"}
    short = {"自研 numpy": "自研",
             "KenLM（默认插值一元）": "KenLM 默认", "KenLM（SRI 式一元）": "KenLM SRI式",
             "SRILM（修改版 KN）": "SRILM 改KN", "SRILM（原始 KN）": "SRILM 原KN"}
    for order in (3, 5):
        for r in rows:
            if r["order"] != order:
                continue
            name = r["tool"]
            if name == "KenLM":
                name += "（SRI 式一元）" if "SRI" in r["smoothing"] else "（默认插值一元）"
            elif name == "SRILM":
                name += "（原始 KN）" if "原始" in r["smoothing"] else "（修改版 KN）"
            if "剪枝" in r["smoothing"] or "量化" in r["smoothing"]:
                continue
            labels.append(f"{order}-gram\n{short.get(name, name)}\nOOV={r['oov']}")
            vals.append(r["ppl"])
            colors.append(color_of.get(r["tool"], "#888"))
    ax.bar(labels, vals, color=colors, width=0.72)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("验证集 PPL（统一口径，含 OOV）")
    ax.set_title("跨工具比较困惑度需要统一词表与 <unk> 口径\n"
                 "（KenLM 关闭一元插值后困惑度明显下降）")
    ax.tick_params(axis="x", labelsize=8, labelrotation=0)
    save(fig, "fig9_unk_effect_on_ppl.png")


# --------------------------------------------------------------------------- #
# 图 10：语料画像（句长分布、词性分布、高频词）
# --------------------------------------------------------------------------- #
def fig10_corpus_profile(ppl: list[dict]) -> None:
    import collections
    stats = load_json(os.path.join(RESULTS, "stats.json")) or load_json(
        os.path.join(HERE, "corpus", "stats.json")) or {}
    lengths = collections.Counter()
    words = collections.Counter()
    with open(os.path.join(HERE, "corpus", "word.train.txt"), encoding="utf-8") as fh:
        for line in fh:
            toks = line.split()
            lengths[min(len(toks), 60)] += 1
            words.update(toks)
    if not lengths:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    xs = sorted(lengths)
    axes[0].bar(xs, [lengths[x] for x in xs], color="#4C72B0", width=0.9)
    axes[0].set_xlabel("句子长度（token，最后一桶为 ≥60）")
    axes[0].set_ylabel("句子数")
    axes[0].set_title(f"句长分布（共 {sum(lengths.values())} 句）")

    tags = list((stats.get("tag_inventory") or {}).items())[:15]
    if tags:
        names = [t for t, _ in tags][::-1]
        vals = [v for _, v in tags][::-1]
        axes[1].barh(names, vals, color="#DD8452")
        axes[1].set_xlabel("出现次数")
        axes[1].set_title("高频词性标签（前 15）")

    top = words.most_common(15)[::-1]
    axes[2].barh([w for w, _ in top], [c for _, c in top], color="#55A868")
    axes[2].set_xlabel("出现次数")
    axes[2].set_title("高频词 / 标点（前 15）")
    fig.suptitle("语料画像：清洗后的《人民日报》1998 训练集", y=1.03)
    save(fig, "fig10_corpus_profile.png")


# --------------------------------------------------------------------------- #
# 图 11：保留词性标签的代价
# --------------------------------------------------------------------------- #
def fig11_tag_ablation(ppl: list[dict]) -> None:
    data = load_json(os.path.join(RESULTS, "tag_ablation.json"))
    if not data:
        return
    c, ent = data["corpus"], data["tag_entropy"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["原始标注\n(含句子编号+词性+括号)", "只保留词性\n(词/词性)", "去标签\n(本项目)"]
    vals = [c["raw_types"], c["tagged_types"], c["word_types"]]
    bars = axes[0].bar(names, vals, color=["#C44E52", "#DD8452", "#4C72B0"])
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom",
                     fontsize=9)
    axes[0].set_ylabel("词型数（token types）")
    axes[0].set_title(f"保留标注让词型数涨到 {vals[0] / vals[2]:.2f}× / {vals[1] / vals[2]:.2f}×")
    axes[0].tick_params(axis="x", labelsize=8)

    b5 = next((r for r in ppl if r["tag"] == "word.kn5"), None)
    word_bits = b5["bits_per_token"] if b5 else 7.53
    bars = axes[1].bar(["每个词的平均信息量\n(词级 5-gram)",
                        "给定词后词性的条件熵\nH(tag|word)"],
                       [word_bits, ent["entropy_bits"]],
                       color=["#4C72B0", "#C44E52"])
    for b, v in zip(bars, [word_bits, ent["entropy_bits"]]):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=10)
    axes[1].set_ylabel("bits / token")
    axes[1].set_title(f"标签信息量很小（常见词性基线准确率 {ent['accuracy']:.1%}），\n"
                      f"但会让 100% 的生成 token 带上标注")
    save(fig, "fig11_tag_ablation.png")


def main() -> int:
    ap = argparse.ArgumentParser(description="把实验结果画成图")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="只画指定编号的图")
    args = ap.parse_args()

    font = setup_style()
    print(f"使用字体：{font}")
    os.makedirs(FIGDIR, exist_ok=True)

    ppl = load_json(os.path.join(RESULTS, "ppl_full.json")) or []
    runs = load_jsonl(os.path.join(RESULTS, "train_runs.jsonl"))
    tools = load_jsonl(os.path.join(RESULTS, "toolchain_runs.jsonl"))
    sweep = load_jsonl(os.path.join(RESULTS, "generation_sweep.jsonl"))
    if sweep:                      # 同 (T,k) 多条样本 -> 先聚合成均值，画图用
        pass

    figs = {
        1: lambda: fig1_order_ppl(ppl),
        2: lambda: fig2_stage_timing(runs),
        3: lambda: fig3_resource(ppl, runs),
        4: lambda: fig4_scaling(runs),
        5: lambda: fig5_pareto(ppl, tools),
        6: lambda: fig6_tools(tools),
        7: lambda: fig7_bits(ppl),
        8: lambda: fig8_gen_sweep(sweep),
        9: lambda: fig9_unk_effect(tools),
        10: lambda: fig10_corpus_profile(ppl),
        11: lambda: fig11_tag_ablation(ppl),
    }
    todo = args.only or sorted(figs)
    for i in todo:
        if i not in figs:
            print(f"  跳过未知编号 {i}")
            continue
        print(f"[图 {i}]")
        try:
            figs[i]()
        except Exception as exc:                       # 缺数据时不让整个脚本挂掉
            print(f"  跳过：{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
