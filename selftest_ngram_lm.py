#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对自研 n-gram 实现做正确性自检：拿一个"字典 + 元组"的朴素参考实现逐项对照。

    python selftest_ngram_lm.py

覆盖：n-gram 计数、前缀区间查询、插值 Kneser-Ney 概率、Stupid Backoff 概率、
困惑度、以及各阶分布是否归一化。
"""

from __future__ import annotations

import math
import sys
import tempfile
from collections import Counter

import numpy as np

import ngram_lm as N


# --------------------------------------------------------------------------- #
# 朴素参考实现（慢，但直接照公式写）
# --------------------------------------------------------------------------- #
class RefLM:
    def __init__(self, sents, vocab, order, alpha=0.4):
        self.order, self.alpha = order, alpha
        self.V = len(vocab)
        self.c = {k: Counter() for k in range(1, order + 1)}
        for ids in sents:
            padded = [N.ID_BOS] * (order - 1) + list(map(int, ids)) + [N.ID_EOS]
            for k in range(2, order + 1):
                for i in range(order - 1, len(padded)):
                    self.c[k][tuple(padded[i - k + 1:i + 1])] += 1
        for ids in sents:
            self.c[1].update(int(x) for x in ids)
            self.c[1][N.ID_EOS] += 1
            self.c[1][N.ID_BOS] += 1
        self.cc = {}
        for m in range(1, order):
            cc = Counter()
            for g in self.c[m + 1]:
                cc[g[1:]] += 1
            self.cc[m] = cc
        self.cc_total = sum(self.cc[1].values())
        self.D = {order: self._disc(self.c[order].values())}
        for m in range(1, order):
            self.D[m] = self._disc(self.cc[m].values())

    @staticmethod
    def _disc(counts):
        n1 = sum(1 for c in counts if c == 1)
        n2 = sum(1 for c in counts if c == 2)
        return n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.5

    def _level(self, k, sub):
        """level k 的 (计数函数, 分母, 不同的后继数)。

        分母按教科书写法取"该上下文所有后继计数之和"
        Σ_{w} c(hw)（低阶用 continuation count），这样 λ = D·N1+(h•)/Σ 才恒 ≤ 1。
        """
        table = self.c[k] if k == self.order else self.cc[k]
        count = table.get
        denom = sum(c for g, c in table.items() if g[:-1] == sub)
        ncont = sum(1 for g in table if g[:-1] == sub)
        return count, denom, ncont

    def base(self, w):
        return self.cc[1].get((w,), 0) / max(self.cc_total, 1)

    def prob(self, w, ctx):
        level = min(self.order, len(ctx) + 1)
        p = self.base(w)
        for k in range(2, level + 1):
            sub = tuple(ctx[-(k - 1):])
            count, denom, ncont = self._level(k, sub)
            if denom <= 0:
                continue
            d = self.D[k]
            p = max(count(sub + (w,), 0) - d, 0) / denom + d * ncont / denom * p
        return p

    def sb(self, w, ctx):
        level = min(self.order, len(ctx) + 1)
        score = 1.0
        total = sum(self.c[1].values())
        for k in range(level, 1, -1):
            sub = tuple(ctx[-(k - 1):])
            if k - 1 == 1:
                c_ctx = self.c[1].get(sub[0], 0)
            else:
                c_ctx = self.c[k - 1].get(sub, 0)
            if c_ctx > 0:
                c = self.c[k].get(sub + (w,), 0)
                if c > 0:
                    return score * c / c_ctx
            score *= self.alpha
        return score * self.c[1].get(w, 0) / total

    def ppl(self, sents):
        nll = n_tok = 0
        for ids in sents:
            hist = [N.ID_BOS] * (self.order - 1)
            for w in list(map(int, ids)) + [N.ID_EOS]:
                nll -= math.log(max(self.prob(w, hist), 1e-300))
                n_tok += 1
                hist = (hist + [w])[-(self.order - 1):]
        return math.exp(nll / n_tok)


# --------------------------------------------------------------------------- #
# 主自检
# --------------------------------------------------------------------------- #
def main() -> int:
    rng = np.random.default_rng(0)
    # 小型人造语料：词表小、重复多，能同时覆盖 OOV / 一元 / 二元 / 三元
    vocab_words = ["天气", "很好", "我们", "去", "公园", "散步", "他", "说", "明天", "下雨", "。"]
    sents_words = [
        "天气 很好 。 我们 去 公园 散步 。",
        "他 说 明天 下雨 。",
        "天气 很好 。 我们 去 公园 。",
        "我们 去 公园 散步 。 天气 很好 。",
        "他 说 天气 很好 。",
        "明天 下雨 。 我们 去 公园 散步 。",
        "天气 很好 。 他 说 明天 下雨 。",
        "我们 去 公园 。 他 说 明天 下雨 。",
    ]
    tmp = tempfile.mkdtemp()
    train_path = f"{tmp}/train.txt"
    with open(train_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sents_words) + "\n")

    order = 3
    lm, stats = N.train(train_path, order=order, smoothing="kn", verbose=False)
    word2id = {w: i for i, w in enumerate(lm.vocab)}
    sents = [np.asarray([word2id[w] for w in s.split()], dtype=np.int32) for s in sents_words]
    ref = RefLM(sents, lm.vocab, order)

    ok = True

    # 1) 计数比对 ------------------------------------------------------------
    wins = N.sentence_windows(sents, order)
    for k in range(2, order + 1):
        grams = np.ascontiguousarray(wins[:, order - k:])
        packed = N.pack_grams(grams)
        keys, cnts = np.unique(packed, axis=0, return_counts=True)
        assert keys.shape[0] == lm.tables[k].size, f"{k}-gram 类型数不一致"
        mism = 0
        for row, c in zip(keys, cnts):
            ids = [int((row[j // 4] >> np.uint64(16 * (3 - j % 4))) & np.uint64(0xFFFF))
                   for j in range(k)]
            if lm.tables[k].lookup(ids) != int(c):
                mism += 1
        ref_cnt = sum(ref.c[k].values())
        got = int(lm.tables[k].counts.sum())
        print(f"{k}-gram：类型 {lm.tables[k].size}（参考 {len(ref.c[k])}），"
              f"token 数 {got}（参考 {ref_cnt}），查找错误 {mism}")
        ok &= (mism == 0 and lm.tables[k].size == len(ref.c[k]) and got == ref_cnt)

    # 2) KN 概率逐项比对 -----------------------------------------------------
    max_err = 0.0
    for ids in sents:
        hist: list[int] = [N.ID_BOS] * (order - 1)
        for w in list(map(int, ids)) + [N.ID_EOS]:
            a = math.exp(lm.logprob(w, hist))
            b = ref.prob(w, hist)
            max_err = max(max_err, abs(a - b))
            hist = (hist + [w])[-(order - 1):]
    print(f"KN  概率最大绝对误差 = {max_err:.3e}")
    ok &= max_err < 1e-10

    # 3) 困惑度比对 ----------------------------------------------------------
    lm_ppl = lm.perplexity(sents)["ppl"]
    ref_ppl = ref.ppl(sents)
    print(f"KN  困惑度：实现 {lm_ppl:.6f} / 参考 {ref_ppl:.6f}")
    ok &= abs(lm_ppl - ref_ppl) < 0.01        # stats 里 ppl 保留两位小数

    # 4) SB 概率比对 ---------------------------------------------------------
    lm_sb, _ = N.train(train_path, order=order, smoothing="sb", alpha=0.4, verbose=False)
    ref_sb = RefLM(sents, lm_sb.vocab, order, alpha=0.4)
    max_err = 0.0
    for ids in sents:
        hist = [N.ID_BOS] * (order - 1)
        for w in list(map(int, ids)) + [N.ID_EOS]:
            max_err = max(max_err, abs(math.exp(lm_sb.logprob(w, hist)) - ref_sb.sb(w, hist)))
            hist = (hist + [w])[-(order - 1):]
    print(f"SB  概率最大绝对误差 = {max_err:.3e}")
    ok &= max_err < 1e-10

    # 5) 各阶分布归一分 ------------------------------------------------------
    for ctx in (["天气"], ["我们", "去"], ["他", "说"]):
        cid = [word2id[w] for w in ctx]
        for name, model in (("kn", lm), ("sb", lm_sb)):
            ids, p = model.distribution(cid, temperature=1.0, top_k=0)
            s = float(p.sum())
            print(f"分布归一 {name} ctx={ctx}: Σp={s:.8f}, 候选 {len(ids)}")
            ok &= abs(s - 1.0) < 1e-8

    # 5b) 逐阶归一化：KN 的每一阶插值分布对全词表求和都必须等于 1
    #     （这正是"分母要用同一张表里所有后继计数之和"的原因，不然 λ 会 >1）
    for ctx in ([word2id["天气"]], [word2id["我们"], word2id["去"]]):
        for level in (1, 2, 3):
            if level > order or level > len(ctx) + 1:
                continue
            tot = sum(lm._lower_prob(level, w, ctx) for w in range(lm.vocab_size))
            print(f"第 {level} 阶分布 Σ_w P(w|{ctx}) = {tot:.8f}")
            ok &= abs(tot - 1.0) < 1e-8
        # 各阶混合之后（归一化之前）总质量也必须是 1：既不能漏掉回退质量，
        # 也不能把同一批词重复计入（早先版本这里会到 1.4）
        vec = np.zeros(lm.vocab_size)
        reach = lm._kn_distribution(ctx, lm.order, vec)
        total = vec.sum() + reach
        print(f"混合分布（未归一）总质量 = {total:.8f}")
        ok &= abs(total - 1.0) < 1e-8

    # 6) 生成流程可用 --------------------------------------------------------
    g = np.random.default_rng(1)
    out = lm.generate([word2id["天气"]], 6, g, temperature=0.8, top_k=5)
    text = " ".join(lm.vocab[i] for i in out)
    print(f"生成示例：天气 {text}")
    ok &= len(out) > 0

    print("\n自检结果：", "全部通过 ✅" if ok else "存在失败 ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
