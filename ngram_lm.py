#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自研 n-gram 语言模型（numpy 版参考实现）
=======================================

提供一个**不依赖任何第三方工具、只用 numpy** 的 n-gram 训练 + 推理实现，用来真正跑出
"训练时间对比" 和 "不同参数下的文本续写" 两组结果。

实现要点
--------
* **计数**：把 n-gram 的每个词 id 打包进 16-bit 槽位，一条 n-gram 压成
  ``ceil(n/4)`` 个 ``uint64``（n<=4 只需 1 个），再用 ``np.unique(axis=0)`` 排序去重计数。
  排序后的键天然按词典序排列，于是"给定上下文找所有后继词"就是一次
  ``np.searchsorted`` 区间查询，不需要任何哈希表 / C++ 扩展。
* **平滑**：实现了两种最典型的方法
    - ``kn``  : 插值 Kneser-Ney（SRILM ``-kndiscount`` / KenLM 默认的同类方法）
    - ``sb``  : Stupid Backoff（Goodman 2001，资料 3 里那种"常数回退"）
* **生成**：给定前缀，按上下文长度合并各阶分布后采样，支持 temperature / top-k / 贪心 / beam。

词表上限 65533（id 用 16-bit 表示），语料里其余低频词统一映射成 ``<unk>``。
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# 特殊符号与常量
# --------------------------------------------------------------------------- #

UNK, BOS, EOS = "<unk>", "<s>", "</s>"
ID_UNK, ID_BOS, ID_EOS = 0, 1, 2
MAX_ID = 0xFFFE              # 16-bit 槽位里合法 id 的上界（0xFFFF 留给区间上界）
MAX_WORD_TYPES = 65533       # 词表上限
_MASK16 = np.uint64(0xFFFF)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 16-bit 打包 / 解包
# --------------------------------------------------------------------------- #

def pack_grams(gram_ids: np.ndarray) -> np.ndarray:
    """(M, n) 的 id 矩阵 -> (M, lanes) 的 uint64 打包键（lanes = ceil(n/4)）。"""
    g = np.ascontiguousarray(gram_ids, dtype=np.uint64)
    m, n = g.shape
    lanes = (n + 3) // 4
    out = np.zeros((m, lanes), dtype=np.uint64)
    for j in range(n):
        out[:, j // 4] |= (g[:, j] & _MASK16) << np.uint64(16 * (3 - j % 4))
    return out


def pack_prefix(ids, lanes: int) -> tuple[np.ndarray, np.ndarray]:
    """把上下文 id 打包成区间查询用的 (下界, 上界)。

    未使用的 16-bit 槽位：下界填 0，上界填 0xFFFF —— 于是
    ``[searchsorted(下界), searchsorted(上界))`` 正好是所有以该上下文开头的 n-gram。
    """
    lo = np.zeros(lanes, dtype=np.uint64)
    hi = np.zeros(lanes, dtype=np.uint64)
    n = len(ids)
    for j, v in enumerate(ids):
        lane, shift = j // 4, 16 * (3 - j % 4)
        packed = np.uint64(int(v) & 0xFFFF) << np.uint64(shift)
        lo[lane] |= packed
        hi[lane] |= packed
    for j in range(n, lanes * 4):
        lane, shift = j // 4, 16 * (3 - j % 4)
        hi[lane] |= _MASK16 << np.uint64(shift)
    return lo, hi


def unpack_lanes(lanes: tuple[np.ndarray, ...], order: int) -> np.ndarray:
    """把打包键还原成 (M, order) 的 id 矩阵。"""
    m = lanes[0].shape[0]
    out = np.empty((m, order), dtype=np.int64)
    for j in range(order):
        lane, shift = j // 4, 16 * (3 - j % 4)
        out[:, j] = ((lanes[lane] >> np.uint64(shift)) & _MASK16).astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# 排序 n-gram 表
# --------------------------------------------------------------------------- #

class SortedGramTable:
    """按词典序排好序的 n-gram 计数表，支持「前缀区间」和「精确查找」。"""

    __slots__ = ("order", "lanes", "counts", "size", "n1_before", "n2_before")

    def __init__(self, order: int, lanes, counts: np.ndarray,
                 n1_before: int = 0, n2_before: int = 0):
        self.order = order
        self.lanes = tuple(lanes)
        self.counts = counts
        self.size = int(counts.shape[0])
        # 剪枝前的 count-of-counts：Kneser-Ney 的折扣要用它估计，
        # 否则剪掉全部 singleton 后 n1=0 会让折扣变成 0、回退项消失（PPL 爆炸）。
        self.n1_before = int(n1_before)
        self.n2_before = int(n2_before)

    # -- 构建 ------------------------------------------------------------ #
    @classmethod
    def build(cls, grams: np.ndarray, min_count: int = 1) -> "SortedGramTable":
        packed = pack_grams(grams)
        keys, counts = np.unique(packed, axis=0, return_counts=True)
        del packed
        n1 = int(np.count_nonzero(counts == 1))
        n2 = int(np.count_nonzero(counts == 2))
        if min_count > 1:
            keep = counts >= min_count
            keys, counts = keys[keep], counts[keep]
        lanes = tuple(np.ascontiguousarray(keys[:, j]) for j in range(keys.shape[1]))
        return cls(grams.shape[1], lanes, counts.astype(np.int64), n1, n2)

    @classmethod
    def from_saved(cls, order: int, lanes, counts: np.ndarray,
                   n12=None) -> "SortedGramTable":
        n1, n2 = (int(n12[0]), int(n12[1])) if n12 is not None else (0, 0)
        return cls(order, [np.ascontiguousarray(x) for x in lanes],
                   counts.astype(np.int64), n1, n2)

    # -- 查询 ------------------------------------------------------------ #
    def _range(self, lo_key: np.ndarray, hi_key: np.ndarray) -> tuple[int, int]:
        lo, hi = 0, self.size
        for j, col in enumerate(self.lanes):
            l, h = lo_key[j], hi_key[j]
            if not l and h == np.uint64(0xFFFFFFFFFFFFFFFF):   # 这一 lane 完全自由
                break
            sub = col[lo:hi]
            a = int(np.searchsorted(sub, l, side="left"))
            b = int(np.searchsorted(sub, h, side="right"))
            lo, hi = lo + a, lo + b
            if lo >= hi:
                return lo, hi
        return lo, hi

    def prefix_range(self, ctx_ids) -> tuple[int, int]:
        nlanes = len(self.lanes)
        lo_key, hi_key = pack_prefix(ctx_ids, nlanes)
        return self._range(lo_key, hi_key)

    def lookup(self, gram_ids) -> int:
        """返回该 n-gram 的计数，不存在返回 0。"""
        nlanes = len(self.lanes)
        lo_key, hi_key = pack_prefix(gram_ids, nlanes)   # 全槽位受限 -> 区间即精确匹配
        lo, hi = self._range(lo_key, hi_key)
        return int(self.counts[lo]) if lo < hi else 0

    def continuations(self, ctx_ids) -> tuple[np.ndarray, np.ndarray]:
        """返回 (后继词 id 数组, 对应计数)。ctx_ids 长度为 order-1。"""
        lo, hi = self.prefix_range(ctx_ids)
        if lo >= hi:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        field = self.order - 1
        lane, shift = field // 4, 16 * (3 - field % 4)
        words = ((self.lanes[lane][lo:hi] >> np.uint64(shift)) & _MASK16).astype(np.int64)
        return words, self.counts[lo:hi]

    def prefix_total(self, ctx_ids) -> float:
        """该上下文所有后继计数之和（= 剪枝后仍然自洽的分母）。"""
        lo, hi = self.prefix_range(ctx_ids)
        return float(self.counts[lo:hi].sum()) if lo < hi else 0.0

    def context_lengths(self) -> int:
        return self.order - 1


# --------------------------------------------------------------------------- #
# 语料编码
# --------------------------------------------------------------------------- #

def count_word_types(path: str, encoding: str = "utf-8") -> "dict[str, int]":
    counts: dict[str, int] = {}
    with open(path, encoding=encoding) as fh:
        for line in fh:
            for w in line.split():
                counts[w] = counts.get(w, 0) + 1
    return counts


def build_vocab(word_counts: dict[str, int], max_types: int = MAX_WORD_TYPES,
                min_count: int = 1) -> list[str]:
    """按频次排序取前 max_types 个词，词表 = [<unk>, <s>, </s>, w3, w4, ...]。"""
    items = [(w, c) for w, c in word_counts.items() if c >= min_count and w not in (BOS, EOS, UNK)]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[: max_types - 3]
    return [UNK, BOS, EOS] + [w for w, _ in items]


def encode_file(path: str, word2id: dict[str, int], limit_lines: int = 0,
                encoding: str = "utf-8") -> list[np.ndarray]:
    """把语料文件编码成 list[int32 数组]（每句一个数组）。"""
    unk = word2id[UNK]
    out: list[np.ndarray] = []
    with open(path, encoding=encoding) as fh:
        for line in fh:
            if limit_lines and len(out) >= limit_lines:
                break
            ids = [word2id.get(w, unk) for w in line.split()]
            if ids:
                out.append(np.asarray(ids, dtype=np.int32))
    return out


def sentence_windows(sentences: list[np.ndarray], order: int) -> np.ndarray:
    """把每句补齐 <s> / </s> 后展开成 (M, order) 的 n-gram 窗口矩阵。

    只在句内滑窗：每个真实 token（以及句末 ``</s>``）作为窗口的最后一个元素，
    这样既不会跨句，也天然给出了条件在 <s> 上的那些 n-gram。
    """
    pad = np.full(order - 1, ID_BOS, dtype=np.int32)
    tail = np.asarray([ID_EOS], dtype=np.int32)
    chunks = []
    for ids in sentences:
        arr = np.concatenate([pad, ids, tail])
        chunks.append(np.lib.stride_tricks.sliding_window_view(arr, order))
    return np.concatenate(chunks, axis=0)


def count_unigrams(sentences: list[np.ndarray], vocab_size: int) -> np.ndarray:
    """统计一元计数。

    ``<s>`` 每个句子算一次（它只作为上下文出现，不参与滑窗），``</s>`` 每句一次。
    """
    if sentences:
        all_ids = np.concatenate(sentences)
        counts = np.bincount(all_ids, minlength=vocab_size).astype(np.int64)[:vocab_size]
    else:
        counts = np.zeros(vocab_size, dtype=np.int64)
    counts[ID_EOS] += len(sentences)
    counts[ID_BOS] += len(sentences)
    return counts


# --------------------------------------------------------------------------- #
# 语言模型
# --------------------------------------------------------------------------- #

@dataclass
class NgramLM:
    """n-gram 语言模型（平滑方法可选 kn / sb）。"""

    vocab: list[str]
    tables: dict[int, SortedGramTable]     # order -> 表（order>=2）
    unigram: np.ndarray                    # id -> 原始计数
    cc1: np.ndarray                        # id -> 一阶 continuation count（KN 用）
    cc: dict[int, SortedGramTable] = field(default_factory=dict)   # m -> continuation 表
    smoothing: str = "kn"
    alpha: float = 0.4                     # Stupid Backoff 的回退系数
    order: int = 3

    # ---- 基础量 -------------------------------------------------------- #
    def __post_init__(self) -> None:
        self.vocab_size = len(self.vocab)
        self.total_unigram = float(self.unigram.sum())
        self.base_raw = self.unigram.astype(np.float64) / max(self.total_unigram, 1.0)
        cc1 = self.cc1.astype(np.float64)
        self.base_cont = cc1 / max(cc1.sum(), 1.0)
        self.discounts: dict[int, float] = {}
        if self.smoothing == "kn":
            self._prepare_kn()

    def _prepare_kn(self) -> None:
        # 一阶 continuation 表：由二元表的后一个词统计得到（已在 set_cc 里算好 cc1）
        for k in range(2, self.order):
            self.cc[k] = self._suffix_table(self.tables[k + 1])
        # 折扣估计（Chen & Goodman 的简单估计式 D = n1 / (n1 + 2 n2)），
        # 用"剪枝前"的 count-of-counts，保证剪枝后仍有正常的回退质量
        for k, tab in self.tables.items():
            self.discounts[k] = _discount_from_n12(tab)
        for k, tab in self.cc.items():
            self.discounts[k] = _discount_from_n12(tab)

    @staticmethod
    def _suffix_table(tab: SortedGramTable) -> SortedGramTable:
        """由 (m+1)-gram 表构造 m-gram 的 continuation 计数表。"""
        ids = unpack_lanes(tab.lanes, tab.order)[:, 1:]     # 去掉最左词 = 后缀
        return SortedGramTable.build(ids, min_count=1)

    # ---- 概率 ---------------------------------------------------------- #
    def _level_stats(self, level: int, ctx: list[int]):
        """第 level 阶的"折扣分布 + 回退权重"，返回 None 表示该阶没有可用观测。

        约定（这也是教科书里插值 KN 的标准写法）：

            P_k(w|h) = max(C_k(h,w) - D_k, 0) / S_k(h) + λ_k(h) · P_{k-1}(w|h')
            S_k(h)   = Σ_w C_k(h,w)          # 同一张表里该上下文所有后继计数之和
            λ_k(h)   = 1 - Σ_w max(C_k(h,w)-D_k,0)/S_k(h)   # 直接由损失的分子质量得到

        * 未剪枝时 λ 就等于教科书里的 D_k·N1+(h•)/S_k(h)；
        * 剪枝后如果分母仍取未剪枝的低阶计数，λ 会塌陷（实测 3-gram 剪到 1/5
          大小时 PPL 从三位数涨到 1.9e4），自洽以后只小幅上升；
        * 用 1-分子质量 定义 λ 还能保证各阶分布严格归一（Σ_w P_k = 1）。
        """
        tab = self.tables[level] if level == self.order else self.cc[level]
        words, counts = tab.continuations(ctx)
        if words.size == 0:
            return None
        d = self.discounts[level]
        s = float(counts.sum())
        if s <= 0:
            return None
        disc = np.maximum(counts - d, 0.0)
        num_mass = float(disc.sum()) / s
        lam = max(0.0, 1.0 - num_mass)
        return words, disc, s, num_mass, lam

    def _unigram_count(self, wid: int) -> int:
        return int(self.unigram[wid])

    def _lower_prob(self, level: int, word: int, ctx: list[int]) -> float:
        """精确计算 P_level(word | ctx)，level 从 1（unigram）开始。"""
        if self.smoothing == "sb":
            return self._sb_prob(word, ctx, level)
        p = self._kn_base(word)
        for k in range(2, level + 1):
            sub = ctx[-(k - 1):]
            pieces = self._level_stats(k, sub)
            if pieces is None:
                continue
            _, _, s, _, lam = pieces
            d = self.discounts[k]
            c = self._level_count(k, sub + [word])
            p = max(c - d, 0.0) / s + lam * p
        return p

    def _level_count(self, level: int, gram: list[int]) -> int:
        if level == self.order:
            return self.tables[level].lookup(gram)
        return self.cc[level].lookup(gram)

    def _kn_base(self, word: int) -> float:
        return float(self.base_cont[word])

    def _sb_prob(self, word: int, ctx: list[int], level: int) -> float:
        """Goodman (2001) Stupid Backoff：从 level 阶往下找，命中即返回，未命中乘 alpha。"""
        score = 1.0
        for k in range(level, 1, -1):
            sub = ctx[-(k - 1):]
            c_ctx = self._raw_context_count(k, sub)
            if c_ctx > 0:
                c = self.tables[k].lookup(sub + [word])
                if c > 0:
                    return score * c / c_ctx
            score *= self.alpha
        return score * float(self.base_raw[word])

    def _raw_context_count(self, level: int, ctx: list[int]) -> int:
        """上下文（level-1 个词）在原始计数下的出现次数。"""
        if level - 1 == 0:
            return int(self.total_unigram)
        if level - 1 == 1:
            return self._unigram_count(ctx[0])
        return self.tables[level - 1].lookup(ctx)

    def logprob(self, word: int, ctx: list[int], level: int | None = None) -> float:
        level = min(self.order, len(ctx) + 1) if level is None else level
        if self.smoothing == "sb":
            p = self._sb_prob(word, ctx, level)
        else:
            p = self._lower_prob(level, word, ctx)
        return math.log(max(p, 1e-12))

    def perplexity(self, sentences: list[np.ndarray], add_eos: bool = True,
                   max_tokens: int = 0) -> dict:
        """在验证集上算困惑度（含词表外 <unk> 与句末 </s>）。"""
        nll = 0.0
        n_tok = 0
        n_oov = 0
        hist: list[int] = []
        for ids in sentences:
            hist = [ID_BOS] * (self.order - 1)
            for wid in ids:
                nll -= self.logprob(int(wid), hist)
                n_tok += 1
                n_oov += int(wid == ID_UNK)
                hist = (hist + [int(wid)])[-(self.order - 1):]
                if max_tokens and n_tok >= max_tokens:
                    break
            if not add_eos:
                continue
            nll -= self.logprob(ID_EOS, hist)
            n_tok += 1
            if max_tokens and n_tok >= max_tokens:
                break
        return {
            "tokens": n_tok,
            "oov": n_oov,
            "oov_rate": round(n_oov / max(n_tok, 1), 4),
            "logprob": round(-nll / max(n_tok, 1), 4),
            "ppl": round(math.exp(min(nll / max(n_tok, 1), 700)), 2),
        }

    # ---- 生成 ---------------------------------------------------------- #
    def distribution(self, ctx: list[int], temperature: float = 1.0,
                     top_k: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """返回 (词 id 数组, 概率数组)，概率按 temperature/top_k 处理过。"""
        vec = np.zeros(self.vocab_size, dtype=np.float64)
        top = min(self.order, len(ctx) + 1)
        if self.smoothing == "sb":
            base_weight = self._sb_distribution(ctx, top, vec)
            vec += base_weight * self.base_raw
        else:
            base_weight = self._kn_distribution(ctx, top, vec)
            vec += base_weight * self.base_cont

        if temperature and temperature != 1.0:
            np.power(vec, 1.0 / temperature, out=vec)

        if top_k and top_k > 0:
            if top_k < self.vocab_size:
                cut = np.argpartition(-vec, top_k - 1)[:top_k]
                mask = np.zeros_like(vec, dtype=bool)
                mask[cut] = True
                vec = np.where(mask, vec, 0.0)

        total = vec.sum()
        if total <= 0:
            vec = np.zeros(self.vocab_size)
            vec[ID_EOS] = 1.0
            return np.asarray([ID_EOS]), np.asarray([1.0])
        vec /= total
        ids = np.flatnonzero(vec)
        return ids, vec[ids]

    def _kn_distribution(self, ctx: list[int], top: int, vec: np.ndarray) -> float:
        """按"混合表征"把各阶贡献累加进 vec，返回最后摊给一元分布的质量。

        P(w) = Σ_k Π_{j>k} λ_j · (1-λ_k) · Q_k(w) + Π_j λ_j · Q_1(w)
        其中 Q_k 是第 k 阶"折扣后"的归一化分布（只在被观测到的后继词上有值）。
        这样逐层摊派不会重复计入同一批词，Σ_w P(w) 严格等于 1。
        """
        reach = 1.0
        for level in range(top, 1, -1):
            sub = ctx[-(level - 1):]
            pieces = self._level_stats(level, sub)
            if pieces is None:
                continue
            words, disc, _s, num_mass, lam = pieces
            if num_mass > 0:
                vec[words] += reach * num_mass * (disc / disc.sum())
            reach *= lam
        return reach

    def _sb_distribution(self, ctx: list[int], top: int, vec: np.ndarray) -> float:
        """累加 SB 各阶命中的分数，返回乘到一元分布上的权重。"""
        score = 1.0
        for level in range(top, 1, -1):
            sub = ctx[-(level - 1):]
            c_ctx = self._raw_context_count(level, sub)
            if c_ctx > 0:
                words, counts = self.tables[level].continuations(sub)
                if words.size:
                    vec[words] += score * counts / c_ctx
            score *= self.alpha
        return score

    def sample(self, ctx: list[int], rng: np.random.Generator,
               temperature: float = 1.0, top_k: int = 0,
               forbid: tuple[int, ...] = (ID_BOS, ID_UNK)) -> int:
        ids, probs = self.distribution(ctx, temperature=temperature, top_k=top_k)
        if forbid:
            keep = ~np.isin(ids, np.asarray(forbid))
            if keep.any():
                ids, probs = ids[keep], probs[keep]
                probs = probs / probs.sum()
        return int(rng.choice(ids, p=probs))

    def greedy(self, ctx: list[int], forbid: tuple[int, ...] = (ID_BOS, ID_UNK)) -> int:
        ids, probs = self.distribution(ctx, temperature=1.0, top_k=0)
        order = np.argsort(-probs)
        for i in order:
            if int(ids[i]) not in forbid:
                return int(ids[i])
        return ID_EOS

    def generate(self, prefix: list[int], n_words: int, rng: np.random.Generator,
                 temperature: float = 1.0, top_k: int = 0, mode: str = "sample",
                 beam: int = 1, min_words: int = 0) -> list[int]:
        """给定前缀 id 序列，续写 n_words 个词。

        ``min_words``：前若干个位置禁止 ``</s>``。KN 的一元基分布会把较多质量压给
        ``</s>``（它的"左上下文种类"很多），生成时容易一句话几个词就结束，
        所以默认保留一个最小长度。
        """
        if mode == "greedy":
            ctx = list(prefix)
            out = []
            for _ in range(n_words):
                forbid = (ID_BOS, ID_UNK, ID_EOS) if len(out) < min_words else (ID_BOS, ID_UNK)
                w = self.greedy(ctx, forbid=forbid)
                if w == ID_EOS:
                    break
                out.append(w)
                ctx = (ctx + [w])[-(self.order - 1):]
            return out

        ctx = [ID_BOS] * (self.order - 1) + list(prefix)
        if mode == "beam":
            return self._beam(prefix, n_words, beam)

        out = []
        for _ in range(n_words):
            forbid = (ID_BOS, ID_UNK, ID_EOS) if len(out) < min_words else (ID_BOS, ID_UNK)
            w = self.sample(ctx, rng, temperature=temperature, top_k=top_k, forbid=forbid)
            if w == ID_EOS:
                break
            out.append(w)
            ctx = (ctx + [w])[-(self.order - 1):]
        return out

    def _beam(self, prefix: list[int], n_words: int, beam: int) -> list[int]:
        ctx0 = [ID_BOS] * (self.order - 1) + list(prefix)
        beams = [(0.0, [], ctx0)]
        for _ in range(n_words):
            cand = []
            for score, words, ctx in beams:
                ids, probs = self.distribution(ctx, temperature=1.0, top_k=0)
                k = min(len(ids), 12)
                top = np.argpartition(-probs, k - 1)[:k]
                for i in top:
                    w = int(ids[i])
                    if w == ID_BOS:
                        continue
                    cand.append((score + math.log(max(probs[i], 1e-12)), words + [w],
                                 (ctx + [w])[-(self.order - 1):]))
            cand.sort(key=lambda x: -x[0] / max(len(x[1]), 1))
            beams = cand[:beam]
            if not beams:
                break
        return beams[0][1] if beams else []

    # ---- 落盘 / 读取 ---------------------------------------------------- #
    def save(self, path: str, meta: dict) -> None:
        arrays = {"vocab": np.asarray(self.vocab, dtype=object).astype(str),
                  "unigram": self.unigram, "cc1": self.cc1}
        for k, tab in self.tables.items():
            for j, lane in enumerate(tab.lanes):
                arrays[f"k{k}_{j}"] = lane
            arrays[f"c{k}"] = tab.counts
            arrays[f"s{k}"] = np.asarray([tab.n1_before, tab.n2_before], dtype=np.int64)
        np.savez_compressed(path, **arrays)
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> tuple["NgramLM", dict]:
        data = np.load(path, allow_pickle=False)
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as fh:
            meta = json.load(fh)
        vocab = [str(x) for x in data["vocab"]]
        order = int(meta["order"])
        tables = {}
        for k in range(2, order + 1):
            lanes = [data[f"k{k}_{j}"] for j in range((k + 3) // 4)]
            n12 = data[f"s{k}"] if f"s{k}" in data else None
            tables[k] = SortedGramTable.from_saved(k, lanes, data[f"c{k}"], n12)
        cc1 = data["cc1"]
        lm = cls(vocab=vocab, tables=tables, unigram=data["unigram"], cc1=cc1,
                 smoothing=meta.get("smoothing", "kn"), alpha=meta.get("alpha", 0.4),
                 order=order)
        return lm, meta


def _discount_from_n12(tab: "SortedGramTable") -> float:
    """Chen & Goodman 的简单折扣估计：D = n1 / (n1 + 2 n2)。

    优先用剪枝前的 n1/n2；如果连剪枝前都没有（例如 1 元表），退回 0.5。
    """
    n1 = float(tab.n1_before) if tab.n1_before else float(np.count_nonzero(tab.counts == 1))
    n2 = float(tab.n2_before) if tab.n2_before else float(np.count_nonzero(tab.counts == 2))
    return n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.5


# --------------------------------------------------------------------------- #
# 训练入口（被 train_ngram.py 调用）
# --------------------------------------------------------------------------- #

def train(train_path: str, dev_path: str = "", order: int = 3, smoothing: str = "kn",
          alpha: float = 0.4, min_count: int = 1, prune: list[int] | None = None,
          max_types: int = MAX_WORD_TYPES,
          min_word_count: int = 1, limit_lines: int = 0, verbose: bool = True,
          encoding: str = "utf-8") -> tuple[NgramLM, dict]:
    """训练 n-gram 模型。

    剪枝用 ``prune`` 按阶指定"最小保留计数"（列表长度 = order-1，对应 2..order 阶），
    例如 5-gram 的 ``prune=[1, 1, 2, 2]`` 表示只剪掉 4-gram / 5-gram 里的 singleton，
    对应 KenLM 的 ``--prune 0 0 1 1``、SRILM 的 ``-gt4min 2 -gt5min 2``。
    ``min_count`` 是简写：只作用于 3 阶及以上的 n-gram（低阶保留，原因见 README）。
    """
    t_all = time.time()
    timing: dict[str, float] = {}

    thresholds = {k: 1 for k in range(2, order + 1)}
    if prune:
        for i, t in enumerate(prune):
            if 2 + i <= order:
                thresholds[2 + i] = int(t)
    elif min_count > 1:
        for k in range(3, order + 1):
            thresholds[k] = min_count

    t = time.time()
    word_counts = count_word_types(train_path, encoding)
    vocab = build_vocab(word_counts, max_types=max_types, min_count=min_word_count)
    word2id = {w: i for i, w in enumerate(vocab)}
    timing["vocab"] = round(time.time() - t, 2)
    if verbose:
        log(f"词表：原始类型 {len(word_counts)} -> 建模词表 {len(vocab)}（含 <unk>/<s>/</s>），"
            f"耗时 {timing['vocab']}s")

    t = time.time()
    train_sents = encode_file(train_path, word2id, limit_lines, encoding)
    timing["encode"] = round(time.time() - t, 2)
    n_tok = sum(len(s) for s in train_sents)
    if verbose:
        log(f"编码：{len(train_sents)} 句 / {n_tok} token，耗时 {timing['encode']}s")

    t = time.time()
    windows = sentence_windows(train_sents, order)
    timing["window"] = round(time.time() - t, 2)
    if verbose:
        log(f"滑窗：{windows.shape[0]} 个 {order}-gram 窗口，耗时 {timing['window']}s")

    unigram = count_unigrams(train_sents, len(vocab))
    tables: dict[int, SortedGramTable] = {}
    for k in range(2, order + 1):
        t = time.time()
        grams = np.ascontiguousarray(windows[:, order - k:])
        tab = SortedGramTable.build(grams, min_count=thresholds[k])
        del grams
        tables[k] = tab
        timing[f"count{k}"] = round(time.time() - t, 2)
        if verbose:
            kept = int(tab.counts.sum())
            log(f"  {k}-gram：类型 {tab.size}（保留计数>={thresholds[k]}），token 覆盖 {kept}，"
                f"耗时 {timing[f'count{k}']}s")
    del windows

    cc1 = np.zeros(len(vocab), dtype=np.int64)
    if order >= 2:
        suffix = unpack_lanes(tables[2].lanes, 2)[:, 1:]
        keys, cnts = np.unique(suffix, return_counts=True)
        cc1[keys.astype(np.int64)] = cnts
    else:
        # 一元模型：没有"左上下文"的概念，退化用原始一元计数当基分布
        cc1 = unigram.copy()

    t = time.time()
    lm = NgramLM(vocab=vocab, tables=tables, unigram=unigram, cc1=cc1,
                 smoothing=smoothing, alpha=alpha, order=order)
    timing["smooth"] = round(time.time() - t, 2)
    if verbose and smoothing == "kn":
        log(f"平滑：Kneser-Ney 折扣 D = " +
            ", ".join(f"{k}:{v:.3f}" for k, v in sorted(lm.discounts.items())))

    stats = {
        "order": order,
        "smoothing": smoothing,
        "alpha": alpha,
        "min_count": min_count,
        "prune": [thresholds[k] for k in range(2, order + 1)],
        "vocab_size": len(vocab),
        "train_sentences": len(train_sents),
        "train_tokens": int(n_tok),
        "ngram_types": {str(k): tables[k].size for k in tables},
        "ngram_tokens": {str(k): int(tables[k].counts.sum()) for k in tables},
        "unigram_tokens": int(unigram.sum()),
        "discounts": {str(k): round(v, 4) for k, v in lm.discounts.items()},
        "timing": timing,
        "train_seconds": round(time.time() - t_all, 2),
    }

    if dev_path:
        t = time.time()
        dev_sents = encode_file(dev_path, word2id, 0, encoding)
        stats["dev_sentences"] = len(dev_sents)
        stats["dev_tokens"] = int(sum(len(s) for s in dev_sents))
        stats["dev_eval_seconds"] = round(time.time() - t, 2)
    return lm, stats
