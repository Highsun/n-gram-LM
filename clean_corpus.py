#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把《人民日报》1998 年分词 / 词性标注语料清洗成"纯文本"n-gram 训练语料。

原始的一行长这样（PKU 标注体系）::

    19980101-01-001-004/m  １２月/t  ３１日/t  ，/w  [中央/n  人民/n  广播/vn  电台/n]nt  ...
    |<---- 句子编号 ---->|  |<- 词语/词性 ->|     |<----- 命名实体方括号 ----->|

本脚本做四件"必须做"的清洗（详见 README.md「为什么必须去掉标签」一节）：

1. 删掉行首的句子编号 ``19980101-01-001-004/m``（语料编号，不是语言的一部分）；
2. 删掉每个词的词性标签 ``迈向/v -> 迈向``；
3. 拆掉命名实体方括号 ``[中央/n 人民/n 广播/vn 电台/n]nt -> 中央 人民 广播 电台``；
4. 修掉语料里 1329 处 ``近年来/l/%`` 这类多余标注，并把全角数字/字母做 NFKC 归一化。

同时按句末标点切句、按"文章"粒度划分训练/验证集，输出给 SRILM / KenLM / 自研
训练器使用的纯文本语料。

用法::

    python clean_corpus.py --input-dir data --out-dir corpus
    python clean_corpus.py --punctuation drop --unit word   # 去掉所有标点 / 只做词级

只依赖 Python 标准库。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

#: 行首句子编号，例如 19980101-01-001-004/m
SENTENCE_ID_RE = re.compile(r"^(?P<sid>\d{8}-\d{2}-\d{3}-\d{3})(?:/\S+)?\s+")

#: 合法的词性标签（PKU 标记集：n, v, w, nr, ns, nt, vn, ad, j, l, ...）
POS_RE = re.compile(r"^[A-Za-z]+$")

#: 语料里 1329 处 "近年来/l/%" 这类多余标注：词/词性后面又挂了一个 /%
EXTRA_TAG_RE = re.compile(r"/%$")

#: 命名实体右括号及其标签，例如 "]nt" / "]ns" / "]nz" / "]i"
CLOSE_BRACKET_RE = re.compile(r"\]\w*$")

#: 句末标点（切句用）
SENT_END = frozenset("。！？…")

#: 紧跟句末标点的收尾符号，仍算同一句
CLOSERS = frozenset("）)】]」』”\"’'》〉")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 单 token / 单行清洗
# --------------------------------------------------------------------------- #

def clean_token(tok: str) -> tuple[str, str]:
    """拆出 (词, 词性)。不是 ``词/词性`` 形式时，词性返回空串。"""
    tok = tok.lstrip("[")                 # 1) 左方括号（最深嵌套 2 层）
    tok = CLOSE_BRACKET_RE.sub("", tok)   # 2) 右方括号 + 实体类别标签 "]nt"
    tok = EXTRA_TAG_RE.sub("", tok)       # 3) 多余标注 "/%"
    if not tok:
        return "", ""
    if "/" in tok:
        word, tag = tok.rsplit("/", 1)    # 4) 以最后一个斜杠切 词/词性
        if POS_RE.match(tag):
            return word, tag
        return tok, ""                    # 词本身含斜杠，例如 1998/1/2
    return tok, ""


_ALNUM = frozenset("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


def normalize(word: str, mode: str) -> str:
    """NFKC 归一化。

    注意：完整 NFKC 会把中文标点也改掉（``，`` U+FF0C -> ``,``、``（`` -> ``(``），
    对中文语料是破坏性的。所以默认用 ``nfkc-alnum``：只把全角数字/字母折成半角
    （``１２月`` -> ``12月``、``Ａ`` -> ``A``），标点原样保留。
    """
    if mode == "none":
        return word
    if mode == "nfkc":
        return unicodedata.normalize("NFKC", word)
    out = []
    for ch in word:
        nf = unicodedata.normalize("NFKC", ch)
        out.append(nf if (len(nf) == 1 and nf in _ALNUM) else ch)
    return "".join(out)


def clean_line(line: str, args: argparse.Namespace) -> tuple[str, str, list[str]] | None:
    """把一行标注语料清洗成 (句子编号, 文章编号, [词...])，空行返回 None。"""
    line = line.rstrip("\r\n")
    if not line.strip():
        return None
    m = SENTENCE_ID_RE.match(line)
    if m:
        sid = m.group("sid")
        body = line[m.end():]
    else:                                  # 兜底：没有编号的行
        sid = "UNKNOWN-000000-000-000"
        body = line
    phrase_id = "-".join(sid.split("-")[:3])   # 19980101-01-001 为一篇文章

    words: list[str] = []
    for tok in body.split():
        word, tag = clean_token(tok)
        if not word:
            continue
        if args.punctuation == "drop" and tag == "w":
            continue
        if args.punctuation == "sentence-end-only" and tag == "w" and word not in SENT_END:
            continue
        word = normalize(word, args.normalize)
        if word:
            words.append(word)
    return sid, phrase_id, words


def split_sentences(words: list[str], max_len: int = 0) -> list[list[str]]:
    """按句末标点切句；句末标点后紧跟的收尾符号并入前一句。"""
    sents: list[list[str]] = []
    buf: list[str] = []
    for i, w in enumerate(words):
        buf.append(w)
        if w and w[-1] in SENT_END:
            nxt = words[i + 1] if i + 1 < len(words) else None
            if nxt is None or nxt not in CLOSERS:
                sents.append(buf)
                buf = []
    if buf:
        sents.append(buf)

    if max_len and max_len > 0:            # 过长句子硬切（默认关闭）
        cut: list[list[str]] = []
        for s in sents:
            for i in range(0, len(s), max_len):
                cut.append(s[i:i + max_len])
        sents = cut
    return sents


def iter_sentences(path: str, args: argparse.Namespace):
    """逐行读取原始语料，产出 (句子编号, 文章编号, [词...])。"""
    with open(path, encoding=args.encoding) as fh:
        for line in fh:
            cleaned = clean_line(line, args)
            if cleaned is None:
                continue
            sid, pid, words = cleaned
            if not words:
                continue
            if args.split_sentences:
                for s in split_sentences(words, args.max_sent_len):
                    if len(s) >= args.min_sent_len:
                        yield sid, pid, s
            else:
                if len(words) >= args.min_sent_len:
                    yield sid, pid, words


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="人民日报 1998 标注语料 -> 纯文本 n-gram 语料")
    p.add_argument("--input-dir", default="data", help="原始语料目录（*.txt）")
    p.add_argument("--out-dir", default="corpus", help="清洗后语料输出目录")
    p.add_argument("--unit", choices=["word", "char", "both"], default="both",
                   help="word=词级，char=字级，both=都生成（默认）")
    p.add_argument("--normalize", choices=["nfkc-alnum", "nfkc", "none"], default="nfkc-alnum",
                   help="NFKC 归一化：nfkc-alnum=只折全角数字/字母（默认）；nfkc=全量；none=不归一")
    p.add_argument("--punctuation", choices=["keep", "drop", "sentence-end-only"],
                   default="keep",
                   help="标点处理：keep=全保留；drop=全删；sentence-end-only=只保留句末标点")
    p.add_argument("--split-sentences", dest="split_sentences", action="store_true", default=True,
                   help="按 。！？… 切句（默认开）")
    p.add_argument("--no-split-sentences", dest="split_sentences", action="store_false",
                   help="以原始行（标题或整段）为一句")
    p.add_argument("--max-sent-len", type=int, default=0,
                   help=">0 时强制截断过长句子（token 数），默认 0 不截断")
    p.add_argument("--min-sent-len", type=int, default=1,
                   help="丢弃长度小于该值的句子（默认 1）")
    p.add_argument("--dev-ratio", type=float, default=0.02,
                   help="验证集比例（按'文章'粒度切分，默认 0.02）")
    p.add_argument("--seed", type=int, default=1998, help="划分验证集的随机种子")
    p.add_argument("--encoding", default="utf-8")
    return p


def dump(out_dir: str, subset: str, rows, unit: str) -> None:
    path = os.path.join(out_dir, f"{unit}.{subset}.txt")
    with open(path, "w", encoding="utf-8") as fh:
        for _, _, words in rows:
            fh.write(" ".join(words if unit == "word" else "".join(words)) + "\n")
    log(f"  写出 {path}（{len(rows)} 行）")


def main() -> int:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".txt"))
    if not files:
        log(f"错误：{args.input_dir} 下没有 .txt 语料")
        return 1

    log(f"输入 {args.input_dir}，共 {len(files)} 个文件：{', '.join(files)}")
    log(f"参数：unit={args.unit} punctuation={args.punctuation} normalize={args.normalize} "
        f"split_sentences={args.split_sentences} dev_ratio={args.dev_ratio}")

    stats: dict = {
        "input_files": files,
        "params": {k: v for k, v in sorted(vars(args).items())},
        "raw_lines": 0,
        "raw_tokens": 0,
        "raw_vocab": 0,
        "clean_tokens": 0,
        "sentences": 0,
        "articles": 0,
        "vocab_word": 0,
        "vocab_char": 0,
        "tag_inventory": {},
        "sentence_len": {},
        "examples": [],
    }
    tag_counter: Counter[str] = Counter()
    word_counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()
    raw_vocab: set[str] = set()
    n_sent = n_tok = 0
    len_hist: Counter[int] = Counter()
    examples: list[dict] = []

    t0 = time.time()
    sents: list[tuple[str, str, list[str]]] = []
    for name in files:
        path = os.path.join(args.input_dir, name)
        with open(path, encoding=args.encoding) as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                stats["raw_lines"] += 1
                toks = line.split()
                stats["raw_tokens"] += max(0, len(toks) - 1)
                for tok in toks[1:]:
                    raw_word, tag = clean_token(tok)
                    if raw_word:
                        raw_vocab.add(raw_word)
                        tag_counter[tag or "(none)"] += 1

                cleaned = clean_line(line, args)
                if cleaned is None or not cleaned[2]:
                    continue
                sid, pid, words = cleaned
                if len(examples) < 6:
                    examples.append({"raw": line[:220], "clean": " ".join(words)[:220]})
                pieces = split_sentences(words, args.max_sent_len) if args.split_sentences else [words]
                for s in pieces:
                    if len(s) < args.min_sent_len:
                        continue
                    sents.append((sid, pid, s))
                    n_sent += 1
                    n_tok += len(s)
                    len_hist[min(len(s), 200)] += 1
                    word_counter.update(s)
                    char_counter.update("".join(s))
        log(f"  {name}: 累计句子 {n_sent}，累计 token {n_tok}")

    stats["clean_tokens"] = n_tok
    stats["sentences"] = n_sent
    stats["raw_vocab"] = len(raw_vocab)
    stats["vocab_word"] = len(word_counter)
    stats["vocab_char"] = len(char_counter)
    stats["articles"] = len({p for _, p, _ in sents})
    stats["tag_inventory"] = dict(tag_counter.most_common(30))
    stats["sentence_len"] = {
        "min": min(len_hist) if len_hist else 0,
        "max": max(len_hist) if len_hist else 0,
        "mean": round(n_tok / max(n_sent, 1), 2),
    }
    stats["clean_seconds"] = round(time.time() - t0, 2)
    stats["examples"] = examples

    # ---------- 按"文章"切 train / dev（防止同文句子同时出现在两边） ----------
    rng = random.Random(args.seed)
    articles = sorted({p for _, p, _ in sents})
    dev_articles = {a for a in articles if rng.random() < args.dev_ratio}
    train = [r for r in sents if r[1] not in dev_articles]
    dev = [r for r in sents if r[1] in dev_articles]
    log(f"文章 {len(articles)} 篇 -> 训练 {len(articles) - len(dev_articles)} 篇 / "
        f"验证 {len(dev_articles)} 篇")
    log(f"句子 {len(train)} 训练 / {len(dev)} 验证")

    units = ["word", "char"] if args.unit == "both" else [args.unit]
    for unit in units:
        dump(args.out_dir, "train", train, unit)
        dump(args.out_dir, "dev", dev, unit)
        dump(args.out_dir, "all", sents, unit)

    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)
    log(f"统计写入 {os.path.join(args.out_dir, 'stats.json')}")
    log(f"完成：原始 token {stats['raw_tokens']} -> 清洗后 token {n_tok}；"
        f"词级词表 {len(word_counter)}，字级字表 {len(char_counter)}；"
        f"耗时 {stats['clean_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
