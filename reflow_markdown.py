#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 Markdown 文档中被硬换行切碎的段落合并回"一段一行"，便于阅读与 diff。

Markdown 里同一段落内的换行在渲染时等价于一个空格，但源码被切成很多行后，
在编辑器、diff、以及不把软换行当空格的渲染器里都会显得零碎。本脚本做一次
机械规整：

* 合并同一段落 / 同一条列表项 / 同一段引用中的续行；
* 保留标题、表格、代码块（含 ``` 与 ~~~）、图片、HTML 块、空行不动；
* 中英混排时按需补空格：两侧都是中日韩字符则直接拼接，只要有一侧是 ASCII 就补一个空格
  （与 Markdown 原本"换行 = 空格"的渲染效果保持一致）。

    python reflow_markdown.py --check  README.md      # 只报告有多少处需要合并
    python reflow_markdown.py --write  README.md ...  # 就地规整（幂等，可反复运行）
"""

from __future__ import annotations

import argparse
import re
import sys

FENCE_RE = re.compile(r"^\s*(```|~~~)")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
TABLE_RE = re.compile(r"^\s*\|")
LIST_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
QUOTE_RE = re.compile(r"^\s*>\s?")
BLOCK_RE = re.compile(r"^\s*(<[a-zA-Z!/]|!\[)")
HR_RE = re.compile(r"^\s*([-*_]\s*){3,}$")


def is_cjk(ch: str) -> bool:
    """中日韩字符与全角标点。"""
    o = ord(ch)
    return (0x3000 <= o <= 0x303F or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF
            or 0xFF00 <= o <= 0xFFEF or 0x2000 <= o <= 0x206F and ch in "——…“”‘’")


def join(prev: str, cur: str) -> str:
    """把两行拼成一行：两侧都是 CJK 时直接拼，否则中间补一个空格。"""
    prev, cur = prev.rstrip(), cur.strip()
    if not prev:
        return cur
    if not cur:
        return prev
    if is_cjk(prev[-1]) and is_cjk(cur[0]):
        return prev + cur
    return prev + " " + cur


def reflow(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    in_fence = False
    in_table = False
    prev_kind = "blank"           # blank | para | item | quote | skip

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if FENCE_RE.match(line):
            in_fence = not in_fence
            out.append(line)
            prev_kind = "skip"
            in_table = False
            continue
        if in_fence:
            out.append(line)
            prev_kind = "skip"
            continue

        if TABLE_RE.match(line):
            out.append(line)
            prev_kind = "skip"
            in_table = True
            continue
        in_table = False

        if not stripped:
            out.append("")
            prev_kind = "blank"
            continue
        if HEADING_RE.match(line) or HR_RE.match(line) or BLOCK_RE.match(line):
            out.append(line)
            prev_kind = "skip"
            continue

        m_list, m_quote = LIST_RE.match(line), QUOTE_RE.match(line)
        if m_list:
            out.append(line)
            prev_kind = "item"
            continue
        if m_quote:
            body = QUOTE_RE.sub("", line)
            if prev_kind == "quote" and out:
                out[-1] = join(out[-1], body)
            else:
                out.append("> " + body.strip())
                prev_kind = "quote"
            continue

        # 普通段落行
        if prev_kind in ("para", "item", "quote") and out:
            prefix = ""
            if prev_kind == "item":
                # 续行并入上一条列表项（保持列表项自身的缩进层级）
                prefix = ""
            out[-1] = join(out[-1], line)
            prev_kind = "para" if prev_kind == "para" else prev_kind
            continue

        out.append(line)
        prev_kind = "para"

    return "\n".join(out)


def count_merges(text: str) -> int:
    return sum(1 for a, b in zip(text.split("\n"), text.split("\n")[1:])
               if a.strip() and b.strip() and not b.lstrip().startswith(("|", "#", "-", "*", "+", ">", "```", "~~~", "<", "!["))
               and not re.match(r"^\s*\d+[.)]\s", b) and not HEADING_RE.match(b))


def main() -> int:
    ap = argparse.ArgumentParser(description="规整 Markdown 的硬换行")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--write", action="store_true", help="就地写入（默认只检查）")
    ap.add_argument("--check", action="store_true", help="只报告，不写入")
    args = ap.parse_args()

    for path in args.files:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        dst = reflow(src)
        n_before, n_after = len(src.split("\n")), len(dst.split("\n"))
        if args.write and not args.check:
            if dst != src:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(dst)
            print(f"{path}: {n_before} 行 → {n_after} 行（合并 {n_before - n_after} 处）")
        else:
            print(f"{path}: 当前 {n_before} 行，规整后 {n_after} 行"
                  f"，建议合并 {n_before - n_after} 处"
                  + ("（已是规整状态）" if dst == src else "，运行 --write 生效"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
