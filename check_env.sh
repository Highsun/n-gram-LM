#!/usr/bin/env bash
# 记录实验环境（写进 results/environment.txt），顺带检查 SRILM / KenLM 是否可用。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results/environment.txt"
mkdir -p "$HERE/results"

{
  echo "== 环境检查 $(date '+%F %T') =="
  echo "OS      : $(uname -a)"
  echo "CPU 核数 : $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo unknown)"
  echo "内存     : $(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.1f GB\n", $1/1024/1024/1024}' || echo unknown)"
  echo "conda   : ${CONDA_DEFAULT_ENV:-未激活}"
  echo "python  : $(command -v python) $(python -V 2>&1)"
  echo -n "numpy   : "; python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "未安装"
  echo -n "nltk    : "; python -c "import nltk; print(nltk.__version__)" 2>/dev/null || echo "未安装"
  echo -n "kenlm(py): "; python -c "import kenlm; print(kenlm.__file__)" 2>/dev/null || echo "未安装"
  echo -n "SRILM ngram-count : "; command -v ngram-count 2>/dev/null || echo "未安装"
  echo -n "KenLM lmplz       : "; command -v lmplz 2>/dev/null || echo "未安装"
  echo
  echo "== 隔离工具链（~/ngram-toolchain，可整目录删除，见 TOOLCHAIN.md）=="
  TOOLCHAIN="$HOME/ngram-toolchain"
  if [ -d "$TOOLCHAIN" ]; then
    echo "存在：${TOOLCHAIN}（$(du -sh "$TOOLCHAIN" 2>/dev/null | cut -f1)）"
    [ -x "$TOOLCHAIN/src/srilm/bin/ngram-count" ] && \
        echo "  SRILM : $TOOLCHAIN/src/srilm (RELEASE $(cat "$TOOLCHAIN/src/srilm/RELEASE" 2>/dev/null))" || echo "  SRILM : 未编译"
    [ -x "$TOOLCHAIN/src/kenlm/build/bin/lmplz" ] && \
        echo "  KenLM : $TOOLCHAIN/src/kenlm (git $(git -C "$TOOLCHAIN/src/kenlm" rev-parse --short HEAD 2>/dev/null))" || echo "  KenLM : 未编译"
    echo "  用法  : SRILM=$TOOLCHAIN/src/srilm KENLM=$TOOLCHAIN/src/kenlm bash run_srilm.sh|run_kenlm.sh"
  else
    echo "不存在（未安装等值工具；run_srilm.sh / run_kenlm.sh 仍可 --dry-run）"
  fi
  echo
  echo "== 语料 =="
  for f in "$HERE"/corpus/*.txt; do
    [ -f "$f" ] && printf '%-28s %8s 行  %6s\n' "$(basename "$f")" "$(wc -l < "$f" | tr -d ' ')" "$(du -h "$f" | cut -f1)"
  done
  echo
  echo "== 原始语料 =="
  for f in "$HERE"/data/*.txt; do
    [ -f "$f" ] && printf '%-28s %8s 行  %6s\n' "$(basename "$f")" "$(wc -l < "$f" | tr -d ' ')" "$(du -h "$f" | cut -f1)"
  done
} | tee "$OUT"
