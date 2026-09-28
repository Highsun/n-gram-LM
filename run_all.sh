#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 一键跑通全部实验并出图（可反复执行：已有的训练产物默认跳过，加 --force 重跑）
#
#   conda activate ai
#   bash run_all.sh                 # 全流程
#   bash run_all.sh --skip-tools    # 不碰 SRILM / KenLM（没装也能跑）
#   bash run_all.sh --force         # 连训练一起重跑
#
# 流程：清洗语料 → 自研实验矩阵(训练+生成) → SRILM/KenLM 实跑（装了才跑）
#       → 三方对比表 → 完整验证集困惑度总表 → 生成参数扫描 → 画图
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

FORCE=0
SKIP_TOOLS=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --skip-tools) SKIP_TOOLS=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
  esac
done

SRILM="${SRILM:-$HOME/ngram-toolchain/src/srilm}"
KENLM="${KENLM:-$HOME/ngram-toolchain/src/kenlm}"
FAILED=0

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
step() { echo "+ $*"; "$@" || { echo "!! 上一条命令失败"; FAILED=$((FAILED + 1)); }; }

PY="${PYTHON:-python}"
if ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "请先激活带 numpy 的环境（例如 conda activate ai）" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
say "0/7 环境"
step bash check_env.sh

# ---------------------------------------------------------------------------
say "1/7 清洗语料（去编号 / 词性 / 方括号，切句并划分 train/dev）"
if [[ -f corpus/word.train.txt && $FORCE -eq 0 ]]; then
  echo "corpus/word.train.txt 已存在，跳过（--force 可强制重跑）"
else
  step "$PY" clean_corpus.py --input-dir data --out-dir corpus --unit both --dev-ratio 0.02
fi

# ---------------------------------------------------------------------------
say "2/7 自研实现：训练矩阵 + 生成样例（run_experiments.py）"
if [[ $FORCE -eq 1 ]]; then
  step "$PY" run_experiments.py --stage all --force
else
  step "$PY" run_experiments.py --stage all
fi

# ---------------------------------------------------------------------------
if [[ $SKIP_TOOLS -eq 0 && -x "$KENLM/build/bin/lmplz" ]]; then
  say "3/7 KenLM：训练 + 二进制转换 + 困惑度"
  if [[ -f models/kenlm.word.o5.trie && $FORCE -eq 0 ]]; then
    echo "models/kenlm.word.o5.trie 已存在，跳过（--force 可强制重跑）"
  else
    step env KENLM="$KENLM" bash run_kenlm.sh
  fi
else
  say "3/7 KenLM：跳过（未安装或 --skip-tools）"
  echo "  安装方式见 TOOLCHAIN.md；dry-run： KENLM=$KENLM bash run_kenlm.sh --dry-run"
fi

# ---------------------------------------------------------------------------
if [[ $SKIP_TOOLS -eq 0 && -x "$SRILM/bin/ngram-count" ]]; then
  say "4/7 SRILM：三种平滑 + 剪枝 + 困惑度 + 采样"
  if [[ -f models/srilm.word.kn.5.arpa && $FORCE -eq 0 ]]; then
    echo "models/srilm.word.kn.5.arpa 已存在，跳过（--force 可强制重跑）"
  else
    step env SRILM="$SRILM" bash run_srilm.sh
  fi
  say "5/7 三方对比（同一语料与验证集，含剪枝/量化变体）"
  step env KENLM="$KENLM" SRILM="$SRILM" \
       "$PY" compare_toolchains.py --order 3 5 --with-pruning
else
  say "4/7 SRILM / 5/7 三方对比：跳过（未安装或 --skip-tools）"
fi

# ---------------------------------------------------------------------------
say "6/7 完整验证集上的困惑度总表 + 生成参数扫描"
step "$PY" ppl_table.py
step "$PY" gen_sweep.py --model models/word.kn5.npz --seeds 10

# ---------------------------------------------------------------------------
say "7/7 绘图（results/figures/*.png）"
step "$PY" plot_results.py

echo
if [[ $FAILED -gt 0 ]]; then
  echo "有 $FAILED 个步骤失败，请检查上面的输出。" >&2
else
  echo "全部完成。产物在 results/ 与 results/figures/。"
fi
