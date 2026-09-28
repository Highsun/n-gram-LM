#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# KenLM 训练 / 评测 / 查询脚本（作业资料 2）
#
# 需要先安装对应工具，并通过环境变量指定安装目录。
# 本实验使用 ~/ngram-toolchain/ 中的隔离安装，详见 TOOLCHAIN.md。
#
#   KENLM=/path/to/kenlm bash run_kenlm.sh            # 真正训练
#   KENLM=/path/to/kenlm bash run_kenlm.sh --dry-run  # 只打印命令
#
# KenLM 环境搭建（简要示例，完整安装步骤见 TOOLCHAIN.md）：
#   git clone https://github.com/kpu/kenlm && cd kenlm
#   mkdir -p build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
#   （依赖 boost / eigen / zlib；建好后 build/bin/lmplz 就是训练器）
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${CORPUS:-$HERE/corpus}"
MODELS="${MODELS:-$HERE/models}"
RESULTS="${RESULTS:-$HERE/results}"
LOGS="$RESULTS/logs"
TIMING="$RESULTS/kenlm_timing.txt"
KENLM="${KENLM:-/usr/local/kenlm}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$MODELS" "$LOGS" "$RESULTS"

LMPLZ="$KENLM/build/bin/lmplz"
BUILD_BINARY="$KENLM/build/bin/build_binary"
QUERY="$KENLM/build/bin/query"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

run() {
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    if /usr/bin/time -l "$@" 2> "$LOGS/last_time.txt"; then
      grep -E "real|maximum resident" "$LOGS/last_time.txt" || true
      # 把命令自己的 stderr（含 lmplz 的训练进度）留档，便于"观察训练过程"
      cat "$LOGS/last_time.txt" >> "$LOGS/kenlm_stderr.log"
    else
      echo "!! 命令失败（exit=$?）：$*" >&2
      tail -n 5 "$LOGS/last_time.txt" >&2
      FAILED=$((FAILED + 1))
    fi
  fi
}

FAILED=0

if [[ $DRY_RUN -eq 0 ]]; then
  for tool in "$LMPLZ" "$BUILD_BINARY"; do
    if [[ ! -x "$tool" ]]; then
      echo "找不到可执行文件：$tool" >&2
      echo "请先编译 KenLM，并用 KENLM=/path/to/kenlm 指定路径；或先跑 --dry-run。" >&2
      exit 2
    fi
  done
fi

echo "KenLM 训练开始：$(date '+%F %T')" | tee -a "$TIMING"

# ---------------------------------------------------------------------------
# 1) 词级 3-gram：默认 KS 平滑（Kneser-Ney+straight 折扣）
#    -S 4G             : 训练时最多用 4GB 内存（资料 2 强调内存/延迟）
#    --discount_fallback: 数据不足时自动回退折扣（小语料常用）
#    -v                : 打印进度（观察训练过程）
# ---------------------------------------------------------------------------
say "词级 3-gram（默认平滑，全量语料）"
run "$LMPLZ" -o 3 -S 4G --text "$CORPUS/word.train.txt" \
    --arpa "$MODELS/kenlm.word.o3.arpa" --verbose_header
run "$BUILD_BINARY" trie "$MODELS/kenlm.word.o3.arpa" "$MODELS/kenlm.word.o3.trie"
run "$BUILD_BINARY" probing "$MODELS/kenlm.word.o3.arpa" "$MODELS/kenlm.word.o3.probing"
# 注意：KenLM 自带 getopt 不重排参数，选项必须写在模型文件前面
run "$QUERY" -v summary "$MODELS/kenlm.word.o3.trie" < "$CORPUS/word.dev.txt"

# ---------------------------------------------------------------------------
# 2) 词级 5-gram + 分阶剪枝（0 表示该阶不剪，这里是"只剪 4/5-gram 的 singleton"）
# ---------------------------------------------------------------------------
say "词级 5-gram + 剪枝"
run "$LMPLZ" -o 5 -S 8G --text "$CORPUS/word.train.txt" \
    --arpa "$MODELS/kenlm.word.o5.arpa" --prune 0 0 1 1 --verbose_header
run "$BUILD_BINARY" trie "$MODELS/kenlm.word.o5.arpa" "$MODELS/kenlm.word.o5.trie"
run "$QUERY" -v summary "$MODELS/kenlm.word.o5.trie" < "$CORPUS/word.dev.txt"

# ---------------------------------------------------------------------------
# 3) 字级 3-gram（只换语料）
# ---------------------------------------------------------------------------
say "字级 3-gram"
run "$LMPLZ" -o 3 -S 4G --text "$CORPUS/char.train.txt" \
    --arpa "$MODELS/kenlm.char.o3.arpa" --verbose_header
run "$BUILD_BINARY" trie "$MODELS/kenlm.char.o3.arpa" "$MODELS/kenlm.char.o3.trie"
run "$QUERY" -v summary "$MODELS/kenlm.char.o3.trie" < "$CORPUS/char.dev.txt"

# ---------------------------------------------------------------------------
# 4) 查询单个句子（KenLM 的优势：低延迟查询）
# ---------------------------------------------------------------------------
say "单句打分 / 查询延迟"
echo "在 阳光 明媚 的 五月 ， 我们 学校 胜利 召开 了 。" \
    | run "$QUERY" -v word "$MODELS/kenlm.word.o5.trie"

echo "KenLM 训练结束：$(date '+%F %T')" | tee -a "$TIMING"
if [[ $FAILED -gt 0 ]]; then
  echo "有 $FAILED 条命令失败，请检查上面的输出。" >&2
fi
echo "提示：KenLM 的 lmplz 不直接支持 Stupid Backoff；SB 在自研实现"
echo "      （--smoothing sb）和资料 3 的分布式流程里实现。"
