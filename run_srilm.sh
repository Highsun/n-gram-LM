#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SRILM 训练 / 评测 / 生成脚本（作业资料 1）
#
# 需要先安装对应工具，并通过环境变量指定安装目录。
# 本实验使用 ~/ngram-toolchain/ 中的隔离安装，详见 TOOLCHAIN.md。
#
#   SRILM=/path/to/srilm bash run_srilm.sh            # 真正训练
#   SRILM=/path/to/srilm bash run_srilm.sh --dry-run  # 只打印命令，不执行
#
# SRILM 环境搭建（简要示例，完整安装步骤见 TOOLCHAIN.md）：
#   wget https://www.speech.sri.com/projects/srilm/download.html 的 srilm-1.7.3.tar.gz
#   make SRILM=/path/to/srilm MACHINE_TYPE=macosx && PATH=$SRILM/bin:$PATH
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${CORPUS:-$HERE/corpus}"
MODELS="${MODELS:-$HERE/models}"
RESULTS="${RESULTS:-$HERE/results}"
LOGS="$RESULTS/logs"
TIMING="$RESULTS/srilm_timing.txt"
SRILM="${SRILM:-/usr/local/srilm}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$MODELS" "$LOGS" "$RESULTS"

NG_COUNT="$SRILM/bin/ngram-count"
NG="$SRILM/bin/ngram"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

run() {
  # 打印并执行（dry-run 时只打印）；用 /usr/bin/time 记录墙钟与峰值内存
  echo "+ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    if /usr/bin/time -l "$@" 2> "$LOGS/last_time.txt"; then
      grep -E "real|maximum resident" "$LOGS/last_time.txt" || true
      # -debug 1 的训练过程（每阶 n-gram 数、剪枝情况）留档
      cat "$LOGS/last_time.txt" >> "$LOGS/srilm_stderr.log"
    else
      echo "!! 命令失败（exit=$?）：$*" >&2
      tail -n 5 "$LOGS/last_time.txt" >&2
      FAILED=$((FAILED + 1))
    fi
  fi
}

FAILED=0

if [[ $DRY_RUN -eq 0 ]]; then
  for tool in "$NG_COUNT" "$NG"; do
    if [[ ! -x "$tool" ]]; then
      echo "找不到可执行文件：$tool" >&2
      echo "请先安装/编译 SRILM，并用 SRILM=/path/to/srilm 指定路径；" >&2
      echo "或先跑一次 --dry-run 查看将要执行的命令。" >&2
      exit 2
    fi
  done
fi

echo "SRILM 训练开始：$(date '+%F %T')" | tee -a "$TIMING"

# ---------------------------------------------------------------------------
# 1) 词级 3-gram：各种平滑方法（资料 1 里"几乎支持所有主流平滑/插值"）
#    -kndiscount       : 修改版 Kneser-Ney 折扣（KenLM lmplz 的同类方法）
#    -ukndiscount      : 原始 Kneser-Ney（单一折扣）
#    -interpolate      : 低阶插值（等价于 KenLM 的默认行为）
#    -wbdiscount       : Witten-Bell
#    -addsmooth        : 加性平滑
#    -debug 1 / 2      : 让 SRILM 把训练过程（每阶多少 n-gram、剪枝前后）打到 stderr
# ---------------------------------------------------------------------------
say "词级 3-gram：修改版 KN / 原始 KN / Witten-Bell 三种平滑"
# -kndiscount = 修改版 Kneser-Ney（KenLM lmplz 的同类方法）
# -ukndiscount = 原始 Kneser-Ney（单一折扣，与自研 ngram_lm.py 的 --smoothing kn 对应）
for spec in "kn:-kndiscount -interpolate" "ukn:-ukndiscount -interpolate" "wb:-wbdiscount"; do
  name="${spec%%:*}"; opts="${spec#*:}"
  run "$NG_COUNT" -order 3 -text "$CORPUS/word.train.txt" \
      $opts -gtmin 1 \
      -lm "$MODELS/srilm.word.$name.3.arpa" -debug 1
  # 困惑度 / OOV
  run "$NG" -order 3 -lm "$MODELS/srilm.word.$name.3.arpa" \
      -ppl "$CORPUS/word.dev.txt"
done

# ---------------------------------------------------------------------------
# 2) 5-gram：修改版 KN，以及 SRILM 自带的熵剪枝
#    注意：SRILM 的 -prune 是"熵剪枝"——若删掉某个 n-gram 只让训练集困惑度
#    上升不到 threshold（相对值），就把它删掉；-gtNmin 属于 Good-Turing 折扣参数，
#    在 -kndiscount 下会被忽略，不能当计数剪枝用。
#    另外 SRILM 的 KN 折扣估计器要求各阶 count-of-counts 都非零，所以"直接剪掉
#    全部 singleton 再估 KN"会报错（KenLM 是先估折扣再剪，自研实现用剪枝前统计估折扣，
#    都能绕开这个限制）。
# ---------------------------------------------------------------------------
say "词级 5-gram：修改版 KN（不剪枝 / 熵剪枝）"
run "$NG_COUNT" -order 5 -text "$CORPUS/word.train.txt" \
    -kndiscount -interpolate -gtmin 1 \
    -lm "$MODELS/srilm.word.kn.5.arpa" -debug 1
run "$NG" -order 5 -lm "$MODELS/srilm.word.kn.5.arpa" \
    -ppl "$CORPUS/word.dev.txt"
run "$NG_COUNT" -order 5 -text "$CORPUS/word.train.txt" \
    -kndiscount -interpolate -prune 1e-6 \
    -lm "$MODELS/srilm.word.kn.5.entropypruned.arpa" -debug 1
run "$NG" -order 5 -lm "$MODELS/srilm.word.kn.5.entropypruned.arpa" \
    -ppl "$CORPUS/word.dev.txt"

# ---------------------------------------------------------------------------
# 3) 字级 3-gram（把语料换成字级即可，别的参数完全一样）
# ---------------------------------------------------------------------------
say "字级 3-gram：KN"
run "$NG_COUNT" -order 3 -text "$CORPUS/char.train.txt" \
    -kndiscount -interpolate -gtmin 1 \
    -lm "$MODELS/srilm.char.kn.3.arpa" -debug 1
run "$NG" -order 3 -lm "$MODELS/srilm.char.kn.3.arpa" \
    -ppl "$CORPUS/char.dev.txt"

# ---------------------------------------------------------------------------
# 4) 用 SRILM 自带采样做续写（和 generate.py 的采样生成对齐）
#    -gen N            : 生成 N 个随机句子（不含前缀）
#    -gen-prefixes f   : 以文件里的每一行为前缀条件生成，每行一句
#                        （注意：输出只是"续写部分"，不回显前缀，也不自动加 <s>）
#    -seed             : 随机种子
# ---------------------------------------------------------------------------
say "SRILM 采样续写"
GEN_PREFIX="$LOGS/srilm_prefix.txt"
echo "在 阳光 明媚 的 五月 ， 我们 学校 胜利 召开 了" > "$GEN_PREFIX"
run "$NG" -order 5 -lm "$MODELS/srilm.word.kn.5.arpa" \
    -gen 1 -gen-prefixes "$GEN_PREFIX" -seed 2024 -debug 1

echo "SRILM 训练结束：$(date '+%F %T')" | tee -a "$TIMING"
if [[ $FAILED -gt 0 ]]; then
  echo "有 $FAILED 条命令失败，请检查上面的输出。" >&2
fi
echo "提示：SRILM 不支持 Stupid Backoff（Goodman 2001），该平滑在自研实现里"
echo "      （--smoothing sb）以及资料 3 描述的 MapReduce 训练流程中实现。"
