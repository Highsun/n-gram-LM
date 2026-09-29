# n-gram-LM

基于 Python 和 NumPy 的中文 n-gram 语言模型项目，提供语料清洗、模型训练、困惑度评测和文本续写，并附带 SRILM / KenLM 的对比实验脚本。适合学习统计语言模型、验证实现和复现小规模实验。

## 功能

- 支持词级与字级建模，n-gram 阶数为 1–8。
- 实现插值 Kneser-Ney（KN，单折扣）和 Stupid Backoff（SB）打分。
- 支持分阶频次剪枝、词表裁剪及限制训练句数。
- 提供采样、贪心和 beam 解码，以及 temperature、top-k 参数。
- 支持 PKU 格式标注语料清洗、按文章划分训练集与验证集。
- 提供正确性自检、批量实验、结果汇总及绘图脚本。

## 安装

需要 **Python 3.10+**，运行环境为 **macOS 或 Linux**。训练与推理依赖 NumPy，绘图依赖 Matplotlib；自研模型无需编译 C/C++ 扩展。

```bash
git clone https://github.com/Highsun/n-gram-LM.git
cd n-gram-LM

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

后续命令均从仓库根目录运行。SRILM 和 KenLM 是可选工具，仅在运行对应实验时需要安装，配置方法见 [TOOLCHAIN.md](TOOLCHAIN.md)。

## 快速开始

以下示例使用人工语料，可在不下载数据的情况下验证训练与生成流程。该语料仅用于演示，不代表实际模型效果。

### 1. 准备语料

训练器读取 UTF-8 文本：每行一句，token 之间用空格分隔。词级模型使用分词后的词，字级模型使用单个字符。

```bash
mkdir -p corpus
cat > corpus/demo.train.txt <<'EOF'
我们 学校 举办 了 活动 。
我们 学校 举办 了 讲座 。
我们 学校 召开 了 会议 。
同学们 参加 了 活动 。
同学们 参加 了 讲座 。
老师 参加 了 会议 。
EOF
cat > corpus/demo.dev.txt <<'EOF'
我们 学校 举办 了 活动 。
EOF
```

### 2. 训练并评测

```bash
python train_ngram.py \
    --train corpus/demo.train.txt \
    --dev corpus/demo.dev.txt \
    --order 3 --smoothing kn --ppl-limit 0 \
    --out models/demo.kn3.npz
```

命令会输出训练统计、验证集 PPL 和 OOV 率，并保存 `models/demo.kn3.npz` 与同名 `.json` 元信息。`--ppl-limit 0` 表示使用完整验证集。

### 3. 生成文本

```bash
python generate.py --model models/demo.kn3.npz \
    --prefix-tokens "我们 学校" \
    --temperature 0.7 --top-k 5 --seed 7 \
    --min-words 3 --max-words 10
```

`--prefix-tokens` 接收已分词的前缀；中文词级模型也可用 `--prefix "我们学校"`，由模型词表进行前向最大匹配分词。

## 使用自己的数据

### 已分词的纯文本

直接将数据路径传给 `train_ngram.py`，无需运行清洗器。建议每行对应一个句子，并使用与训练语料一致的方式处理验证集和生成前缀。

```bash
python train_ngram.py \
    --train corpus/word.train.txt --dev corpus/word.dev.txt \
    --order 5 --smoothing kn --prune "1 1 2 2" \
    --ppl-limit 0 --out models/word.kn5.pruned.npz
```

此处 `--prune "1 1 2 2"` 分别指定 2、3、4、5 阶的最小保留计数，即删除 4、5 阶中仅出现一次的序列。

### PKU 格式的标注语料

将原始 `.txt` 文件放入 `data/`，或通过 `--input-dir` 指定目录：

```bash
python clean_corpus.py --input-dir data --out-dir corpus \
    --unit both --dev-ratio 0.02 --seed 1998
```

清洗器去除文本编号、词性标签和实体标记，归一化数字与字母，切句后按文章划分训练集与验证集。输出包括 `word.{train,dev,all}.txt`、`char.{train,dev,all}.txt` 及 `stats.json`。双层词性后缀的处理仍有局限，见下方“已知限制”。

**仓库不包含原始语料、清洗语料或预训练模型。** `data/`、`corpus/` 和 `models/` 均已加入 `.gitignore`；请自行准备数据并遵守其许可。

### 更多命令与 Python API

命令行完整参数以各脚本的帮助输出为准：

```bash
python clean_corpus.py --help
python train_ngram.py --help
python generate.py --help
```

生成时可用 `--mode greedy` 或 `--mode beam --beam 5` 切换解码方式，使用 `--show-top 10` 查看前缀后的候选词。

也可以在仓库根目录通过 Python 调用模型：

```python
import ngram_lm as N

lm, stats = N.train("corpus/demo.train.txt", order=3, smoothing="kn")
lm.save("models/demo.api.npz", stats)

word2id = {word: i for i, word in enumerate(lm.vocab)}
dev = N.encode_file("corpus/demo.dev.txt", word2id)
print(lm.perplexity(dev, max_tokens=0))
```

## 验证与实验复现

无需外部语料即可运行小语料正确性自检：

```bash
python selftest_ngram_lm.py
```

自检将计数、KN 概率及 SB 分数与朴素参考实现比较，并检查分布归一化。其覆盖范围见“已知限制”。

准备好实验语料后，可运行批量流程：

```bash
bash run_all.sh --skip-tools  # 自研训练、评测、生成扫描与绘图
bash run_all.sh               # 同时运行已配置的 SRILM / KenLM 实验
```

复现实验使用 `data/` 中的 PKU 格式原始语料，与快速开始的人工语料示例不同。已有训练产物默认复用，`--force` 可重新清洗和训练；流程可能更新 `results/` 中的记录。标签对照与正确性自检需单独运行，详细步骤见 [EXPERIMENTS.md](EXPERIMENTS.md)。

仓库保留了《人民日报》1998 年上半年语料的实验记录和 11 张图，方便查阅：

| 资料 | 内容 |
|---|---|
| [实验运行手册](EXPERIMENTS.md) | 复现步骤、指标定义和图表目录 |
| [训练统计](results/training_time.md) | 不同配置的耗时、资源开销与模型规模 |
| [完整验证集指标](results/ppl_full.md) | PPL、OOV 与信息量指标 |
| [工具链对比](results/toolchain_comparison.md) | 自研实现、KenLM 与 SRILM 的记录 |
| [生成样例](results/generation_samples.md) / [参数扫描](results/generation_sweep_summary.md) | 解码配置与温度、top-k 的影响 |
| [复核记录](results/verification.json) | 数据重建、自检结果及现有实验局限 |

## 项目结构

```text
.
├── ngram_lm.py            # 模型核心：计数、平滑、推理与模型读写
├── clean_corpus.py        # PKU 标注语料清洗
├── train_ngram.py         # 训练与验证集评测入口
├── generate.py           # 文本生成入口
├── selftest_ngram_lm.py   # 小语料正确性自检
├── run_all.sh            # 批量实验入口
├── run_experiments.py    # 自研模型与生成实验矩阵
├── compare_toolchains.py # SRILM / KenLM 对比
├── requirements.txt      # Python 依赖
├── EXPERIMENTS.md         # 实验手册
├── TOOLCHAIN.md           # 外部工具配置
└── results/              # 已保存的统计、日志与图表
```

其余脚本分别负责完整验证集评测（`ppl_table.py`）、生成参数扫描（`gen_sweep.py`）、绘图（`plot_results.py`）、标签对照（`tag_ablation.py`）和环境记录（`check_env.sh`）。

## 已知限制

- **规模与覆盖率：** 自研实现采用 16 bit 词 id，词表最多 65,533 项，包含 3 个特殊符号；其余词映射为 `<unk>`。训练与推理结构驻留内存，高阶模型的资源开销较大。
- **平滑与解码：** KN 采用每阶单一折扣，未实现修改版 KN。SB 分数不构成归一化概率分布，当前采样累加各阶分数，与最高阶命中的打分语义不一致。
- **清洗与标签对照：** `次/q/m` 等双层后缀仅去掉最外层，现有训练语料残留 5 个 `次/q`；标签对照的切句、归一化及前缀编码尚未完全对齐。
- **指标可比性：** 不同词表的 PPL 不宜直接排名；SRILM 对比值仅重算分母，未补回零概率 OOV 的代价；bits/字为近似换算。SB 分数指数不能作为概率困惑度评价参数优劣。
- **验证范围：** 自检覆盖小型、未剪枝的 3-gram 语料，尚未验证 SB 采样与逐词打分的一致性，也不能代替全部真实语料配置的验证。
- **运行平台：** 训练入口使用 `resource`，部分工具脚本使用 macOS 的 `/usr/bin/time -l`；在 Linux 运行这些脚本时需调整计时选项，当前不支持直接在 Windows 运行。

## 参与贡献

欢迎通过 [Issues](https://github.com/Highsun/n-gram-LM/issues) 报告问题或提出改进建议，也欢迎提交 Pull Request。问题报告请包含运行环境、命令及最小复现样例；涉及模型计算的修改请附上与参考实现的对照验证。请勿提交原始语料、大型模型或本地环境文件。

## 许可证

项目代码采用 [MIT License](LICENSE)。外部语料、SRILM 和 KenLM 遵循各自的许可。
