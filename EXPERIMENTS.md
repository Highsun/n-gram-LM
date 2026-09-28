# 实验运行手册：怎么跑通、看哪些指标、有哪些图

这份文档只回答三件事：**怎么跑**、**产出什么**、**指标怎么读**。结论与分析不在本文档范围内，写实验报告时可以直接按第 4 节的指标字典取数、按第 5 节的图目录贴图。

## 0. 前置准备

```bash
python -m pip install -r requirements.txt  # 从仓库根目录运行；也可使用已有 Python 环境
python -c "import numpy, matplotlib; print(numpy.__version__, matplotlib.__version__)"
```

* 原始标注语料放在 `data/`（`199801.txt` … `199806.txt`），`run_all.sh` 会自动清洗；
* SRILM / KenLM 是**可选**的（不装也能跑完自研部分与全部绘图），安装方式见 [TOOLCHAIN.md](TOOLCHAIN.md)；
* 如果 `~/.matplotlib` 不可写（容器/沙箱里常见），脚本会自动把缓存改到临时目录，也可以自己 `export MPLCONFIGDIR=/tmp/mplconfig`。

## 1. 一条命令跑通

```bash
# 从克隆后的仓库根目录运行（本地总项目先 cd code）
bash run_all.sh                  # 全流程；已有的训练产物默认跳过
bash run_all.sh --force          # 连训练一起重跑
bash run_all.sh --skip-tools     # 不碰 SRILM / KenLM（没装时用这个）
```

| 步骤 | 做什么 | 产物 | 本机实测耗时 |
|---|---|---|---|
| 0/7 | 记录环境（python/numpy/工具链/语料规模） | `results/environment.txt` | 1 s |
| 1/7 | 清洗语料：去句子编号、词性、实体方括号，切句，按文章划分 train/dev | `corpus/*.txt`、`corpus/stats.json` | 11 s |
| 2/7 | 自研实现的训练矩阵 + 生成矩阵 | `models/*.npz`、`results/training_time.md`、`results/generation_samples.md`、`results/logs/*.log` | 6.5 min（首次） |
| 3/7 | KenLM 训练/转换/困惑度（装了才跑） | `models/kenlm.*`、`results/kenlm_timing.txt` | 30 s |
| 4/7 | SRILM 三种平滑 + 熵剪枝 + 采样（装了才跑） | `models/srilm.*.arpa`、`results/srilm_timing.txt` | 2.5 min |
| 5/7 | 三方同语料对比（含剪枝/量化变体） | `results/toolchain_comparison.md`、`toolchain_runs.jsonl` | 4 min |
| 6/7 | 完整验证集上的困惑度总表 + 生成参数扫描 | `results/ppl_full.{md,json}`、`results/generation_sweep*.{jsonl,md}` | 2.5 min |
| 7/7 | 绘图 | `results/figures/*.png` | 15 s |

> 另有一个可以单独运行的对照实验：`python tag_ablation.py`（约 1 分钟），它保留词性标签重训一个模型，用来展示词表膨胀与生成标注泄漏；现有切句与前缀编码未完全对齐，不能量化去标签的质量收益，产物是 `results/tag_ablation.{md,json}` 与图 11。

## 2. 分步执行

| 想做什么 | 命令 | 产物 |
|---|---|---|
| 只清洗语料 | `python clean_corpus.py --input-dir data --out-dir corpus --unit both` | `corpus/word.*.txt`、`corpus/char.*.txt` |
| 训练一个模型 | `python train_ngram.py --train corpus/word.train.txt --dev corpus/word.dev.txt --order 5 --out models/word.kn5.npz` | 模型 + 同名 `.json` 统计 |
| 续写 / 看候选词 | `python generate.py --model models/word.kn5.npz --prefix "在阳光明媚的五月，我们学校胜利召开了" --temperature 0.7 --top-k 5` | 终端输出（`--json-out` 可落盘） |
| 训练矩阵 + 生成矩阵 | `python run_experiments.py --stage all` | `results/training_time.md`、`generation_samples.md` |
| 自研 vs KenLM vs SRILM | `KENLM=… SRILM=… python compare_toolchains.py --order 3 5 --with-pruning` | `results/toolchain_comparison.md` |
| 完整验证集 PPL 总表 | `python ppl_table.py` | `results/ppl_full.{md,json}` |
| 温度/top-k 扫描 | `python gen_sweep.py --seeds 10` | `results/generation_sweep*.{jsonl,md}` |
| 去标签对照实验 | `python tag_ablation.py` | `results/tag_ablation.{md,json}` |
| 出图 | `python plot_results.py` | `results/figures/*.png` |
| 正确性自检 | `python selftest_ngram_lm.py` | 打印概率误差与归一化检查 |

## 3. 产物清单

| 文件 | 内容 | 生成者 |
|---|---|---|
| `results/training_time.md` | 19 个自研配置的训练耗时（分阶段）、峰值内存、模型体积、PPL | `run_experiments.py` |
| `results/ppl_full.md` / `.json` | **统一口径**下每个模型在完整验证集上的 PPL、OOV、bits/token、bits/字 | `ppl_table.py` |
| `results/toolchain_comparison.md` / `toolchain_runs.jsonl` | 自研 / KenLM / SRILM（含剪枝、量化变体）的对比表 | `compare_toolchains.py` |
| `results/srilm_timing.txt` / `kenlm_timing.txt` | 两个工具的实测记录、遇到的问题与规避方式 | `run_srilm.sh` / `run_kenlm.sh` + 手工整理 |
| `results/generation_samples.md` | 不同阶数 / 平滑 / 解码方式下的续写样例 | `run_experiments.py` |
| `results/generation_sweep_summary.md` / `generation_sweep.jsonl` | 温度 × top-k 的量化指标与样例 | `gen_sweep.py` |
| `results/tag_ablation.md` / `.json` | 保留词性标签的代价：词表膨胀、标签预测交叉熵、碎片化、生成标注泄漏 | `tag_ablation.py` |
| `results/figures/*.png` | 11 张图，见第 5 节 | `plot_results.py` |
| `results/logs/*.log` | 每次训练的分阶段日志；第三方工具的 stderr（含 lmplz 的 `=== 1/5 … 5/5 ===` 进度） | 各脚本 |
| `results/environment.txt` | 机器/环境/语料规模快照 | `check_env.sh` |

## 4. 指标字典（写报告时按这张表取数）

### 4.1 语言模型质量

| 指标 | 定义 | 方向 | 读的时候注意 |
|---|---|---|---|
| PPL（困惑度） | `10^(-总log10概率 / N)`，`N = 验证词数 + 句末 </s> 数` | 越低越好 | **必须同词表、同 OOV 才算可比**；自研把词表截到 65,533、KenLM/SRILM 保留全部 14 万词型，dev OOV 分别是 3,226 / 1,992 |
| logprob/token | 平均对数概率（nats） | 越高越好（= PPL 越低） | 与 PPL 一一对应，只是换了个尺度 |
| bits/token | `-logprob/ln2` | 越低越好 | 单位是"每个 token 需要多少比特" |
| **bits/字** | `bits/token ÷ 每 token 字数`（词级 1.644 字/词，字级 1.0） | 越低越好 | 使用训练集平均字数/词的近似换算；没有严格对齐 EOS 分母和 OOV 丢失的字面信息，不能据此判断粒度优劣 |
| OOV 率 | 不在建模词表里的验证 token 占比 | 只作说明 | 它同时受词表大小和 `<unk>` 处理策略影响，换词表就会变 |
| 提前结束比例 | 生成在 `max_words` 之前碰到 `</s>` 的比例 | 视任务而定 | 与"平均长度"一起看 |

### 4.2 训练代价与模型规模

| 指标 | 定义 | 读的时候注意 |
|---|---|---|
| 训练耗时（分段） | 词表统计 / 编码 / 滑窗 / 各阶计数 / 平滑构建 | 自研按阶段记录；KenLM、SRILM 只有总时间（`/usr/bin/time -l`） |
| 峰值内存 | 进程 `ru_maxrss` | macOS 上单位是字节（其它平台是 KB），脚本里已换算成 MB |
| 模型体积 | 磁盘上的 `.npz` / `.trie` / `.arpa` | **格式不同不能直接比**：npz 是 zlib 压缩的整数表，trie 未量化，ARPA 是纯文本 |
| n-gram 类型数 | 每一阶不同的 n-gram 个数 | 与 token 数一起看才能体现稀疏度 |
| singleton 比例 | 只出现一次的 n-gram 占比 | 词级 5-gram 达 94.9%，是"高阶收益递减 + 需要剪枝"的直接证据 |
| 剪枝/量化权衡 | 同一模型在"体积 ↓"与"PPL ↑"之间的取舍 | 三条曲线可对比：自研 count 剪枝、KenLM `--prune`/`-q`、SRILM 熵剪枝 |

### 4.3 生成质量

| 指标 | 定义 | 读的时候注意 |
|---|---|---|
| 平均长度 | 生成 token 数 | 与"提前结束比例"互为补充 |
| 模型 logprob/token | 生成文本在**同一模型**下的平均对数概率 | 越高说明越"像训练语料"；温度升高会快速下降 |
| TTR（类符/形符比） | 排除脚本列出的部分标点后的去重 token 数 / token 数 | 受长度和标点过滤方式影响；不能替代语法、语义和主题相关性评价 |
| 重复率 | `1 - TTR` | 与 TTR 等价地反映保守程度 |
| `<unk>` 率 | 生成里未知词占比 | 正常情况下应为 0（生成时已禁用 `<unk>`） |

### 4.4 跨工具比较的三个前提

1. **同一训练语料 + 同一验证集**：本文档里统一用 `corpus/word.{train,dev}.txt`；
2. **同一评测口径**：自研/KenLM 含 OOV 与 EOS；当前 SRILM 跳过零概率 OOV，对比表仅重算分母，没有补回未知词代价，尚不可严格横比；
3. **对齐 `<unk>` 处理**：词表大小与 `<unk>` 质量都会显著影响 PPL。字级 OOV 仅 7 个，是更接近的对照；KenLM 的 `--interpolate_unigrams 0` 仅用于检查一元插值与未知词设置的影响，不能视为三方概率完全对齐。

## 5. 图目录（`results/figures/`）

| 图 | 画的是什么 | 数据来源 | 建议放报告的哪一节 |
|---|---|---|---|
| `fig1_order_vs_ppl_and_sparsity.png` | 阶数 vs 困惑度；阶数 vs singleton 比例 | `ppl_full.json` | 主结果：阶数的影响 |
| `fig2_training_time_breakdown.png` | 训练耗时分解（堆叠柱） | `train_runs.jsonl` | 训练过程与时间开销 |
| `fig3_memory_and_model_size.png` | 峰值内存 / 模型体积 vs 阶数 | `ppl_full.json` | 资源开销 |
| `fig4_corpus_scaling.png` | 语料规模 vs PPL（双轴：PPL/耗时） | `train_runs.jsonl` | 数据规模的影响 |
| `fig5_size_ppl_tradeoff.png` | 体积–PPL 权衡散点（剪枝/量化） | `ppl_full.json` + `toolchain_runs.jsonl` | 剪枝与量化的取舍 |
| `fig6_three_tool_comparison.png` | 三工具在 3/5-gram 上的 PPL、耗时、体积 | `toolchain_runs.jsonl` | 工具对比 |
| `fig7_bits_per_unit.png` | bits/token 与近似 bits/字（跨粒度解读有限制） | `ppl_full.json` | 词级 vs 字级 |
| `fig8_generation_sweep.png` | 温度/top-k vs 质量、多样性、长度 | `generation_sweep.jsonl` | 生成参数分析 |
| `fig9_unk_effect_on_ppl.png` | `<unk>` 口径对 PPL 的影响（含 OOV 标注） | `toolchain_runs.jsonl` | 评测口径的注意事项 |
| `fig10_corpus_profile.png` | 语料画像：句长分布、词性分布、高频词 | `corpus/` + `stats.json` | 数据一节 |
| `fig11_tag_ablation.png` | 保留标签的代价：词型数对比 + 标签预测交叉熵 | `tag_ablation.json` | 数据清洗的必要性 |

## 6. 复现注意事项与现有局限

* **PPL 分母**：自研/KenLM 是"词数 + 句末数"，SRILM 在有 OOV 时口径不同——分母重算写在 `compare_toolchains.py` 里，但没有补回零概率 OOV 的代价；`ppl_table.py` 保证所有自研模型在同一份完整验证集上评测（字级验证集有 243,485 个 token，注意别被 `--ppl-limit` 截断）。
* **KenLM**：`query` 的选项必须写在模型文件**前面**；当前 master 的 `lmplz` 没有 `-v`； `lmplz` 不接受输入文本中出现 `<unk>`（会抛异常），所以"先把语料映射成 `<unk>` 再训练 KenLM"这条路走不通。
* **SRILM**：`-kndiscount` 是修改版 KN、`-ukndiscount` 才是单折扣原始 KN；`-prune` 是**熵剪枝**， `-gtNmin` 属于 Good-Turing 折扣参数（在 `-kndiscount` 下无效）；"先剪掉全部 singleton 再估 KN" 会因为 count-of-counts 为零直接报错；`-ppl` 加 `-debug 1` 会把每句都打印出来。
* **生成**：所有采样固定随机种子；`min_words` 之前禁止 `</s>`（KN 的一元基分布偏向句末），并禁用 `<unk>`；SRILM 的 `-gen-prefixes` 输出只含续写部分，不回显前缀。
* **中文字体**：`plot_results.py` 会自动挑选系统里的中文字体（本机选中 PingFang SC），找不到时回退 DejaVu Sans（中文会显示成方框，请换机器时注意）。

## 7. 参考：核心数字速查

自研取 `ppl_full.json`，第三方取 `toolchain_runs.jsonl`，耗时为不同实验运行的单次记录。MB 标签按 MiB 换算；SRILM 评测值仅重算分母。

| 配置 | 训练耗时 | 模型体积 | PPL / 分母重算值 |
|---|---|---|---|
| 词级 3-gram KN（自研） | 11.23 s | 17.4 MB | 198.88 |
| 词级 5-gram KN（自研） | 22.47 s | 75.6 MB | 184.74 |
| 词级 5-gram KN + 剪枝（自研） | 18.15 s | 21.1 MB | 236.51 |
| 字级 3-gram KN（自研） | 14.42 s | 10.7 MB | 39.76 |
| 字级 5-gram KN（自研） | 31.07 s | 67.6 MB | 31.84 |
| KenLM 词级 3-gram | 2.74 s | 52.8 MB（trie） | 254.01 |
| KenLM 词级 5-gram + `--prune 0 0 1 1` | 2.84 s | 41.4 MB（trie） | 256.69 |
| SRILM 词级 5-gram（修改版 KN） | 18.80 s | 90.3 MB（ARPA） | 209.74（自报 225.33） |
| SRILM 词级 5-gram + 熵剪枝 `1e-7` | 19.91 s | 44.6 MB | 236.57（自报 254.57） |

现有清洗仍有 5 个双层标注残留；SB 采样累加低阶项，与最高阶命中的打分语义不同；标签对照切句、归一化和前缀编码未对齐。整理时保留原算法与实测结果，复核见 `results/verification.json`，清洗统计见 `results/corpus_stats.json`。修正算法后须重跑受影响实验。
