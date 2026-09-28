# 隔离安装 SRILM / KenLM（可整目录删除）

本项目本身只依赖 Python + numpy，但 `run_srilm.sh` / `run_kenlm.sh` 需要两个 C++ 工具。为了**不污染现有的 conda 环境与 Homebrew**，它们被装在单个自包含目录里：

```
~/ngram-toolchain/                 # 唯一的新增目录，删掉它 = 完全卸载
├── envs/deps/                     # conda prefix 环境：cmake / boost-cpp / eigen / zlib / bzip2 / xz
├── pkgs/                          # conda 包缓存（用 CONDA_PKGS_DIRS 指到这里，不写 ~/miniconda3/pkgs）
├── src/
│   ├── kenlm/                     # 源码 + build/bin/{lmplz,build_binary,query,…}
│   └── srilm/                     # 源码 + 就地编译，bin/{ngram-count,ngram,…}
├── logs/                          # 构建日志
└── downloads/                     # 源码压缩包
```

实测总占用 **856 MB**（envs 393 MB + src 298 MB + pkgs 163 MB + logs 1.4 MB）。

隔离要点：

* 只创建**一个** conda 环境，而且是 `-p` 前缀环境（不在 `~/miniconda3/envs/` 下）， `base` / `ai` / `hands-on-rl` 的包、numpy 版本、PATH 全都不动；
* conda 包缓存改指到工具链目录内，不往 `~/miniconda3/pkgs` 写；
* SRILM、KenLM 都是"就地编译"，安装前缀就是工具链目录内的路径，不写 `/usr/local`；
* 唯一在工具链目录之外留下的痕迹是 conda 的全局环境登记表 `~/.conda/environments.txt` 多出的一行（卸载时用 `conda env remove -p` 会一并删掉）。

## 怎么装（本机实际执行的步骤，可复现）

```bash
ROOT="$HOME/ngram-toolchain"
mkdir -p "$ROOT"/{pkgs,envs,src,downloads,logs}

# ---------- 1) 隔离的构建依赖（conda-forge，只装构建需要的东西）----------
export CONDA_PKGS_DIRS="$ROOT/pkgs"
conda create -y -p "$ROOT/envs/deps" --override-channels -c conda-forge \
    cmake boost-cpp eigen zlib bzip2 xz

# ---------- 2) KenLM（cmake + boost + eigen，多线程，编译约 1 分钟）----------
git clone --depth 1 https://github.com/kpu/kenlm.git "$ROOT/src/kenlm"
DEPS="$ROOT/envs/deps"
cd "$ROOT/src/kenlm" && mkdir -p build && cd build
"$DEPS/bin/cmake" .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$DEPS" -DKENLM_MAX_ORDER=6
make -j4          # 产出 build/bin/{lmplz, build_binary, query, filter, kenlm_benchmark, ...}

# ---------- 3) SRILM（官网 speech.sri.com 不可达，用 BitSpeech 镜像；RELEASE 1.7.3）----------
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/BitSpeech/SRILM.git "$ROOT/src/srilm"
export SRILM="$ROOT/src/srilm"
cd "$SRILM"
# 3a) 只编 misc/dstruct/lm 三个模块：自带的 zlib 模块在现代 clang 下编不过，
#     而我们只需要 ngram-count / ngram（都在 lm 里）
make World -j4 SRILM="$SRILM" MODULES="misc dstruct lm"
# 3b) 补上链接需要的 libz.a（用隔离环境里 conda-forge 的 zlib，避免动 SRILM 源码）
cp "$DEPS/lib/libz.a" "$SRILM/lib/macosx/libz.a"
cd "$SRILM/lm/src" && make SRILM="$SRILM" MACHINE_TYPE=macosx programs -j4
make SRILM="$SRILM" MACHINE_TYPE=macosx release-programs
# 3c) 在 $SRILM/bin 下建软链接，使 $SRILM/bin/ngram-count 可用（脚本按这个路径找）
cd "$SRILM/bin" && for f in macosx/*; do b=$(basename "$f"); [ -e "$b" ] || ln -s "$f" "$b"; done
```

## 怎么用

```bash
cd code
SRILM="$HOME/ngram-toolchain/src/srilm" bash run_srilm.sh
KENLM="$HOME/ngram-toolchain/src/kenlm" bash run_kenlm.sh
KENLM="$HOME/ngram-toolchain/src/kenlm" SRILM="$HOME/ngram-toolchain/src/srilm" \
    python compare_toolchains.py --order 3 5 --with-pruning   # 三方对比表
bash check_env.sh        # 会打印工具链位置与版本
```

实测结果见 [`results/toolchain_comparison.md`](results/toolchain_comparison.md)、 [`results/srilm_timing.txt`](results/srilm_timing.txt)、 [`results/kenlm_timing.txt`](results/kenlm_timing.txt)。

## 怎么（干净地）卸载

```bash
# 1) 注销 conda 前缀环境（会同时删掉 ~/.conda/environments.txt 里对应的一行）
conda env remove -p "$HOME/ngram-toolchain/envs/deps"

# 2) 删除整个工具链目录（SRILM/KenLM 源码、编译产物、包缓存全在里面）
rm -rf "$HOME/ngram-toolchain"

# 3) 可选：清掉可能残留的 conda 包缓存（若你之前没有设置 CONDA_PKGS_DIRS）
#    conda clean -p -y

# 4) 验证：下面几条都应该没有输出/没有该目录
ls "$HOME/ngram-toolchain" 2>/dev/null; conda env list | grep -c ngram-toolchain
```

卸载后，`run_srilm.sh` / `run_kenlm.sh` 仍可用 `--dry-run` 查看命令（不会报错），自研的 `ngram_lm.py`、`train_ngram.py`、`generate.py` 完全不受影响。

## 注意事项

* SRILM 的许可证只允许研究用途、不允许再分发，因此这里用源码编译、不把二进制放进仓库。
* 三个 shell 脚本里的 `/usr/bin/time -l` 是 **macOS/BSD 语法**（Linux 上要用 `-v`，且部分发行版需要额外安装 `time` 包）；`check_env.sh` 里的 `sysctl` 在 Linux 上会回退到 `nproc`。
* `train_ngram.py` 用到了 `resource` 模块，整个项目面向 macOS / Linux，不支持 Windows。
