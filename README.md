# data_filtering

面向机器人学习数据集（LeRobot 风格）的七阶段质检流水线。包名是 `check_pipeline`，源码放在本仓库根目录。

**只检测和记录，不删除任何数据。** 帧级命中汇总成删帧工单，写在每条结果 JSON 的 `frame_deletion` 里，交给下游写出器。检测可以反复重跑，删除不可逆，这两件事分开。写出器还没写，见文末。

阶段 0 是 `probe_video()`：逐路解码，每路只解一次。下面七个检测都在这次解码的结果上跑。

## 七个检测

| # | 阶段 | 检测 | 级别 | 看什么 | 触发后 |
|---|---|---|---|---|---|
| 1 | 1 | `corrupt` | episode | 文件能不能用、各模态是否对齐 | 整条删除 |
| 2 | 1 | `black` | episode | 画面是不是黑的 | 整条删除 |
| 3 | 1 | `occlusion` | episode | 镜头是不是被糊住了 | 整条删除 |
| 4 | 1 | `stillness` | episode | 机器人是不是长时间没动 | 整条删除 |
| 5 | 2 | `saa` | episode | 动作指令和状态变化对不对得上 | 整条删除 |
| 6 | 2 | `sudden_change` | 帧 + episode | 信号有没有突然跳变 | 少量删帧，超限整条删除 |
| 7 | 2 | `state_extreme` | 帧 | 数值有没有离谱到会带偏归一化 | 删帧 |

**只有 `sudden_change` 和 `state_extreme` 会请求删帧。** 其余五个要么放行，要么整条删除。规则码、阈值和「为什么是这个数」见 [pipeline.md](pipeline.md)。

`corrupt` 是闸门：它确认「第 i 个视频帧 == 第 i 行动作数据」。对齐类规则命中时，帧级工单标成不可执行。默认 `short_circuit=False`，即使已经判了整条删除，后面的检测也跑完，用来做交叠统计。阈值冻结后再打开短路。

一条结果有三种互斥终态：`failed`（没有结论，不进删除率分母）、`drop_episode`（整条删除）、两者都为假（保留；工单 `actionable` 时按工单删帧）。`run_episode()` 不抛异常，失败写在结果字段里。

## 安装

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 里的依赖：

| 包 | 用途 |
|---|---|
| `numpy>=1.20` | 各检测的数组运算 |
| `scipy>=1.7` | `savgol_filter`、`medfilt` |
| `pandas>=1.3` | 读 parquet |
| `pyarrow>=6.0` | pandas 的 parquet 后端 |
| `av>=9.0` | `probe.py` 里的视频解码 |

检测模块本身只用 numpy / scipy。parquet 和视频解码分别只在 `signals.py` 和 `probe.py` 里用。非 parquet 数据集可以不装 pandas / pyarrow，由格式适配器自己读信号，经 `signals=` 传入。

### 怎么让 `import check_pipeline` 成功

文档和代码里的导入是：

```python
from check_pipeline.pipeline import run_episode, DropBudget
from check_pipeline.manifest import EpisodeAssets
```

模块之间用相对导入（`from .pipeline import ...`）。本仓库没有 `setup.py` 或 `pyproject.toml`。`__init__.py`、`pipeline.py`、`run_dataset.py` 等文件直接在仓库根目录，克隆下来的目录名是 `data_filtering`，Python 不会把它认成包 `check_pipeline`。

克隆时把目录命名为 `check_pipeline`，把**上一级**放进 `PYTHONPATH`：

```bash
git clone https://github.com/liujzh33/data_filtering.git check_pipeline
cd check_pipeline
python -m pip install -r requirements.txt
cd ..
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

python -m check_pipeline.run_dataset --help
```

仓库如果已经在别的目录名下，在它的上一级做一个同名符号链接，效果相同：

```bash
# 当前目录是仓库的上一级，仓库目录叫 data_filtering
ln -sfn data_filtering check_pipeline
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
```

入口是 `python -m check_pipeline.run_dataset`。相对导入依赖包名 `check_pipeline`。

## 快速开始

### 批量 CLI

`run_dataset.py` 读 LeRobot 布局：路径从 `meta/info.json` 的模板解析（按 `chunks_size` 分桶），相机名取 `features` 里 `dtype == "video"` 的项。路径以 `obs://` 开头时，`read_bytes` 会按需 `import moxing`（不在 `requirements.txt` 里）。

```bash
python -m check_pipeline.run_dataset \
    --root    /path/to/dataset \
    --profile dataset_profile.json \
    --out     qc_out/ \
    --limit   50
```

| 参数 | 作用 |
|---|---|
| `--root` | 数据集根目录，内含 `meta/`、`data/`、`videos/` |
| `--profile` | `dataset_profile.json`。顶层键 `_params` 会被拆出来传给各检测，其余作为 profile |
| `--out` | 输出目录 |
| `--limit` | 只跑前 N 条；`0`（默认）为全部 |
| `--workers` | episode 级进程数，默认 `1` |
| `--short-circuit` | 判整条删除后跳过剩余检测。调阈值时保持默认（关闭），见 [pipeline.md](pipeline.md) §3 |

`stillness` 的 `state_diff_thresh` 没有默认值。CLI 从 profile 顶层的 `_params` 读取，例如 `"_params": {"stillness": {"state_diff_thresh": 1e-3}}`。`saa`、`sudden_change`、`state_extreme` 各要一段完整配置，字段见 [dataset_profile.md](dataset_profile.md)。缺段时该检测报 `UnsupportedDataError`，原文进这条结果的 `errors`，其余检测照常出结论。

跑完后数据集原样保留，`qc_out/` 里有：

- `episode_000000.json`：每条一份，删帧工单在 `frame_deletion`
- `summary.json`：`DropBudget` 汇总（总删除率、唯一归因、帧级段数）
- `report.txt`：人读的报告

hdf5 / jsonl 没有现成 CLI。写一个同构入口，把信号读成 `{profile 里声明的 column 名: (T, D) 矩阵}`，用 `signals=` 传给 `run_episode`。检测模块不用改。

### 单条 Python

下面的路径是一条 LeRobot episode 的例子。批量跑用上面的 CLI，路径由 `info.json` 生成。

```python
import json

from check_pipeline.manifest import EpisodeAssets
from check_pipeline.pipeline import DropBudget, run_episode

with open("dataset_profile.json", encoding="utf-8") as f:
    profile = json.load(f)

def read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

result = run_episode(
    videos={
        "head": read("videos/chunk-000/head/episode_000123.mp4"),
        "left_wrist": read("videos/chunk-000/left_wrist/episode_000123.mp4"),
    },
    parquet_bytes=read("data/chunk-000/episode_000123.parquet"),
    episode_id="ep_000123",
    profile=profile,
    assets=EpisodeAssets(
        data_format="lerobot",
        table_path="data/chunk-000/episode_000123.parquet",
        video_paths={
            "head": "videos/chunk-000/head/episode_000123.mp4",
            "left_wrist": "videos/chunk-000/left_wrist/episode_000123.mp4",
        },
        meta_paths={"episodes": "meta/episodes.jsonl"},
    ),
    params={"stillness": {"state_diff_thresh": 1e-3}},
)

budget = DropBudget()
budget.add(result)
print(budget.format_report())
# result.to_dict()["frame_deletion"] 是删帧工单。这次调用不改任何数据文件。
```

`videos` 的值可以是 `None`，表示文件缺失（`corrupt` 报 `VIDEO_FILE_MISSING`）。产生删帧时要提供 `assets`，否则工单列不全目标，标成不可执行。

## 仓库布局

```
.
├── __init__.py                 包入口（导入名 check_pipeline）
├── pipeline.py                 run_episode、EpisodeResult、DropBudget
├── run_dataset.py              LeRobot 批量 CLI
├── manifest.py                 EpisodeAssets、删帧工单
├── probe.py                    视频解码（阶段 0）
├── signals.py                  从 parquet 取信号矩阵
├── errors.py                   PipelineEnvironmentError / UnsupportedDataError
├── intervals.py                帧区间合并
├── corrupt.py  black.py  occlusion.py  stillness.py
├── saa.py  sudden_change.py  state_extreme.py
├── tools/scan_percentiles.py              扫全数据集，算 q1 / q99 / range
├── tools/build_state_extreme_profile.py   把已有分位文件投影成 state_extreme 片段
├── tests/                      python -m check_pipeline.tests.test_pipeline 等
├── requirements.txt
├── pipeline.md                 流水线总览
├── dataset_profile.md          profile schema 与 stage 0 方案
├── prompts/stage0_survey.md    给数据侧模型的探查提示词
└── corrupt.md  black.md  occlusion.md  stillness.md  saa.md  sudden_change.md  state_extreme.md
```

## 更细的文档

- [pipeline.md](pipeline.md) — 阈值含义、执行顺序、结果 JSON、删帧工单、三种终态、第一批怎么看 `DropBudget` 报告
- [dataset_profile.md](dataset_profile.md) — 接入新数据集前的 stage 0：哪些用脚本测、哪些问模型、profile 怎么冻结
- [prompts/stage0_survey.md](prompts/stage0_survey.md) — 探查提示词。`dataset_profile.json` 是七个检测的配置来源
- 各检测：[corrupt.md](corrupt.md)、[black.md](black.md)、[occlusion.md](occlusion.md)、[stillness.md](stillness.md)、[saa.md](saa.md)、[sudden_change.md](sudden_change.md)、[state_extreme.md](state_extreme.md)

分位数扫描已经实现。它产出各维的 `q1` / `q99` / `range`，供 profile 使用。`dataset_facts.json` 的扫描仍待实现，见下一节。

```bash
python -m check_pipeline.tools.scan_percentiles \
    --root /path/to/dataset \
    --columns action observation.state \
    --out scan.json
```

## 还没做

对应 [pipeline.md](pipeline.md) §9，本仓库里仍然缺这些：

- **删帧写出器。** 流水线只产出 `frame_deletion` 工单。执行删除，以及写完后的校验（视频帧数等于表格行数、`frame_index` 连续、重跑 corrupt、stats 一致，不过就回滚），都还没有。
- **stage 0 扫描脚本。** `dataset_facts.json` 的确定性扫描还没写，方案在 [dataset_profile.md](dataset_profile.md)。`tools/scan_percentiles.py` 只算分位数。
- **合并视频的帧偏移。** `video_granularity` 为 `per_chunk`（一个 mp4 里装多条 episode）时，`probe_video` 还不能按偏移切片。当前读取层按一条 episode 一个 mp4 处理。

[pipeline.md](pipeline.md) §9 还提到旧目录 `new/`（`new/occlusion.py` 等）没有被新流水线引用。本仓库里没有 `new/` 目录。
