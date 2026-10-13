# 七阶段质检流水线 —— 总览

**代码**：`check_pipeline/pipeline.py`、`check_pipeline/manifest.py`
**入口**：`run_episode(...) → EpisodeResult`、`DropBudget`

**整条流水线只检测和记录，不删任何数据。** 帧级命中汇总成一张工单（manifest），交给下游写出器执行。检测可以重跑一万次不改数据，删除是不可逆的——这两件事必须分开。

---

## 1. 七个检测一览

| # | 阶段 | 检测 | 级别 | 看什么 | 规则码 | 触发后 |
|---|---|---|---|---|---|---|
| 1 | 1 | `corrupt` | episode | 文件能不能用、各模态是否对齐 | 8 条（见下） | 整条删 |
| 2 | 1 | `black` | episode | 画面是不是黑的 | `BLACK_RUN_RATIO_EXCEEDED` | 整条删 |
| 3 | 1 | `occlusion` | episode | 镜头是不是被糊住了 | `OCCLUSION_RUN_RATIO_EXCEEDED` | 整条删 |
| 4 | 1 | `stillness` | episode | 机器人是不是长时间没动 | `STILLNESS_RUN_RATIO_EXCEEDED` | 整条删 |
| 5 | 2 | `saa` | episode | 动作指令和状态变化对不对得上 | `SAA_DA_BELOW_THRESHOLD`<br>`SAA_CAUSALITY_VIOLATED` | 整条删 |
| 6 | 2 | `sudden_change` | **帧 + episode** | 信号有没有突然跳变 | `SUDDEN_CHANGE_SHORT`<br>`SUDDEN_CHANGE_LONG` | 删帧<br>整条删 |
| 7 | 2 | `state_extreme` | **帧** | 数值有没有离谱到会带偏归一化 | `STATE_EXTREME_FRAMES` | 删帧 |

**只有两个检测会产生删帧**：`sudden_change` 和 `state_extreme`。其余五个要么放行，要么整条删——不存在"删一半"的中间态。

corrupt 的 8 条规则：`VIDEO_FILE_MISSING`、`VIDEO_UNDECODABLE`、`VIDEO_NO_READABLE_FRAME`、`FRAME_DECODE_FAILED`、`FRAME_SIZE_CHANGED`、`VIEW_FRAME_COUNT_MISMATCH`、`ACTION_VIDEO_LENGTH_MISMATCH`、`PARQUET_UNREADABLE`。全部是 `drop_episode`——**只要出现坏帧就整条删**，不做"坏得少就删帧"。

---

## 2. 每个阈值是什么意思，怎么定的

### 阶段 1：四个看画面的

**`black`：`mean_abs=18`，`std_abs=8`，`max_black_run_ratio=0.10`**

判一帧是不是黑：灰度均值 < 18 **并且** 灰度标准差 < 8。

为什么要两个条件而不是只看均值：单看均值会把"暗但有内容的场景"（夜间、背光）也当成黑屏。加上标准差这个条件，要求画面**又暗又平**——真正的信号丢失（掉线、盖镜头盖）是一整片均匀的黑，标准差趋近 0；暗场景再暗也有明暗结构，标准差还在十几以上。

`18` 这个数不是拍脑袋：视频编码有 limited range（TV range）的概念，纯黑在里面编码成 16 而不是 0，解码回来在 16 左右浮动。阈值定在 18 是为了把这种"编码意义上的黑"也算进去。

`0.10` 是判 episode 的：**任意一路相机**上，**最长的连续**黑帧段达到该 episode 总帧数的 10%，就整条删。

> 强调**连续**而不是总量：分散在全片的 10% 黑帧（偶发丢帧）和开头连续 10% 的黑（开机没准备好）是完全不同的两件事，后者才是真正需要删的。

**`occlusion`：`std_abs=8`，`mean_abs=18`，`max_occlusion_run_ratio=0.10`**

判据是 black 的补集：标准差 < 8（画面很平）**但** 均值 ≥ 18（不暗）。也就是"糊住了但不黑"——手挡住镜头、贴上白纸、严重失焦。两个检测共用同一组数字，是为了保证**任何一帧要么归 black 要么归 occlusion，不会两边都算或两边都漏**。

**`stillness`：`state_diff_thresh` 必须自己给，`image_diff_thresh=2.0`，`max_still_run_ratio=0.10`**

判一帧静止：相邻帧的机器人状态变化 < `state_diff_thresh` **并且** 画面变化 < `image_diff_thresh`。两个都要满足，因为单看画面会把"机器人在动但摄像头拍不到"误判成静止。

`state_diff_thresh` **故意没有默认值**——它的单位取决于机型（弧度？米？归一化值？），给一个默认值等于鼓励所有机型都用同一个错的数。不给就报错，逼你去实测：抽几条 episode，量一下"明显在运动的一段"和"明显静止的一段"的相邻帧差中位数，阈值取两者之间。

### 阶段 2：三个看信号的

**`saa`：`da_threshold=0.7`**

DA（方向一致性）衡量的是：状态的变化方向，和动作指令的变化方向，有多大比例是一致的。

直觉上：机器人被命令"关节往正方向转"，那它的状态就该往正方向变。两者方向一致的帧占比就是 DA。**DA = 1 表示完全对齐，DA = 0.5 表示纯随机**——因为方向只有正负两种，瞎猜也能对一半。

所以 `0.7` 的含义是"比随机水平高 40%"。Qwen 给的经验范围是 0.6–0.7，这里取上限，更严格。低于它说明时间戳错乱或丢包，整条删。

> 一个真实的坑：如果数据是**增量动作**（action 是变化量而不是目标位置）却没先积分，DA 实测掉到 0.496——几乎正好是随机水平，于是整个数据集会被判成不对齐而全删。所以 profile 里必须写明 `action_space` 是 `absolute` 还是 `delta`。

**`sudden_change`：`jump_frac=0.02`，`max_frame_drop=10`**

`jump_frac` 的含义非常直白：**一次跳变的幅度超过该维度量程的百分之几，算异常**。0.02 就是 2%。

能做到这么直白是这次重写的核心：原始的三个偏差信号（残差、加速度、跃度）单位互不相同、量级差几十倍，没法用一个数统一卡。现在把三者都归一化到"等效单次跳变幅度"，于是只剩一个参数，而且这个参数的单位就是人能想象的"跳了多大"。

`max_frame_drop=10` 决定删帧还是删整条：异常总帧数 ≤ 10 帧就只删这几帧（`SUDDEN_CHANGE_SHORT`），超过就整条删（`SUDDEN_CHANGE_LONG`）。这直接对应 `table.txt`(46) 的要求——少量就删帧，大量就删 episode，用绝对帧数而不是百分比（1000 帧的 10% 是 100 帧，太多了）。

**`state_extreme`：`alpha=1.0`**

band = `[q1 − α·(q99−q1), q99 + α·(q99−q1)]`，落在外面的帧被标记。q1/q99 是**该数据集该机型**的全量 1/99 分位。

α 就是"在正常范围之外再放宽几倍"。α=0 就是 `[q1, q99]` 本身。

为什么默认 1.0 而不是 0：实测 α=0 会删掉 13.4% 的帧，其中 10.5% 的 episode 被删掉一半以上——而这些数据完全正常，只是在工作区边缘作业。α 从 0 抬到 0.5，删帧率掉 170 倍，而**真离群值的检出率一点没变**（100%）。

校准直觉：正态分布下 q1/q99 相当于 ±2.33σ，α=1 的 band 是 **±6.98σ**。够得着的只有数量级上的错误——传感器毛刺、单位搞错、坐标系错误。这正是这个检测该防的东西（目的是防止离群值带偏分位数归一化，不是修剪正常分布的尾巴）。

---

## 3. 执行顺序与依赖

```
[阶段 0]  probe_video()            逐路解码，每路只解一次（占单条耗时 95% 以上）
              |
[阶段 1]  corrupt                  gate：确立"第 i 个视频帧 == 第 i 行动作数据"
              |
          black / occlusion / stillness    看画面，三者互不依赖
              |
[阶段 2]  saa                      episode 级，不过就整条丢
              |
          sudden_change / state_extreme    帧级，产出删帧区间
```

### 为什么 corrupt 必须第一个跑

它确立的是**下标映射**这件事本身。后面所有检测都建立在"第 i 个视频帧 = 第 i 行动作数据 = 每一路相机的第 i 帧"之上：black 说"第 30–50 帧是黑的"，只有当各路视频和动作表共享同一条时间轴时才有意义。

corrupt 的 `VIEW_FRAME_COUNT_MISMATCH` / `ACTION_VIDEO_LENGTH_MISMATCH` 一旦命中，帧号就退化成一串没有所指的数字。所以**帧级删除工单在 corrupt 未过时一律标为不可执行**。

### 为什么 saa 排在两个帧级检测之前

saa 是 episode 级的，`sudden_change` / `state_extreme` 是帧级的。整条都不要了，再算删哪几帧没有意义。顺序放在前面，开了短路开关时能省掉后两个检测。

### 短路：默认关，阈值冻结后再开

`short_circuit=False`（默认）时，哪怕 corrupt 已经判了整条删除，后面六个检测照常跑完。

短路能省一点计算，但会让**交叠统计**永远做不了。而交叠统计回答的是：

```
七个检测各自命中率都是 5%，总删除率是多少？
  完全重合   →   5%    有六个检测是冗余的
  完全不相交 →  35%    数据有七类独立的问题
```

只看各自的命中率，这两种情况分不出来，而处置完全相反。调阈值阶段需要这个数（`only_rate` 就是"关掉这个检测能救回多少数据"），所以默认跑满。**阈值冻结、进入量产之后把 `short_circuit` 打开**，那时统计已经没用了，省下的是真金白银的机时。

### 为什么阶段 1 的三个检测不并行

它们都是在阶段 0 已经解出来的数组上做向量运算，耗时相对解码可以忽略。在 check 这一层开线程/进程，调度和序列化开销比省下来的计算还多。真正该并行的粒度是 **episode 级**——多进程各跑各的，进程内部串行。

这也是统一入口的主要收益：**解码从 4 次降到 1 次**。

---

## 4. 结果 JSON：哪个阶段、哪条规则、要删哪些文件

```jsonc
{
  "episode_id": "ep_000123",
  "n_frames": 300, "fps": 30.0,

  "failed": false,                    // 整条没有结论（解码失败/环境问题）
  "drop_episode": true,
  "dropped_by": ["black"],            // 是谁判的

  // 扁平的命中清单——不用钻进各检测就能回答"这条数据被谁以什么理由处置了"
  "triggered": [
    {"stage": 1, "check": "black", "rule": "BLACK_RUN_RATIO_EXCEEDED",
     "description": "任一视角最长连续黑帧段占比超过阈值",
     "disposition": "drop_episode",
     "views": ["head", "left_wrist"], "dims": [], "frame_ranges": [[0, 42]]}
  ],

  "errors":  {"state_extreme": {"stage": 2, "type": "UnsupportedDataError",
                                "message": "profile 缺少 'state_extreme' 段。..."}},
  "skipped": {"saa": "profile 标记本数据集不做 saa (reason: 该数据集无 action 通道)"},

  "frame_deletion": { /* 见下节 */ },
  "checks": { /* 每个检测的完整 report，含全部实测数字 */ }
}
```

### 报错分三档，各归各位

检测**从不静默降级**。出问题一律带着具体原因落到 JSON 里：

| 档 | 字段 | 含义 | 后果 |
|---|---|---|---|
| 整条失败 | `failed` + `failed_stage` + `failed_error_type` + `failed_error` | 解码失败，或某检测抛 `PipelineEnvironmentError`（scipy 缺失这类） | **没有结论**——既不是干净也不是坏，必须单独归档，不进任何比率的分母 |
| 单检测报错 | `errors[check]` | 该检测抛 `UnsupportedDataError`：profile 没配、维度对不上、有 NaN、格式不支持 | 只影响这一个检测，其余六个照常出结论 |
| 主动跳过 | `skipped[check]` | profile 里标了 `enabled: false`，或开了短路 | 单独记账 |

**错误信息原文照抄，不做二次概括。** `"数据格式 'rosbag' 未知; 支持的格式: ['hdf5','lerobot','parquet']"` 和 `"profile 缺少 q1"` 是完全不同的两件事，归并成"数据错误"就没法定位了。

**"零命中"和"根本没跑"在 JSON 里必须长得不一样。** 一个配错的 `enabled: false` 如果被当成通过，会让你以为这批数据很干净。

---

## 5. 删帧清单（`frame_deletion`）

### 为什么删一帧要牵动这么多文件

一条 episode 的第 i 帧不是一个东西，是**一组**东西：

```
videos/head/episode_000123.mp4          第 i 个视频帧
videos/left_wrist/episode_000123.mp4    第 i 个视频帧
data/chunk-000/episode_000123.parquet   第 i 行（state / action / timestamp / ...）
```

它们靠**下标对齐**，没有别的锚点。删第 i 帧必须所有地方同时删，少删一个，从第 i 帧往后所有模态整体错位一帧——**而且错位之后每个文件自己看都是自洽的**（长度合法、时间戳连续），没有任何检测能事后发现。一次失败的删帧，等于给数据集注入一个 SAA 级别的损坏。

所以清单**逐个列出**所有目标，而不是只写一句"删第 i 帧"：

```jsonc
"frame_deletion": {
  "actionable": true,                       // 现在能不能执行
  "blocked_reason": null,
  "frame_ranges": [[150, 153]],             // 左闭右开，单帧记为 [5, 6)，不加 margin
  "n_frames_deleted": 3, "n_frames_total": 300, "delete_ratio": 0.01,
  "n_segments": 1,                          // 删完会留下几处时间不连续
  "sources": {"sudden_change": [[150, 153]]},   // 这些帧是被谁要求删的
  "data_format": "lerobot",
  "targets": [
    {"kind": "video", "camera": "head",
     "path": "videos/chunk-000/head/episode_000123.mp4",
     "action": "删除这些帧后重新编码", "frame_ranges": [[150, 153]]},
    {"kind": "video", "camera": "left_wrist", "path": "...", "frame_ranges": [[150, 153]]},
    {"kind": "table", "format": "lerobot",
     "path": "data/chunk-000/episode_000123.parquet",
     "action": "删除对应行 LeRobot (parquet + 独立 mp4)。删行之外还要重算本条的 frame_index/timestamp、全局 index、meta/episodes.jsonl 的 length、以及 episodes_stats 里的统计量 —— 全局 index 是跨 episode 连续的，少一行会让后面所有 episode 的索引整体前移",
     "frame_ranges": [[150, 153]]},
    {"kind": "meta", "name": "episodes", "path": "meta/episodes.jsonl",
     "action": "连带更新 (长度/索引/统计量)"}
  ]
}
```

`sudden_change` 和 `state_extreme` 都命中时，帧区间**取并集**——同一帧被两个检测命中只删一次，但两个来源都记在 `sources` 里，事后能回答"这一帧是因为什么被删的"。

### 三种格式的连带后果不同

| 格式 | 删行本身 | 连带 |
|---|---|---|
| `lerobot` | 重写 parquet | 重算本条 `frame_index`/`timestamp`、**全局 `index`**、`meta/episodes.jsonl` 的 `length`、`episodes_stats` 的统计量。全局 index 跨 episode 连续，少一行会让**后面所有 episode** 的索引整体前移 |
| `parquet` | 重写整个文件 | `timestamp`/`frame_index` 这类自增列要重算 |
| `hdf5` | **h5py 不能原地删行** | 每个 dataset 读出来、切片、重写一个新文件；属性和软链接要一并搬过去 |

### 什么时候 `actionable: false`

| 情况 | 原因 |
|---|---|
| episode 已被判整条删除 | 整条都不要了，删几帧没意义。区间仍记录，便于回答"为什么被删" |
| corrupt 命中对齐类规则 | 下标映射不成立，帧号没有确定的所指。照着删只会把错位坐实 |
| 没提供 `EpisodeAssets` | 列不全要删的文件。宁可不执行也不能漏删一路 |

清单还会在**相机集合对不上**时直接报错——`video_paths` 少一路，就意味着那一路不会被删帧，从该帧起它和其他模态就错开了。

### `n_segments` 比"删了多少帧"更重要

删 5 帧但分散在 5 处，是给轨迹开了 5 个口子；删 5 帧集中在开头，只是把 episode 截短了一点。

`state_extreme` 的命中通常是**散点**（分布尾巴上的零星帧），`sudden_change` 的通常是**连续段**（可定位的事件）。所以 `DropBudget` 会统计段数分布：如果大量 episode 出现几十处不连续，说明 α 或 `jump_frac` 定得太紧，该回去调参数而不是硬删。

---

## 6. 三种终态

`EpisodeResult` 有三种互斥终态，调用方**必须分开处理**：

| 终态 | 判据 | 含义 |
|---|---|---|
| 失败 | `failed == True` | 检测没能完成。**既不是干净也不是坏** |
| 删除 | `drop_episode == True` | 至少一个检测判了丢弃，`dropped_by` 说明是哪些 |
| 通过 | 两者都 False | 保留；若 `frame_deletion.actionable` 则按工单删掉若干帧 |

失败的那类必须单独归档。`DropBudget` 把它们排除在分母之外——**把检测失败算成"通过"会让删除率显得比真实情况好看**，这是统计上最容易犯的错。

`run_episode()` 本身**不抛异常**，失败信息以结构化字段返回。理由：批量跑十万条时，让每条自己带着结论回来，比让调用方在循环里 try/except 更不容易出错——后者很容易被写成 `except: continue`，于是检测失败被静默当成通过。

---

## 7. 用法

```python
from check_pipeline.pipeline import run_episode, DropBudget
from check_pipeline.manifest import EpisodeAssets

result = run_episode(
    videos={"head": head_mp4, "left_wrist": lw_mp4},
    parquet_bytes=pq,
    episode_id="ep_000123",
    profile=profile,                        # dataset_profile.json 的内容
    assets=EpisodeAssets(
        data_format="lerobot",
        table_path="data/chunk-000/episode_000123.parquet",
        video_paths={"head": "videos/chunk-000/head/episode_000123.mp4",
                     "left_wrist": "videos/chunk-000/left_wrist/episode_000123.mp4"},
        meta_paths={"episodes": "meta/episodes.jsonl"}),
    params={"stillness": {"state_diff_thresh": 1e-3}})
```

批量 + episode 级并行：

```python
from concurrent.futures import ProcessPoolExecutor

budget = DropBudget()
with ProcessPoolExecutor(max_workers=16) as ex:
    for result in ex.map(_run_one, episode_paths):
        budget.add(result)
        json.dump(result.to_dict(), open(f"{result.episode_id}.json", "w"),
                  ensure_ascii=False, indent=2)
print(budget.format_report())
```

**非 parquet 数据集**（hdf5 / jsonl）由格式适配器先把信号取出来，用 profile 里声明的 `column` 名作键传进 `signals=`，流水线本身不关心物理格式。

`profile` 从哪来：见 `prompts/stage0_survey.md`（给数据侧模型的探查提示词）和 `dataset_profile.md`（schema 说明）。

---

## 8. 跑第一批的建议顺序

1. 先跑 500～1000 条，`print(budget.format_report())`
2. 看 **`failed_stages`**：应该是 0。不是 0 说明环境或格式有问题，**这批结果全部作废重跑**
3. 看 **`error_counts`**：哪个检测因为 profile 没配好而没跑起来。这些是"假通过"的来源，先修 profile
4. 看 **总删除率**是否可接受。超了先别调阈值，先看第 5 步
5. 看 **唯一归因**（`only_rate`）：只调唯一归因高的那个检测。调唯一归因为 0 的检测是白费力气——它删的 episode 别人也都会删
6. 看 **帧级删除的段数分布**：大量 episode 出现几十处不连续，说明 `alpha` 或 `jump_frac` 太紧
7. 调参数时用各检测报告里的诊断量，而不是拍脑袋：
   - `state_extreme` → `per_dim_alpha_needed_to_clear`（α 调到多大这个维度就不再命中）
   - `sudden_change` → 各通道的等效跳变幅度
   - `black`/`occlusion`/`stillness` → `measurements` 里的百分位分布

---

## 9. 还没做的

- **删帧的写出器**：本流水线只产出工单，真正执行删除的模块还没写。执行完必须逐项校验（视频帧数 == 表格行数、`frame_index` 连续、重跑 corrupt、stats 一致），不过就回滚
- **stage 0 扫描脚本**：`dataset_facts.json` 的确定性扫描（见 `dataset_profile.md`）
- **合并视频的帧偏移**：`video_granularity: per_chunk` 时 `probe_video` 需要支持按偏移切片
- 旧的 `new/__init__.py`、`new/occlusion.py`、`new/stillness_frame.py`、`new/metrics.py` 仍在原地，**没有被新流水线引用**，确认新流水线跑通后可以删除
