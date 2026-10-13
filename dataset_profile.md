# dataset_profile —— 接入新数据集前的探查阶段（stage 0）

> 本文只定方案：profile 的 schema、给 LLM 的提示词、以及校验规则。扫描脚本和代码接入还没写。

---

## 1. 为什么需要这一步

现在的流水线（`check_pipeline/`）是一个**检测算法库**，不是数据集 loader。它的输入是 `{相机名: mp4 字节}` + 单 episode 的 parquet 字节，磁盘上长什么样它完全不知道。

这带来一个后果：**流水线有一批硬需求，但它自己发现不了这些需求有没有被满足**。审计出来的清单：

| 流水线需要知道的 | 现在怎么来的 | 出错时的表现 |
|---|---|---|
| 有几路相机、分别叫什么 | 调用方传的字典 key | 漏传一路 → 那一路的问题**完全检不出**，报告里看不出少了东西 |
| 哪些 parquet 列是 state | 列名含 `state`/`observation.state`/`qpos` 的子串匹配 | 数据集叫 `joint_positions` → `UnsupportedDataError`，整个 stillness 跑不了 |
| 哪些列是导数通道（要排除） | 子串黑名单 | 黑名单里有 `vel_` 但没有 `vel`，`joint_vel` **不会被排除**，会混进静止判据 |
| 哪些列是 action | 列名含 `action` | 认不出 → corrupt 报错 |
| `state_diff_thresh` 取多少 | 必传，无默认 | 不传就报错（这是刻意的，见 `stillness.md` §5） |
| 一个 mp4 装几条 episode | **假设是 1 条** | 装多条 → 帧轴和 parquet 直接错位 |
| 数据集一共几条 episode | 调用方遍历 | 只读了 `chunk-000` → 剩下的数据**静默漏检** |

最后两行是你担心的那两件事，它们的性质不一样：**路径/chunk 漏读**是调用方的正确性问题，靠交叉校验能拦住；**视频合并**是流水线的能力缺口，靠提示词解决不了，得改代码（见 §7）。

---

## 2. 分工：哪些该测，哪些该问 LLM

需要确定的信息分两类，性质完全不同，混在一起做必然出问题。

### 事实类 → 脚本确定性地测，**不要让 LLM 看目录树**

有多少个 chunk、每个 chunk 几个文件、每路相机的视频分辨率、parquet 有哪些列、每列的 dtype 和 shape、episode 总数、每条 episode 多少帧……

这些必须靠 glob + 实际打开文件测出来。让 LLM"检索一下数据集"去回答这类问题，它会漏 `chunk-001`、会把没见过的相机名当成不存在、而且**不可复现**——同一个数据集问两次可能给出两个答案。质检流水线的输入不能是这种东西。

### 语义类 → LLM 来做，这是它真正有价值的地方

14 维 state 里哪几维是关节角、哪维是夹爪开合、哪几维是底盘位姿？`observation.state.effort` 算不算导数通道？哪个相机是腕部相机？state 里没有关节量时，action 的哪几维可以代替？

这些**没有闭式规则**。各数据集命名习惯不统一（`observation.state.arm.position` / `qpos` / `joint_positions` / `obs/robot0_joint_pos`），靠子串黑白名单永远追不完——上面那个 `vel_` 漏掉 `joint_vel` 的洞就是活证。这正是 LLM 该干的活。

### 人 → 确认一次，然后冻结

LLM 的输出**不直接进流水线**。它产出一份带证据的草稿，人过一遍，确认后冻结成 `dataset_profile.json` 提交进版本库。之后所有 episode 都读这份冻结的 profile。

**LLM 绝不进入 per-episode 循环。** 接入一个新数据集跑一次，不是每条数据跑一次。

```
    磁盘数据集
        │
        ▼
  [1] 确定性扫描脚本  ──►  dataset_facts.json   (纯事实, 可复现)
        │                        │
        │                        ▼
        │                  [2] LLM 语义标注  ──►  profile 草稿 (带证据)
        │                        │
        │                        ▼
        │                  [3] 人工确认 / 修正
        │                        │
        │                        ▼
        └──────────────►  [4] 交叉校验  ──►  dataset_profile.json (冻结)
                                 │
                                 ▼
                          check_pipeline 逐 episode 跑
```

---

## 3. 一个必须澄清的点：info.json 到底能不能读

`corrupt.py` 开头写着"不读 info.json"，这条原则**不能照搬到 stage 0**，否则会得出错误结论。原则的准确含义是：

> **不信任 info.json 对数据内容的声明**（帧数、分辨率、时长）——这些必须现场测，因为声明值和实际数据不符正是要检出的问题之一。

但有些信息**只存在于元数据里，别处根本推不出来**：

| 信息 | 来源 | 能不能现场测 |
|---|---|---|
| 视频/数据文件的路径模板 | `meta/info.json` 的 `video_path_template` / `data_path_template` | 不能，只能靠猜目录结构 |
| 每条 episode 在合并文件里的边界 | `meta/episodes/*.parquet` 的 `from_timestamp` / `dataset_from_index` | **完全不能**，mp4 里没有 episode 边界这个概念 |
| episode 总数 | 同上 | 不能（能数文件，但合并布局下文件数 ≠ episode 数）|

所以 stage 0 **必须读** info.json 和 episode 元数据，但读的是**布局信息**而不是**内容声明**。读完之后，profile 里同时记下"元数据声明的"和"实测的"两份数字，让 §6 的交叉校验去对——**对不上就报错**，这才是原则的正确用法。

具体到路径：**不要硬编码 `videos/chunk-000/<camera>/episode_XXXXXX.mp4`**。LeRobot v2.1 和 v3.0 的布局就不一样：

```
v2.1:  data/chunk-000/episode_000000.parquet
       videos/chunk-000/<camera>/episode_000000.mp4      一文件一 episode

v3.0:  data/chunk-000/file-000.parquet                   一文件多 episode
       videos/<camera>/chunk-000/file-000.mp4            一文件多 episode
```

注意 v3 把 `<camera>` 和 `chunk-xxx` 的层级顺序换了。硬编码任何一种都会在另一种上崩掉，而且是那种"路径拼错 → 文件不存在 → 当成缺失相机"的**静默降级**。正确做法是从 `info.json` 读 `*_path_template` 然后 `.format(...)`。

---

## 4. `dataset_facts.json` —— 扫描脚本的输出

这是喂给 LLM 的输入，**纯事实，无解释**。

```jsonc
{
  "dataset_root": "/path/to/dataset",
  "scanned_at": "2026-08-07T19:40:00+08:00",

  "layout": {
    "codebase_version": "v3.0",              // 来自 info.json, 没有就 "unknown"
    "data_path_template": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path_template": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "episodes_meta_path": "meta/episodes/chunk-000/file-000.parquet",

    // 实际 glob 到的, 不是模板推出来的
    "n_data_files": 12,
    "n_chunks_seen": [0, 1],                 // 实际存在的 chunk 编号, 用来抓"只读了 chunk-000"
    "video_files_per_camera": {"cam_high": 12, "cam_left": 12, "cam_wrist": 11},
    "n_episodes_declared": 480               // 来自 episode 元数据的行数
  },

  "video_granularity": {
    "verdict": "per_chunk",                  // per_episode | per_chunk | inconsistent
    "evidence": {
      "sample_file": "videos/cam_high/chunk-000/file-000.mp4",
      "decoded_frames": 45000,
      "episodes_mapped_to_this_file": 30,
      "sum_of_their_lengths": 45000          // 相等 => per_chunk 且边界自洽
    }
  },

  "cameras": [
    {"key": "cam_high",  "width": 640, "height": 480, "fps": 30.0,
     "pix_fmt": "yuv420p", "codec": "h264", "n_files": 12},
    {"key": "cam_wrist", "width": 320, "height": 240, "fps": 30.0,
     "pix_fmt": "yuv420p", "codec": "h264", "n_files": 11}
  ],

  // 关键: 不只给列名, 给采样统计值。光看名字分不出关节角/夹爪/底盘位姿。
  "columns": [
    {
      "name": "observation.state",
      "dtype": "float32", "shape": [14], "n_rows_sampled": 20000,
      "per_dim": [
        {"i": 0, "min": -3.11, "max": 3.09,  "mean": 0.02, "std": 1.21, "n_unique": 19875},
        {"i": 1, "min": -1.57, "max": 1.55,  "mean": -0.4, "std": 0.88, "n_unique": 19901},
        // ... 12–13 维往往是夹爪, 看 n_unique 和值域就能认出来
        {"i": 13, "min": 0.0,  "max": 1.0,   "mean": 0.61, "std": 0.47, "n_unique": 2}
      ],
      "sample_rows": [[0.01, -0.42, "..."], [0.01, -0.41, "..."], [0.02, -0.39, "..."]]
    },
    {"name": "action", "dtype": "float32", "shape": [14], "per_dim": ["..."], "sample_rows": ["..."]},
    {"name": "timestamp", "dtype": "float64", "shape": [], "per_dim": ["..."]},
    {"name": "episode_index", "dtype": "int64", "shape": []}
  ],

  // 抽样若干条 episode 实测的对齐情况
  "alignment_samples": [
    {"episode_index": 0,   "parquet_rows": 1500, "video_frames": {"cam_high": 1500, "cam_wrist": 1500}},
    {"episode_index": 239, "parquet_rows": 1483, "video_frames": {"cam_high": 1483, "cam_wrist": 1483}}
  ],

  // 用来定 state_diff_thresh 的分布 (stillness 的必传参数)
  "state_diff_distribution": {
    "column": "observation.state",
    "n_episodes_sampled": 30,
    "percentiles": {"p1": 0.0, "p5": 0.00012, "p25": 0.0031, "p50": 0.019, "p75": 0.061, "p99": 0.42},
    "histogram_note": "双峰 / 单峰 / 无法判断"
  },

  "frame_diff_distribution": {
    "per_camera": {"cam_high": {"p1": 0.0, "p50": 6.4, "p99": 21.3}}
  },

  "scan_errors": []      // 打不开的文件、glob 到但元数据里没有的文件等
}
```

**为什么一定要给 `per_dim` 统计和 `sample_rows`**：这是整个方案里最关键的一处。只给列名的话，LLM 只能靠命名猜，那和现在的子串匹配没有本质区别。给了值域和 `n_unique` 之后：

- 值域 ±3.14、`n_unique` 上万、std 约 1 → 关节角（弧度）
- 值域 0–1、`n_unique` = 2 → 夹爪开合（二值）
- 值域 0–255、`n_unique` 上百 → 夹爪开度（归一化到字节）
- 值域 ±0.5、单位像米 → 底盘/末端位置
- 均值接近 0、std 大、相邻行跳变剧烈 → 速度类（导数通道）

这些判断有据可依，而且 §6 的校验能验证。

---

## 5. 给 LLM 的提示词

输入是上面的 `dataset_facts.json`，输出是 profile 草稿。

````text
你在为一个机器人数据集质检流水线做接入前的语义标注。

输入是一份 dataset_facts.json，里面是脚本从数据集里**实测**出来的事实：文件布局、
相机参数、parquet 每一列的 dtype/shape/每维的 min/max/mean/std/n_unique，以及采样行。

你的任务是把这些事实翻译成流水线能用的语义标注。你**只做语义判断**，不要重新计算
或修改任何事实数值。

======================================================================
硬性规则
======================================================================

1. 【不许发明】只能引用 dataset_facts.json 里实际出现的列名、相机名、维度下标。
   凡是输入里没有的，一律不许出现在输出里。

2. 【不许猜】任何一项判断，如果证据不足以支持，必须填 "uncertain" 并在
   uncertainties 里说明缺什么信息。**填 "uncertain" 是正确行为，猜一个看起来
   合理的答案是错误行为。**
   这条流水线的设计原则是"有问题就报错，不要静默地给出可能错误的答案"。一个猜出来
   的标注不会报错，它只会让后续所有 episode 的检测结果悄悄失真。

3. 【必须给证据】每一项判断都要在 evidence 字段里引用**具体的**事实：
   哪个列名、哪一维、哪个数值区间。
   ✗ "看起来像关节角"
   ✓ "dim 0-5 值域在 ±3.14 内, n_unique > 19000, std ≈ 1.2, 符合弧度制关节角"

4. 【区分测量与指令】state 是**测量值**（机器人实际在哪），action 是**指令值**
   （控制器让它去哪）。两者语义不同，不能混用而不声明。

5. 【只输出 JSON】不要有任何解释性前后文。

======================================================================
判断依据
======================================================================

■ 维度语义 —— 优先看数值特征，其次才看命名

  关节角(弧度)    值域约 ±π, n_unique 大, std 0.5~1.5
  关节角(角度)    值域约 ±180
  夹爪(二值)      n_unique = 2, 值域 {0,1} 或 {0,255}
  夹爪(连续)      n_unique 中等, 值域 [0,1] 或 [0,255] 且单调段明显
  位置(米)        值域约 ±2, 单位量级远小于关节角
  四元数          连续 4 维, 值域 ±1, 且该 4 维平方和恒为 1
  速度/力矩       均值接近 0, 相邻采样行跳变剧烈 (看 sample_rows)

  命名只作为**旁证**。命名和数值特征冲突时，以数值特征为准，并在 uncertainties
  里记下这个冲突 —— 这种冲突本身往往就是数据问题。

■ 静止检测(stillness)要用哪些维度

  目的是判断"机器人有没有动"，所以要的是**位置类**维度：关节角、末端位置、
  夹爪开合。要排除：
  - 导数通道(速度/加速度/力矩/力/jerk)。差分一个速度信号得到的是加加速度，
    和"有没有动"不是一回事。
  - IMU、时间戳、episode_index、task_index 这类非本体状态量。

  夹爪维度**要保留**。夹爪在动就说明任务在推进，那些帧不是冗余帧。

  如果 state 列里找不到任何关节/位置类维度，就去 action 列里找同类维度作为替代，
  并且**必须**把 state_source 标成 "action_fallback"、把 caveat 写清楚：
  用 action 判静止，判的是"控制器没发运动指令"，不是"机器人没动"。机器人卡在
  障碍物上时 action 可能仍在变化，这种情况会被判成"在动"。

■ 相机

  按命名和分辨率推断每一路的安装位置(head/wrist/side/top/...)。
  推不出来就填 "unknown" —— 这个字段只用于报告可读性，不参与判定，填 unknown
  没有任何代价。

======================================================================
输出格式
======================================================================

{
  "dataset_id": "<从 dataset_root 推断的简短标识>",
  "confidence": "high | medium | low",

  "cameras": [
    {"key": "<必须是 facts 里的相机名>",
     "mount": "head|wrist|side|top|unknown",
     "evidence": "..."}
  ],

  "state": {
    "source": "state | action_fallback",
    "column": "<必须是 facts 里的列名>",
    "dims_used": [0,1,2,3,4,5,6,13],
    "dims_excluded": [
      {"dims": [7,8,9,10,11,12], "reason": "四元数姿态, 与关节角冗余", "evidence": "..."}
    ],
    "dim_semantics": [
      {"dims": [0,1,2,3,4,5,6], "meaning": "arm_joint_angle", "unit": "rad", "evidence": "..."},
      {"dims": [13], "meaning": "gripper", "unit": "binary", "evidence": "..."}
    ],
    "caveat": "<source 为 action_fallback 时必填, 否则为 null>"
  },

  "action": {"column": "...", "dims": 14, "evidence": "..."},

  "suggested_params": {
    "stillness": {
      "state_diff_thresh": 0.0035,
      "basis": "state_diff_distribution 的 p5=0.00012 与 p50=0.019 之间存在明显间隔,
                取静止峰上沿",
      "confidence": "high | medium | low"
    }
  },

  "uncertainties": [
    {"field": "state.dims_semantics[2]",
     "issue": "dim 10-12 值域 ±0.3, 命名无线索, 可能是底盘位置也可能是力传感器",
     "need": "该机型的 state 维度定义文档, 或一段已知机器人静止时的采样"}
  ],

  "blocking_issues": [
    "<会让流水线跑不了或结果不可信的问题。没有就空数组。>"
  ]
}
````

**关于 `state_diff_thresh`**：让 LLM 从分布里读一个建议值是可以的（它就是在双峰之间找间隔），但这个值**必须经人确认**才能进冻结的 profile。`stillness.py` 把它设成必传参数就是为了防止有人随手填一个数——那个防线不能被一个自动生成的 profile 绕过去。

---

## 6. 交叉校验：profile 冻结前必须过的检查

这一步是整个方案里真正拦住"路径嵌套 / chunk 漏读"的地方。**LLM 不参与，纯脚本，任何一条不过就拒绝冻结。**

### 6.1 覆盖完整性（防漏读）

```
n_episodes_declared           元数据里的 episode 条数
n_episodes_resolvable         按 path_template 解析后, 文件确实存在的条数
必须相等 —— 不等说明有 episode 的文件找不到

n_chunks_seen                 实际 glob 到的 chunk 编号集合
必须是连续的 [0..k]           出现 [0, 2] 说明 chunk-001 漏了或缺了

video_files_per_camera        各相机的视频文件数
所有相机必须相等              上面 facts 示例里 cam_wrist 是 11 而别人是 12,
                              这就是一条必须拦下的错误
```

最后这条特别值得强调：**少一个视频文件，现在的流水线是完全无感的**。调用方传进来的字典里就没有那一路，`check_black` 只会检查它拿到的相机。报告干干净净，而那一路的所有问题一个都没检出。

### 6.2 布局自洽（防路径拼错）

按 `path_template` 解析出的文件集合，和 `glob` 出来的文件集合，必须**完全相同**。

- 模板解析出来但磁盘上没有 → 模板错了或数据缺了
- 磁盘上有但模板解析不到 → **有数据没进流水线**，这是最危险的一种，因为不报错

### 6.3 对齐自洽（防合并视频切错）

对抽样的每条 episode：

```
parquet 行数 == 该 episode 在各路视频里的帧数     (逐相机)
各相机之间帧数一致
sum(所有 episode 的长度) == 合并视频文件的总帧数   (per_chunk 布局下)
```

最后一条是判定 `video_granularity` 的依据，也是**切片正确性的总账**。差一帧都说明边界算错了——见 §7 的警告。

### 6.4 列可解析

profile 里写的每个列名，必须能在实际 parquet 里找到，且 shape 与 facts 一致。`dims_used` 的下标必须都在 `[0, shape)` 范围内。

---

## 7. 视频合并对现有代码的影响（这部分要改代码）

确认了数据里存在"多条 episode 合并进一个 mp4"，这是**能力缺口**，不是配置问题。现状：

```python
def probe_video(video_bytes, cam="", collect_gray=True) -> VideoProbe
```

`probe_video` 从头解到尾，把整个文件当成一条 episode，没有任何切片入口。合并布局下直接用它，帧轴和 parquet 全错位。

### 7.1 需要的改动

**接口加帧区间**，并且**输入从 `bytes` 换成路径或 file-like**：

```python
probe_video(video_src, cam="", collect_gray=True,
            from_ts=None, to_ts=None)     # 秒, 来自 episode 元数据
```

换掉 `bytes` 是因为：一个 chunk 文件可能装 30 条 episode，每条都把整个文件的字节读一遍，I/O 会炸。给 PyAV 一个可 seek 的对象，让它只读需要的部分。

**实现要点**：seek 到 `from_ts` 之前最近的关键帧 → 往后解码 → 丢掉 `pts < from_ts` 的帧 → 解到 `pts >= to_ts` 停。代价约等于"episode 长度 + 一个 GOP"，30 条 episode 跑完一个 chunk 文件总代价约 1–2 倍文件长度，可以接受。

**帧号必须重新锚定到 episode 内部**：现在 `probe.py` 用 PTS 算帧号（这是对的，防止坏帧导致的漂移），切片后要改成相对 `from_ts` 的偏移，否则第 0 帧的编号会是它在 chunk 里的全局位置。

**`frame_diff[0]` 保持 NaN**：切片后 episode 的第一帧在物理文件里是有前一帧的，但那是**上一条 episode 的最后一帧**，跨 episode 的差分没有意义。必须按 episode 边界断开。

### 7.2 一个必须提前想到的风险

corrupt 的 `ACTION_VIDEO_LENGTH_MISMATCH` 比对 parquet 行数和视频帧数，而 corrupt 现在的规则是**任何一条 finding 都删整条 episode**。

如果时间戳到帧号的换算有系统性的 off-by-one（浮点 `from_ts` × fps 的取整边界，这非常容易发生），那就是**每一条 episode 都 mismatch，整个数据集被删光**。

所以切片实现完之后，第一件事是拿一批 episode 验证 §6.3 的总账：`sum(各 episode 帧数) == 文件总帧数`。这个等式成立才说明边界算对了。在它成立之前，不要跑全量。

### 7.3 顺带变化：corrupt 的影响范围变了

合并布局下，一个 mp4 解不开意味着**这个 chunk 里所有 episode 都完了**，不再是"一条 episode 完了"。`VIDEO_UNDECODABLE` 的爆炸半径从 1 条变成几十条。报告层面需要能表达"这批 episode 是被同一个文件的损坏连坐的"，否则会看到几十条互相独立的删除记录，看不出它们是同一个根因。

---

## 8. 落地顺序

1. 写扫描脚本，产出 `dataset_facts.json`——先只跑一个数据集，把 §6 的校验跑通
2. 用 §5 的提示词跑 LLM，人工确认 profile
3. 改 `probe_video` 支持切片，用 §7.2 的总账验证
4. profile 接进 `pipeline.py`：相机列表、state 列与维度、`state_diff_thresh` 都从 profile 读，不再靠子串猜
5. 第 4 步做完之后，`stillness.py` 里的 `STATE_COLUMN_HINTS` / `DERIVATIVE_HINTS` 子串启发式就可以退成"profile 缺失时的兜底"，或者干脆删掉改成必须有 profile

第 5 步是这套方案真正的收益：现在那套子串黑白名单（`vel_` 匹配不到 `joint_vel` 这种洞）永远追不完，因为它试图用固定规则去覆盖开放的命名空间。profile 把这个问题**一次性地、显式地**解决在接入阶段。
