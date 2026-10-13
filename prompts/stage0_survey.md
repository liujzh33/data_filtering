# 阶段 0 数据探查 —— 给数据侧模型的提示词

把下面 `====` 之间的全部内容发给数据侧的模型（GLM-5.2），一个数据集发一次。它产出的
`dataset_profile.json` 是七个检测的**唯一配置来源**。

为什么必须先做这一步：后面七个检测里有五个需要知道"用哪些维度"。旧版靠列名子串猜，
实测栽了三次——`"effector"` 是 `"end_effector"` 的子串，把末端位姿整体当手爪豁免；
`d0..d13` 这类命名里没有语义，豁免完全失效；robottwin2/franka 套用了别的机型的维度
模板，把两个真关节当手爪、两个真手爪当关节。**这些错误全都不会报错，只会静默地检错东西。**

---

====

# 任务

你要探查一个机器人操作数据集，产出一份 `dataset_profile.json`。这份文件会驱动一套
七阶段的数据质检流水线，**填错的后果是静默检错对象**，而不是报错，所以准确性远比覆盖率重要。

## 三条硬性规则

**规则一：不确定就写 `"unknown"` 并在 `notes` 里说明，绝不猜。**
流水线对 `unknown` 会直接报错停下，让人来看一眼；而对一个猜错的值会照单全收，
一路算下去给出看似正常的结论。停下来的代价是十分钟，猜错的代价是整批数据白洗。

**规则二：每一个语义判断都要给证据。**
说某一维是手爪，要说清楚是怎么判断的：是 URDF/文档写明的？是列名？还是从数值范围
推断的（比如取值恒在 `[0, 1]` 或 `[0, 0.08]` 这种窄正区间，而关节是弧度量级的正负值）？
把依据写进对应的 `evidence` 字段。

**规则三：实际打开文件确认，不要只看目录名和文档。**
路径模板要用 `meta/info.json` 里的 `*_path_template` 解析，不要硬编码 `chunk-000`——
超过一个 chunk 的数据集会被漏读。列的维度要实际读一个 parquet/hdf5 确认 shape，
不要照抄文档。

---

# 你要产出的 JSON

```jsonc
{
  "dataset": "robochallenge2",              // 数据集标识
  "embodiment": "ALOHA",                    // 机型标识；一个数据集多机型就产出多份
  "percentile_scope": "robochallenge2/ALOHA",  // 上面两者拼接，用于追溯分位数来源

  // ---------- 1. 格式与文件布局 ----------
  "format": {
    "data_format": "lerobot",               // lerobot | parquet | hdf5，三选一，不能 unknown
    "table_path_template": "data/chunk-{chunk:03d}/episode_{index:06d}.parquet",
    "video_path_template": "videos/chunk-{chunk:03d}/{camera}/episode_{index:06d}.mp4",
    "meta_files": ["meta/info.json", "meta/episodes.jsonl"],
    "n_episodes": 1200,
    "n_chunks": 2,                          // 实际 glob 出来的数量，不是假设的 1
    "video_granularity": "per_episode",     // per_episode | per_chunk（多条 episode 合成一个 mp4）
    "evidence": "路径模板取自 meta/info.json 的 video_path_template；glob 到 chunk-000/001"
  },

  // ---------- 2. 相机 ----------
  "cameras": [
    {"name": "head",       "width": 640, "height": 480, "fps": 30, "role": "third_person"},
    {"name": "left_wrist", "width": 640, "height": 480, "fps": 30, "role": "wrist"}
  ],

  // ---------- 3. 列布局（事实，不含语义判断）----------
  "columns": [
    {"name": "observation.state", "shape": [14], "dtype": "float32"},
    {"name": "action",            "shape": [14], "dtype": "float32"}
  ],

  // ---------- 4. 每一维是什么（核心，最容易出错）----------
  "dim_semantics": {
    "action": [
      {"dim": 0,  "name": "left_joint_0",  "kind": "joint",    "unit": "rad",
       "observed_min": -0.83, "observed_max": 0.37},
      {"dim": 6,  "name": "left_gripper",  "kind": "gripper",  "unit": "m",
       "observed_min": 0.0,   "observed_max": 0.105},
      {"dim": 13, "name": "right_gripper", "kind": "gripper",  "unit": "m",
       "observed_min": 0.0,   "observed_max": 0.105}
    ],
    "observation.state": [ /* 同上结构 */ ],
    "evidence": "维度名取自 info.json 的 features.action.names；手爪由单位(m)与 [0,0.105] 的窄正区间确认"
  },
  // kind 只能取: joint | gripper | eef_position | eef_rotation | velocity | force | base | other | unknown

  // ---------- 5. 静止帧检测：用哪些维度、阈值多少 ----------
  "stillness": {
    "state_dims": [0,1,2,3,4,5,7,8,9,10,11,12],
    "state_diff_thresh": 0.001,
    "evidence": "取 state 的关节维度；阈值来自实测：正常运动段的相邻帧差中位数是 0.012 rad，静止段是 2e-5"
  },

  // ---------- 6. SAA（状态-动作趋势对齐）----------
  "saa": {
    "enabled": true,
    "reason": "",                           // enabled=false 时必填，说明为什么不做
    "state_column": "observation.state",
    "action_column": "action",
    "action_space": "absolute",             // absolute | delta，绝不能猜，见下方说明
    "da_threshold": 0.7,
    "shared_dims": [
      {"name": "left_joint_0", "state_dim": 0, "action_dim": 0}
    ],
    "evidence": "state[t+1] 与 action[t] 的数值量级一致且直接可比 -> absolute；关节维度在两侧下标相同"
  },

  // ---------- 7. 突变检测 ----------
  "sudden_change": {
    "enabled": true,
    "dims": [
      {"name": "left_joint_0", "column": "action", "dim": 0, "range": 1.20}
    ],
    "evidence": "range = 全数据集该维度的 p99 - p1（不是单条 episode 的）"
  },

  // ---------- 8. 极值过滤 ----------
  "state_extreme": {
    "enabled": true,
    "source": "action",                     // state | action，这份分位数是从哪种信号算的
    "percentile_scope": "robochallenge2/ALOHA",
    "dims": [
      {"name": "left_joint_0", "column": "action", "dim": 0, "q1": -0.834, "q99": 0.369}
    ],
    "evidence": "全数据集该机型所有 episode 逐帧池化后取 1/99 分位"
  },

  "notes": ["把所有拿不准的地方列在这里"]
}
```

---

# 各字段怎么确定，以及各自的坑

## 4. `dim_semantics` —— 最重要的一节

流水线里五个检测都靠它选维度。判断 `kind` 的依据，按可靠性从高到低：

1. `meta/info.json` 的 `features.<column>.names`，或数据集自带的 URDF / 文档
2. **数值特征**（当名字不可信时，这是最可靠的交叉验证）：
   - 手爪：单位是米或归一化值，取值在 `[0, 0.1]` 或 `[0, 1]` 这类**窄正区间**，
     且分布**双峰**（大量样本聚在两端，中间很少）——因为手爪基本只有开和合两个状态
   - 关节：弧度量级，取值跨正负，分布连续
   - 速度/力矩：均值接近 0 且左右对称，与位置维度的量级差一个数量级以上
3. 列名子串——**只能当辅助**，且必须用上面两条交叉验证

> **真实教训**：某份配置把 franka 的 16 维套用了另一个机型的 6+6+2 模板，结果 dim 12/13
> （取值 `[1.84, 3.58]` 弧度，明显是关节）被标成手爪，而 dim 14/15（取值 `[0, 1]`，
> 明显是手爪）被当成关节。**如果当时看一眼数值范围就不会错。** 请务必对每一维填上
> `observed_min` / `observed_max`，它们本身就是自检。

自由度数量也是一个校验：franka 是 7 自由度，UR5 / ARX5 是 6，双臂就翻倍。
如果你数出来的关节数和机型的自由度对不上，说明维度切分错了。

## 5. `stillness.state_dims`

只放**关节**维度。手爪要不要放取决于任务——如果任务里有"保持抓握不动"的合理片段，
放进去会导致误判；默认不放。

`state_diff_thresh` 要实测，不要照抄 `1e-3`：随便抽 5 条 episode，
分别算"明显在运动的一段"和"明显静止的一段"的相邻帧差中位数，阈值取两者之间。
把这两个实测数字写进 `evidence`。

## 6. `saa.action_space` —— 猜错的代价最大

绝对位置（absolute）指 `action[t]` 直接就是目标关节角；增量（delta）指
`action[t]` 是相对当前位置的变化量，要先积分才能和 state 比。

**判别方法**：取任意一条 episode，比较 `action[t]` 和 `state[t]` 的数值范围。
量级相当（都是 ±2 弧度这种）→ absolute；action 的量级小一两个数量级（±0.02）
且均值接近 0 → delta。

**实测证据**：delta 数据没积分就直接做 SAA，DA 会掉到 0.496——几乎正好是随机水平，
于是整个数据集会被判成"状态与动作不对齐"而全部删掉。

## 6. `saa.shared_dims`

只放**同时满足**三个条件的维度对：

1. state 和 action 里**都有**
2. 是**关节**（或末端位姿），不是速度、力矩、传感器读数
3. 两侧含义一致（`state_dim` 和 `action_dim` 指的是同一个物理关节）

**手爪不要放。** 手爪是开关式的双峰信号，方向一致性对它没有意义。

如果找不到任何满足条件的维度对（比如数据集根本没有 action 通道），
填 `enabled: false` 并在 `reason` 里说明。**不要硬凑。**

## 7. `sudden_change.dims[].range`

`range` 是该维度**在整个数据集上**的 p99 − p1，不是单条 episode 的。

原因：阈值是"跳变幅度超过 range 的百分之几"。如果 range 按每条 episode 各算各的，
同一个关节在动作幅度小的 episode 里阈值就小、大的就大，同样大小的异常在不同 episode
里结论相反。必须用数据集级的固定值。

范围比 SAA 宽：**不要求 state/action 同时存在**，只要是有物理意义的连续信号就可以进，
包括末端位姿。手爪可以放（它的突变是真实的动作），速度/加速度这类本身就是导数的通道
不要放。

## 8. `state_extreme`

- **只放关节维度。** 手爪因为双峰分布不参与——不是"运行时豁免"，是**根本不进这个列表**。
- `q1` / `q99` 按**每个数据集的每个机型**分别算：把该机型所有 episode 的所有帧池化，
  逐维取 1 / 99 分位。
- `source` 必须写清是从 `state` 还是 `action` 算的，**并且要和 `dims[].column` 一致**。
  用 action 的分位数去卡 state 的值是一个真实发生过的错误：两者隔着跟踪误差和控制滞后，
  分布不重合，而输出里完全看不出来。
- 优先用 action（指令信号比反馈信号干净）。如果数据集没有 action 通道，用 state，
  把 `source` 如实写成 `"state"`。

## 1. `format` 的两个坑

**别硬编码 `chunk-000`。** 用 `info.json` 的 `*_path_template` 解析，然后 glob 验证
实际有几个 chunk。只读 chunk-000 会漏掉后面全部数据。

**确认 `video_granularity`。** 如果一个 mp4 里装了多条 episode（`per_chunk`），
那么"第 i 帧"需要加上该条 episode 在文件里的起始偏移才能定位。这件事必须提前说明，
否则所有帧号都会错位。

---

# 交付前的自检清单

逐条核对，把结果写在输出末尾：

- [ ] 每个 `kind: "gripper"` 的维度，`observed_min`/`observed_max` 是否确实是窄正区间？
- [ ] 关节维度的数量是否和该机型的自由度对得上（单臂 6 或 7，双臂翻倍）？
- [ ] `saa.shared_dims` 里有没有混进手爪？
- [ ] `state_extreme.dims` 里有没有混进手爪、末端位姿、速度？
- [ ] `state_extreme.source` 和 `dims[].column` 指的是同一种信号吗？
- [ ] `sudden_change.dims[].range` 是数据集级的 p99−p1，不是单条 episode 的？
- [ ] `action_space` 是实际比较过数值范围得出的，不是从字段名猜的？
- [ ] 路径模板是从 `info.json` 解析的，且 glob 验证过 chunk 数量？
- [ ] 所有 `evidence` 字段都填了具体依据，不是"根据命名推断"这种空话？
- [ ] 所有拿不准的地方都写成 `unknown` 并列进 `notes` 了？

====

---

## 拿到结果之后

profile 不要直接投产，先过两道校验：

1. **脚本校验**：`dims[].column` 在实际文件里存在、`dims[].dim` 不越界、
   `q99 > q1`、相机数量与 `video_path_template` glob 出来的一致。
   （`check_pipeline` 各模块自己会做这些校验，跑一条 episode 就能暴露）
2. **人看一遍 `dim_semantics`**，重点是手爪那几维的数值范围。这是最容易错、
   且错了最不容易发现的地方。

确认之后冻结，profile 变更要留记录——同一批数据用了两个版本的 profile，
结果就没有可比性了。
