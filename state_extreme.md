# state_extreme — 极值过滤（现状分析与重构方案）

对应旧实现: `new/state_extreme_frame.py`（仍在旧架构: `.base` 的 `Issue`/`severity`/`exclude_mask`，未进 `check_pipeline/`）。

参考做法（Qwen）: 按维度计算 q1 和 q99 百分位（**按实体类型分别计算**），超出 `[q1 − α(q99 − q1), q99 + α(q99 − q1)]` 的帧被剔除，防止分位数归一化被扭曲。手爪维度因双峰分布不参与。

目的这句话很关键：**是为了防止归一化被离群值带偏，不是为了修剪正常分布的尾巴。** 下面的实测表明现在的实现和这个目的偏得很远。

---

## 一、实测: 现有实现的问题

### 1.1 `alpha` 是死参数，table.txt(52) 要求的扫描现在做不了

`embodiment_percentiles_ext.json` 里存的是**已经算好的 `lo`/`hi`**，而代码走的正是这两个字段：

```python
if embodiment_percentiles and pct_name in name_to_pct:
    lo = pct["lo"]; hi = pct["hi"]      # t["alpha"] 在这条路径上完全没用上
```

而文件里的 `lo`/`hi` 是按 **α=3 烘死**的（实测 ALOHA 第 0 维）：

| α | `q1 − α·(q99−q1)` | 文件里的 `lo` |
|---|---|---|
| 0 | −0.6594 | −5.0184 |
| 1 | −2.1124 | −5.0184 |
| 3 | **−5.0184** | −5.0184 |

`t["alpha"]` 只在"没有 embodiment_percentiles"的退化分支里才起作用。所以：

- 文档写 `alpha: 0`、代码写 `alpha: 3.0`、文件烘的是 `3.0` —— 三处不一致，而且都不影响主路径。
- **table.txt(52) 要求测 α = 0/0.5/1/2/3，现在改这个参数没有任何效果**，必须重算整个 json 才能换一个 α。

修法很简单: 文件里存 `q1`/`q99`，`lo`/`hi` 在运行时按 α 算。α 就变成一个活的、可扫的参数。

### 1.2 "α=0 只会删 2% 的帧"——实测不成立

table.txt(53) 的推论是"α=0 也只删 2%，所以根本不需要到 episode 级别"。结论（只做帧级）我同意，但**理由不成立**，而且实际删除量比 2% 大一个数量级。

两个原因叠加：

**其一，代码取的是各维度的并集**（`total_flagged |= dim_flag`）。单维 2%，多维并集远不止 2%：

| 维度数 | 各维独立 | 中等相关 ρ=0.5 | 强相关 ρ=0.9 |
|---|---|---|---|
| 1 | 2.0% | 2.0% | 2.0% |
| 7 | 13.2% | 10.3% | 5.0% |
| 14 | **24.6%** | 16.6% | 6.4% |

**其二，q1/q99 是数据集级统计量，单条 episode 的命中率方差极大。** 模拟 200 条任务不同、工作区中心不同的正常 episode（7 维，α=0）：

| | 命中率 |
|---|---|
| 全数据集平均 | 13.4% |
| 中位数 episode | 2.2% |
| 第 90 百分位 | **54.5%** |
| 第 99 百分位 | 94.8% |

**10.5% 的 episode 会被删掉一半以上的帧，而它们的数据完全正常**——只是在工作区边缘作业。中位数看着像 2%，掩盖了尾部的灾难。

### 1.3 α 扫描: 小 α 并不能提高检出，只是在切正常分布的尾巴

同一份模拟数据，另外注入真实离群值（0.1% 的帧，幅度 20 倍量程，模拟传感器毛刺/单位错误）：

| α | 总删帧率 | 命中>20% 的 episode | 命中>50% 的 episode | 真离群值检出率 |
|---|---|---|---|---|
| 0 | **13.43%** | 21.0% | 10.5% | 100% |
| 0.5 | 0.08% | 0.0% | 0.0% | 100% |
| 1 | 0.08% | 0.0% | 0.0% | 100% |
| 2 | 0.08% | 0.0% | 0.0% | 100% |
| 3 | 0.08% | 0.0% | 0.0% | 100% |

α 从 0 抬到 0.5，删帧率掉了 170 倍，**而真离群值的检出率一点没变（始终 100%）**。

这正是回到那句设计意图: 要防的是"归一化被离群值带偏"。真正的离群值是数量级上的错误（毛刺、单位搞错、坐标系错误），离 band 远得很；α=0 抓的那 13% 全是正常分布的尾巴。

补一个直觉校准: 正态分布下 q1/q99 是 ±2.33σ，跨度 4.65σ。α=1 的 band 就是 ±6.98σ，α=3 是 ±16.3σ。**α≥0.5 之后，band 已经宽到只有真正的错误值才够得着。**

建议默认 **α = 1.0**，第一批跑完看 `per_dim` 的实际命中分布再调。

### 1.4 退化分支在数学上是无意义的

没有 `embodiment_percentiles` 时，代码用**这条 episode 自己**的 q1/q99 建 band：

```python
q1, q99 = np.percentile(x, [1, 99])       # 用自己算
lo = q1 - alpha*(q99-q1); hi = q99 + alpha*(q99-q1)
dim_flag = (x < lo) | (x > hi)            # 再拿去卡自己
```

实测两条完全正常的 episode：

```
alpha=0.0: 命中 2.00%      alpha=3.0: 命中 0.00%
alpha=0.0: 命中 2.00%      alpha=3.0: 命中 0.00%
```

结果**由 α 决定，不由数据决定**——α=0 时按定义恒为 2%，α=3 时恒为 0。一条数据完美的 episode 和一条充满毛刺的 episode 得到一模一样的结论。这个分支产出的是看起来很确信的垃圾。

### 1.5 NaN 让整个维度静默失效

```
含 NaN 时 np.percentile -> q1=nan q99=nan
(x < nan) | (x > nan)  全 False  ->  同一条信号里 100.0 这个真极值也没被抓到
```

NaN 传染到 band 之后，所有比较都是 False，该维度**一个帧都不会被标记**，而且输出里看不出发生过这件事。缺失值恰恰常常和数据异常同时出现，于是最该查的维度被静默放行。

### 1.6 手爪豁免靠子串匹配，而且会误伤末端位姿

```python
if any(k in low for k in ("gripper", "finger", "effector", "jaw")):
```

两个问题：

- **`"effector"` 是 `"end_effector"` 的子串。** 任何把末端位姿命名为 `end_effector_pose_*` 的数据集，末端位姿维度会被整体当成手爪豁免掉，静默不检测。
- OXE / GM-100 那种"一个 `observation.state` 列装 14 维"的布局展开后是 `d0..d13`，**名字里没有任何语义**，手爪豁免完全失效——这和 SAA 旧版踩的是同一个坑。

顺带: `embodiment_percentiles_ext.json` 里其实已经有 `is_gripper` 字段，代码却没用，而是自己去猜名字。

### 1.7 维度名转换脆弱，匹配不上就静默跳过

```python
pct_name = name
if ".state." in pct_name: pct_name = pct_name.split(".state.", 1)[1]
pct_name = pct_name.replace(".d", "_").replace(".", "_")
```

`.replace(".d", "_")` 是无差别替换，维度名里任何位置出现 `.d` 都会被改（`left.dof3` → `left_of3`）。匹配不上时：

```python
elif embodiment_percentiles is not None:
    continue        # 静默跳过, 不报错、不计数
```

一个维度没被检查，和一个维度检查通过了，在输出里长得一模一样。

### 1.8 用 action 的分位数去卡 state 的值

模块叫 `state_extreme`，提取的是 `sig.get("state")`；而百分位文件由 `calc_action_percentiles.py` 产出、README 里叫 `action_percentiles.json`。**band 来自 action 的分布，却拿去卡 state 的值。**

state 和 action 之间隔着跟踪误差和控制滞后，分布不重合。这个错配会系统性地偏移命中率，而且从输出里完全看不出来。

现在的 `embodiment_percentiles_ext.json` 里也**没有任何字段记录这份分位数是从 state 还是 action 算出来的**。

### 1.9 其他

| 问题 | 说明 |
|---|---|
| 四档严重度 + `drop_episode` | `flagged_ratio ≥ 0.20 → critical → drop_episode`，与"只做帧级筛选"直接冲突。新架构也已统一成规则码 + 单一 disposition |
| `exclude_mask` | 新流水线已废弃 |
| 没有校验 embodiment 对不对得上 | `embodiment_percentiles` 由调用方传入，传错了就用错误的 band 卡整条数据，无任何自检 |
| `_extract_state_and_names` 复用 `sudden_change._is_derivative_channel` | 按名猜的那套（`vel` 会误伤 `level`/`travel`），而且形成了跨模块的隐式耦合 |

---

## 二、对照 table.txt(51–53)

| 要求 | 现状 |
|---|---|
| 51. 必须对每个数据集的每个机型计算 q1/q99 | 文件结构已是 `{数据集: {机型: {维度: ...}}}`，方向对。但存的是烘死 α 的 `lo`/`hi`，且没记录来源是 state 还是 action |
| 52. 测试 α = 0/0.5/1/2/3 的区别 | **现在做不到**（§1.1）。改完存 q1/q99 之后即可扫，结果见 §1.3 |
| 53. α=0 只删 2%，不需要 episode 级 | 结论（只做帧级）同意；理由不成立——实测 α=0 删 13.4%，10.5% 的 episode 被删掉一半以上（§1.2） |

---

## 三、重构方案

### 3.1 只检 action 的关节维度，由 profile 显式给出

维度选择不再靠名字猜，与 SAA / sudden_change 共用 `dataset_profile.json`：

```json
"state_extreme": {
  "enabled": true,
  "alpha": 1.0,
  "source": "action",
  "percentile_scope": "robochallenge1/ALOHA",
  "dims": [
    {"name": "left_j0", "column": "action", "dim": 0, "q1": -0.659, "q99": 0.794},
    {"name": "left_j1", "column": "action", "dim": 1, "q1": -1.204, "q99": 0.331}
  ]
}
```

要点：

- **只放关节维度。** 手爪不是"运行时判断要不要豁免"，而是**根本不出现在 dims 里**——豁免变成一个配置事实，不是一处名字匹配。同理，末端位姿、速度、力矩也不进这个列表。
- 存 **`q1`/`q99`**，不存 `lo`/`hi`。band 在运行时按 α 算，α 才是活参数（§1.1）。
- `percentile_scope` 记录这份分位数是哪个数据集哪个机型算的，写进输出，便于事后核对。
- `source` 记录是从 state 还是 action 算的，杜绝 §1.8 那种错配。

### 3.1.1 source 只能按数据集定，不能一刀切成 action

原计划是"一律用 action 的 joint 维度"，查下来这条规则在现有三个数据集上有一半执行不了：

| 数据集 | 机型数 | source | 原因 |
|---|---|---|---|
| robochallenge1 / 2 | 8 | `state` | **该格式没有 action 通道。** `states/states.jsonl` 只有 state，`state_action_align/robochallenge.py` 的 SAA 检测直接 raise。维度名 `joint_positions_*` / `gripper_width` 也是 state 字段名 |
| robottwin2 | 5 | `action` | 维度名 `joint_action_*` 直接对应 HDF5 的 `joint_action/` 组 |

所以 §1.8 说的"用 action 分位数卡 state 值"，真相是**这份百分位文件本身就是混着的**，只是没有任何地方写下来。

真正的不变量不是"必须是 action"，而是**算分位数的信号和被检的信号必须是同一个**——`state_extreme.py` 强制要求声明 `source`，错配从此隐形不了。倾向 action 仍然成立（指令信号比反馈信号干净），只是在没有 action 的数据集上退回 state。

### 3.1.2 profile 由投影得到，不是重扫

`check_pipeline/tools/build_state_extreme_profile.py` 把 `embodiment_percentiles_ext.json` 投影成上面的 schema。**不产生任何新统计量**——`q1`/`q99` 在原文件里就是原样存着的，`lo`/`hi` 只是 α=3 的派生值（13 个机型实测精确等于 3.00）。重扫对 robochallenge 不可能（没有 action），对 robottwin2 是空转（会得到同样的数字）。

脚本做的是三件修正：丢掉 `lo`/`hi`、按数据集标 `source`、剔掉 `is_gripper` 维度。外加一处真实的错误修复：

**robottwin2/franka 的维度映射是错的。** 该机型 16 维，但名字套用了 aloha-agilex 的 6+6+2 模板：

```
 11      joint_action_right_arm_5   [-2.333, 2.478]
 12 GRIP joint_action_left_gripper  [ 1.840, 3.584]   <- 弧度，这是关节
 13 GRIP joint_action_right_gripper [-2.053, 2.706]   <- 弧度，这是关节
 14      dim_14                     [ 0.000, 1.000]   <- 这才是夹爪
 15      dim_15                     [ 0.000, 1.000]   <- 这才是夹爪
```

后果正好反着：两个真夹爪（双峰分布，明确要排除的）进了检测，两个真关节被豁免。franka 是 7 自由度，按 7+7+2 重映射。**该映射未经真实数据验证**（本机无 robottwin2 副本，moxing 未安装），输出里带 `needs_verification: true`，等能读到真数据时拿一条 episode 对一遍。

其余 12 个机型自洽：夹爪都是 0~0.087 米、0~1 归一化或 UR5 的 3~228 整数行程，关节都在弧度量级。

脚本还会提示 span 异常窄的维度（低于本机型关节 span 中位数的 10%）。当前命中一处：`robochallenge2/UR5` 的 `joint_positions_4`，span 仅 0.073 弧度（中位数的 5.9%），该关节在全数据集上几乎没动。α=1 给它的 band 只有 ±0.073 弧度，会异常敏感——跑之前得确认这个关节是真的被锁住了。

### 3.2 profile 与数据的自检

分位数传错机型的后果是"用错误的 band 卡整条数据"，而且看起来一切正常。加一道便宜的自检：

> 若超过半数维度的**本 episode 中位数**落在该维度的 `[q1, q99]` 之外，几乎可以断定这份分位数不属于当前机型 → 抛 `UnsupportedDataError`。

正常情况下 episode 的中位数应该稳稳落在数据集的 1–99 分位区间内；大面积落在外面只有一种解释。

多峰任务不会误触发这条：`[q1, q99]` 覆盖全数据集 98% 的帧，各个模态的中位数都稳稳在里面。真要触发，得是一条 episode 有一半以上的时间待在全数据集 1% 的尾巴里——那本身就值得看一眼。

维度数少于 3 时跳过这道自检：那种情况下分不清"分位数拿错了"和"这条 episode 罕见"。

### 3.3 单一规则，只删帧

| 规则码 | 触发条件 | disposition |
|---|---|---|
| `STATE_EXTREME_FRAMES` | 任一被检维度的值落在 band 外 | `drop_frames` |

没有 episode 级规则（table.txt 53）。区间左闭右开，单帧记为 `[i, i+1)`，不加 margin。

**但"不设 episode 规则"不等于对病态情况视而不见。** §1.2 表明 α 选得不好时会出现"某条 episode 被删掉 90% 的帧"。这件事应该由**流水线层的删除预算**（`DropBudget`）去发现，而不是让 state_extreme 自己改判 episode——单个检测不该越过自己的职责去做整条数据的处置决定。报告里输出 `flagged_ratio`，预算层看到分布异常就该停下来调 α。

### 3.4 NaN 报错，不静默放行

任一被检维度出现 NaN/Inf → `UnsupportedDataError`。缺失值该由上游处理掉，不能让它把整个维度的检测悄悄关掉（§1.5）。

### 3.5 参数

只有 `alpha` 一个（默认 1.0，可由 profile 按数据集覆盖）。`lo_q`/`hi_q` 固定 1/99 —— PDF 明确写死的东西不该做成参数；真要改就是换了一套定义，应该改 profile 里的 q1/q99 本身。

四档严重度的四个比例阈值全部删除。

### 3.6 让"先跑一遍再调 α"这件事有据可依

table.txt(52) 想要的是"先定一个值，跑完看结果再改"。要让这个循环闭上，光报"删了多少帧"不够——你还得知道**α 往哪个方向调多少**才会改变结果。

所以每个维度都输出 `alpha_needed_to_clear`：

> = `max over frames of (超出 [q1, q99] 的距离) / (q99 - q1)`
>
> 含义是"把 α 调到这个值以上，本维度在这条 episode 上就一帧都不再命中"。

跑完一批之后把这个数的分布拉出来，就能直接读出各档 α 的效果：正常 episode 集中在 0.1 以下，注入的毛刺在 10 以上，中间的空档就是 α 的安全区间。不用为了试一个 α 重跑一遍全量。

---

## 四、删帧的标记与同步

这是流水线里第二个产出 `drop_frames` 的检测（第一个是 sudden_change）。格式相关的坑（LeRobot 的全局 `index` 重排、`episodes_stats` 重算、H.264 无法抽帧、h5py 不能原地删行）见 `sudden_change.md` §6，此处不重复。这里只说 state_extreme 特有的两点：

**其一，删除必须是全模态同步的。** 某一帧被标记后，该 episode 里所有相机的视频帧 + parquet/hdf5 的对应行必须一起删。删了一边没删另一边就是制造全局错位——正是 SAA 要检出的失败模式。一次失败的删帧会给数据集注入 SAA 级别的损坏。

**其二，state_extreme 的命中通常是散点，不是连续段。** 这和 sudden_change 很不一样：突变是可定位的事件，极值是分布尾巴上的散点，可能散布在整条 episode 上。后果是删完之后 episode 被打成筛子——即使总删除量不大，几十个孤立的删除点意味着几十处时间不连续。

所以建议**删帧的执行不按单个检测来做，而是在 episode 级别汇总所有 `drop_frames` 之后统一决策**：

1. 汇总 sudden_change 与 state_extreme 的帧区间，合并。
2. 看合并后的形态：集中在头尾 → 截断；少数几段在中间 → 切分；散布几十处 → 这条 episode 本身就不该要，或者说明 α 定得太紧，回去调参数而不是硬删。
3. 决策定了再写出，写完必须校验（视频帧数 == parquet/h5 行数、`frame_index` 连续、重跑 corrupt、stats 一致），不过就回滚。

检测阶段只负责产出区间和 `flagged_ratio`，不碰数据。
