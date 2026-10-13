# saa —— State-Action 趋势对齐（stage 2）

> 本文只定方案，`saa.py` 还没写。

对应 PDF §2.4 Stage 2 / Qwen 的做法：

> *"In a correctly recorded episode, action commands should temporally lead or coincide with resulting state changes, which is a causal invariant violated when timestamps are unsynchronized or there is packet loss. For each shared joint dimension, we smooth both the state and action trajectories, then estimate the optimal temporal lag via cross-correlation, then compute a directional agreement (DA) metric on lag-aligned first-order differences. Dimensions with DA below a dataset-specific threshold (typically 0.6-0.7) are flagged and their episodes excluded. For datasets using delta actions, we first integrate the action sequence to recover absolute values before comparison."*

前四个检测（corrupt / black / occlusion / stillness）看的是**画面**，这个检测第一次去看**信号本身**：机器人的实际运动，是不是真的由记录下来的指令引起的。它抓的是时间戳不同步、丢包、state 与 action 来自不同时钟这类问题——这些在画面上完全看不出来，但会让训练出来的策略学到错误的因果。

---

## 1. 为什么必须重写：旧版的通过率是它自己的 bug 报告

旧版是 `new/state_action_alignment_frame.py`。它的 docstring 里记着实测通过率：

| 平台 | 通过率 |
|---|---|
| AgileX | 91.5% |
| AgiBotG1 | 20.6% |
| RoboMIND | 18.9% |
| GalaxeaR1Pro | **12.9%** |

同一套检测在不同平台上差 7 倍。这个形态不像数据质量差异，像检测在某些平台上根本没工作。用**完全正常的合成数据**跑旧版的 DA 逻辑，能复现出两种 DA≈0.5 的情况——0.5 就是随机猜的水平，说明比的两个信号根本无关：

| 输入（因果全部正常） | DA | 旧版判定 |
|---|---|---|
| 近静止关节（整条 episode 只有传感器噪声） | 0.498 | 不通过 |
| delta action 未积分 | 0.496 | 不通过 |
| delta action **积分恢复后** | 1.000 | 通过 |
| 绝对 action 领先 2 帧 | 1.000 | 通过 |

### 1.1 delta action 的积分完全没实现

PDF 原文那句 *"we first integrate the action sequence to recover absolute values"* 被完整抄进了旧版 docstring，但**全仓库搜 `cumsum` / `integrat` / `积分` 只匹配到这句引文本身**，没有任何实现。

后果是上表第二行：state 是位置、action 是位移，`diff(state)` 是速度而 `diff(action)` 是加速度，比这两个东西的符号在物理上没有意义。

AgileX 91.5% 与 Galaxea 12.9% 的差距，符合"一部分平台用 delta action"这个预测。这是可以直接验证的（见 §3.2 的判别方法）。

### 1.2 近静止维度必然失败

旧版有两道本该拦住它的防线，两道都是空的：

```python
"min_std": 1e-6,                        # 维度级: std 小于它就跳过
...
nonzero = (s_sign != 0) & (a_sign != 0) # 帧级: 只排除严格等于 0 的差分
```

`min_std = 1e-6` 是绝对阈值套在单位未知的量上（和旧 stillness 的 `state_diff_thresh=1e-4` 同病）。实测里一个"没动"的关节，噪声 std 是 `1e-3`——比阈值大三个量级，**照检不误**。

帧级那道更彻底：平滑之后浮点差分几乎永远不会**严格**等于 0，所以 `1e-9` 弧度的噪声抖动和一次真实运动**投一样的票**。14 维里有 4 维近静止，失败率就是 28.6%，直接进 `severe` 档。

这大概率是低通过率的主因。新版从根上换掉它：**维度不再由代码猜，由 profile 指定**（§3.1）。

### 1.3 检测把自己要抓的东西修正掉了

这是设计层面最严重的一处。Qwen 的定义是 action **领先或同时于** state，即 τ ≥ 0 才合法；τ < 0（state 先动）就是因果被破坏，正是要检出的目标。

但旧版在 `[-5, +5]` 全域搜最佳 lag，**负 lag 照样接受**，然后按它对齐再算 DA：

```python
best_corr, best_tau = -1.0, 0
for tau in range(-max_lag, max_lag + 1):
    ...
    c = abs(c)                    # 还取了绝对值
    if c > best_corr:
        best_corr, best_tau = c, tau
```

一条时间戳错位、state 领先 action 3 帧的 episode，会被 τ=-3 对齐得整整齐齐，DA 很高，顺利通过。**检测把自己该报的缺陷修正掉了。** τ 只在 DA 已经失败时才被记进 `failed_dims`，它本身从不触发任何东西。

`c = abs(c)` 还有一层问题：完全反相关（corr = -1）被当成最佳对齐，符号约定相反的数据集会被引到一个错误的 lag 上。

### 1.4 夹爪豁免在最常见的布局下失效

旧版注释写着 "gripper/effector 全豁免"，靠的是列名子串匹配。实测：

```
场景A  OXE / GM-100 风格 (一个 observation.state 列装 14 维)
       展开后维度名是 d0 ... d13, 名字里没有 "gripper" 这个词
       匹配到的共享维度: ['d0', ..., 'd12', 'd13']
       夹爪维 d13 被排除? False        ← 豁免失效

场景B  显式命名 (states.gripper.position)
       夹爪被排除? True                ← 豁免生效
```

夹爪恰恰是最容易 DA 失败的维度（离散指令 vs 连续机械响应，还带大滞后），把它算进去纯属往失败率里灌水。

这条 bug 也是"**靠列名猜维度语义这条路已经被证伪**"的最好证明——`d13` 这种名字里没有任何信息，再完善的黑白名单也救不了。

### 1.5 其余问题

- **名字匹配实为位置匹配**：`d0` 配 `d0`，假设 state 和 action 的维度顺序一致。顺序不同就静默配错，比不做还危险。
- **`_safe_savgol` 的 `except Exception: return x`**：scipy 缺失时所有维度静默改用未平滑数据，噪声不去除、DA 暴跌，而且没有任何记录。
- **`T = min(T_s, T_a)`**：state 与 action 长度不一致时静默截断。
- **`max_lag = 5` 单位是帧**：30fps 下是 167ms，10fps 下是 500ms，物理含义随数据集漂移。
- **单一全局 lag + episode 级 DA**：1500 帧里 200 帧的丢包突发被稀释成 DA≈0.87，顺利通过；而且无法定位问题在哪一段。
- **三级分档 + `split_or_drop_episode`**：和其他模块同样的毛病。而且 `moderate → drop_segment_with_margin` 里的"段"就是整条 episode（`frame_range=(0, T)`），这个动作本身没有意义。
- **`_directional_agreement` 在没有非零差分时 `return 1.0`**："没测出来"被当成"测过了没问题"，是假阴性。

---

## 2. 新版的结构

```
profile 给出:  共享 joint 维度配对 + action_space + da_threshold
                     │
                     ▼
    ① 按 profile 取出 state / action 的对应维度
    ② action_space == delta  ->  积分恢复绝对量
    ③ 平滑 (Savitzky-Golay, 窗口按秒给)
    ④ 互相关估计最佳 lag τ*  (全域 [-L, +L], 取**正相关**最大)
    ⑤ τ* 显著为负  ->  CAUSALITY_VIOLATED
    ⑥ 在 τ = max(τ*, 0) 对齐, 算一阶差分上的 DA
    ⑦ 任一维度 DA < da_threshold  ->  drop_episode
```

**"测量与判定分离"** 和前四个检测一致：④⑥ 只测量，⑤⑦ 才判定。所有中间量（每维的 τ、corr、DA、可用帧数）都进 `measurements`，不管过没过。

---

## 3. 判据的三个关键决定

### 3.1 维度从 profile 来，不在代码里猜

这是新版和旧版最大的区别，也是 §1.2 和 §1.4 的根治办法。

`dataset_profile.json` 里给出（由 stage 0 的扫描 + LLM 标注 + 人工确认产出，见 `dataset_profile.md`）：

```jsonc
"saa": {
  "action_space": "absolute",          // absolute | delta
  "shared_dims": [                     // 只有 state/action 都有且维度一致的关节才进来
    {"name": "left_arm_joint_0",  "state_dim": 0, "action_dim": 0},
    {"name": "left_arm_joint_1",  "state_dim": 1, "action_dim": 1},
    {"name": "right_arm_joint_0", "state_dim": 7, "action_dim": 7}
  ],
  "excluded_dims": [
    {"dims": [6, 13], "reason": "gripper",   "note": "离散指令 vs 连续响应, 不适用 DA"},
    {"dims": [14, 15], "reason": "velocity", "note": "导数通道"}
  ],
  "da_threshold": 0.7
}
```

选维度的规则（对应 `table.txt` 36–37）：

- 只要**运动相关**的关节维度：单臂 / 双臂 / 底盘
- 排除电机速度、力矩、传感器等**非位置量**
- 必须 **state 和 action 都有且维度对得上**，缺一边的不进
- **夹爪不参与 SAA**

`shared_dims` 给的是**显式的下标配对**，不是名字。名字只用于报告可读性。这样 `d0`/`d13` 那种无信息命名不再是问题，state 与 action 维度顺序不一致也不再是问题——配对关系在 profile 里是写死的。

profile 缺失或 `shared_dims` 为空 → `UnsupportedDataError`。**不退化成"用列名猜"**，那正是旧版出问题的地方。

### 3.2 delta vs absolute：profile 钉死，运行时复验

`action_space` 决定要不要先积分，搞错了 DA 直接掉到 0.5（§1.1 实测）。

**自动判别方法**（进 `dataset_facts.json`，人确认后冻进 profile）：把两种假设都算一遍——

```
DA_as_absolute = 符号一致率( diff(state), diff(action) )   # 假设 action 是绝对量
DA_as_delta    = 符号一致率( diff(state), action )         # 假设 action 是增量
```

因为 delta action 本身就约等于 `diff(state)`，哪个假设成立哪个数就高。实测分离度：

| 真实类型 | DA_as_absolute | DA_as_delta | std(action)/std(state) | 判定 |
|---|---|---|---|---|
| 绝对 | 0.938 | 0.498 | 1.000 | absolute |
| 增量 | 0.497 | 0.938 | 0.010 | delta |

两个互相独立的证据（DA 对比 + 幅度比，后者差 **100 倍**）互相印证，判错概率很低。

**运行时复验**：检测里再算一次这两个数，和 profile 声明的不一致就 `UnsupportedDataError`。一条 episode 的 action 空间不可能和数据集其他部分不同，真出现了那是数据有问题，正该报出来——而不是让检测自己改主意。

**积分的实现细节**：`a_abs[t] = state[0] + Σ_{i<t} action[i]`。锚到 `state[0]` 只是让曲线可读，对结果无影响（互相关会去均值）。

值得注意的一个恒等式：`diff(cumsum(action))[t] = action[t+1]`。所以**就 DA 而言**，"先积分再一阶差分"等价于"直接用 action"，差一帧偏移会被 lag 搜索吸收——这正是上面判别式能成立的原因。但**互相关估计 lag 时积分是必要的**：在绝对轨迹上做互相关比在差分上稳得多。

**已知风险**：积分会累积漂移，1500 帧下来小偏置会攒成明显的趋势，影响互相关。DA 本身对漂移不敏感（差分抵消掉了），受影响的是 lag 估计。做出来之后要看 `measurements` 里积分轨迹与 state 的末端偏差。

### 3.3 因果方向是**判据**，不是对齐手段

这条直接修 §1.3。

- 互相关在 `[-L, +L]` 全域搜索，取**正相关**最大（不取 `abs`），得到 τ*
- **τ* < -1 帧**（state 显著领先 action）→ 触发 `SAA_CAUSALITY_VIOLATED`，独立成一条规则
- 算 DA 时用 `τ = max(τ*, 0)`——不允许用负 lag 去"修好"因果

留 1 帧容差是因为平滑和量化本身会带来亚帧级的相位偏移，硬卡 τ*<0 会误报。

L 按**秒**给（`max_lag_sec`），运行时换算成帧，这样 10fps 和 100fps 的数据集含义一致。

---

## 4. 规则表

| 规则码 | 触发条件 | 处理 |
|---|---|---|
| `SAA_DA_BELOW_THRESHOLD` | **任意一个**共享关节维度的 DA < `da_threshold` | `drop_episode` |
| `SAA_CAUSALITY_VIOLATED` | **任意一个**共享关节维度的 τ* < −1 帧（state 领先 action） | `drop_episode` |

按 `table.txt` 第 4 条：**episode 级，不对就整条删**。没有分档，没有删帧，和 black / occlusion / stillness 形态一致。

"任意一个维度"而不是"超过百分之多少的维度"：Qwen 原文是 *"Dimensions with DA below ... are flagged and their episodes excluded"*，一个维度不对就说明这条 episode 的时序有问题，没有理由认为其他维度的数据是可信的。这也去掉了旧版 `fail_ratio` 那套分档。

### 两条规则会同时命中的情况

如果某一维的 action 信号和 state **完全无关**（不是错位，是根本对不上），两条规则都会触发。实测：

```
da = 0.495   best_lag = -6   cross_corr = 0.003
triggered: ['SAA_CAUSALITY_VIOLATED', 'SAA_DA_BELOW_THRESHOLD']
```

判定没错——这条 episode 本来就该删。但**报出来的理由会误导人**：`best_lag = -6` 读起来像"state 领先 6 帧的时间戳错位"，而 `cross_corr = 0.003` 说明这个 lag 是在拟合噪声，"领先 6 帧"这个说法根本不成立。

这里**不加相关系数下限**去压掉它——那又是一个没有依据的阈值，而且它不改变任何删除决策，只影响诊断可读性。改成让证据自己说话：`SAA_CAUSALITY_VIOLATED` 的 evidence 里 `per_dim_cross_corr` 和 `per_dim_best_lag` 并排给出。看报告时的规则是：**cross_corr 很低就别信那个 lag**，去看同时触发的 DA 规则，问题在 action 信号本身。

---

## 5. 参数

```python
DEFAULT_PARAMS = {
    "da_threshold": 0.70,        # 0.6-0.7, 由 profile 按数据集给, 默认取严的一端
    "max_lag_sec": 0.20,         # 互相关搜索范围, 按秒 (30fps -> ±6 帧)
    "smooth_window_sec": 0.17,   # Savitzky-Golay 窗口, 按秒 (30fps -> 5 帧)
    "smooth_poly": 2,
    "causality_tol_frames": 1,   # τ* 允许的负向容差
}
```

### 5 个参数里只有 2 个决定删不删

| 参数 | 类型 | 作用 |
|---|---|---|
| `da_threshold` | **判定** | DA 低于它 → 删 |
| `causality_tol_frames` | **判定** | τ\* 比它更负 → 删 |
| `max_lag_sec` | 测量 | lag 搜索范围 |
| `smooth_window_sec` | 测量 | 平滑窗口 |
| `smooth_poly` | 测量 | 平滑阶数 |

后三个是"怎么测"而不是"在哪切"——调它们只会让 DA 这个数轻微移动，本身不构成任何一条过/不过的分界线。判定路径上只有两个数：`da < da_threshold` 和 `τ* < -causality_tol_frames`。

对比旧版：`da_threshold` 加上分档表里的 `fail_ratio >= 0.5`、`min_da < 0.3`、`fail_ratio >= 0.2`、`min_da < 0.5` 共 **5 个判定阈值**，外加 `min_std` / `min_episode_len`，输出 3 种不同动作。新版是 2 个判定阈值、1 种动作。

`causality_tol_frames = 1` 不是"调松调紧"的旋钮，是个容差：平滑和量化会带来亚帧级相位偏移，硬卡 `τ* < 0` 会把这些数值噪声全报成因果错乱。

其余说明：

- 三个"按秒"的参数是为了跨帧率可比，旧版按帧给是错的
- **没有 `min_std`**：静止维度的问题在 profile 层面解决（只选运动相关的关节），不在这里靠阈值兜

### 为什么不能只留 DA 一个阈值

Qwen 原文只提了 DA 一个阈值，所以自然会问：能不能把 lag 搜索限制在非负区间，让因果错位自己掉进 DA 里？**实测不行。**

state 领先 k 帧（因果异常），lag 搜索限制在 `[0, 6]`：

| 运动特征 | k = −3 | k = −6 |
|---|---|---|
| 慢速（周期 40 帧） | DA = 0.858 → 通过 | DA = 0.713 → 通过（压线） |
| 快速（周期 12 帧） | DA = 0.503 → 拦下 | DA = **1.000** → 通过 |

原因：DA 是**符号**统计，对平滑信号上的小时移天生不敏感——导数只在拐点附近变号，错位几帧的符号大部分仍然一致。

最后那格是另一个问题：`max_lag = 6` 与运动周期 12 帧接近时，lag 搜索找到了一个**混叠**的假对齐，DA 反而是完美的 1.000。

**所以 `max_lag_sec` 必须显著小于关节运动的特征周期**。0.2 秒在 30fps 下是 6 帧；如果某个数据集的关节运动周期只有十几帧，这个值要往下调，否则 lag 搜索会自己造出假对齐。第一批跑完看 `per_dim` 里的 `lag_frames` 分布：如果大量维度顶在 `max_lag` 边界上，就是这个信号。

---

## 6. 报错项（`UnsupportedDataError` / `PipelineEnvironmentError`）

| 情况 | 处理 |
|---|---|
| 没有 profile，或 `saa.shared_dims` 为空 | `UnsupportedDataError` |
| `shared_dims` 的下标超出实际 state/action 维度范围 | `UnsupportedDataError` |
| state 与 action 行数不一致 | `UnsupportedDataError`（**不截断**） |
| 实测的 action_space 与 profile 声明不符 | `UnsupportedDataError` |
| episode 长度不足以做平滑/lag 搜索 | `UnsupportedDataError` |
| scipy 不可用 | `PipelineEnvironmentError`（**不静默改用未平滑数据**） |

最后一条是旧版 `except Exception: return x` 的直接反面。平滑失效会让 DA 系统性偏低进而大批误删，这种事必须停下来而不是继续跑。

---

## 7. 保持段的影响：不加死区，但要能验证

**决定**：DA 不加帧级死区（不排除小幅差分的帧）。理由是维度选择已经在 profile 层面把"整条 episode 不动的关节"排除掉了，选进来的关节在操作过程中大部分时间是在动的。

但这个决定有个边界，实测量化如下。一个**整体在动**、但 episode 内周期性保持不动的关节，因果完全正常：

| 保持段占比 | DA（无死区） | DA（死区=\|Δ\| 中位数） | 判定 @0.7 |
|---|---|---|---|
| 0% | 0.952 | 1.000 | 通过 |
| 20% | 0.855 | 0.989 | 通过 |
| 30% | 0.795 | 0.983 | 通过 |
| 40% | 0.761 | 0.976 | 通过 |
| 50% | 0.714 | 0.916 | 通过（余量已经很薄） |
| **60%** | **0.660** | 0.836 | **误删** |

结论：这个决定在保持段占比 ≤50% 时成立，**超过 60% 会开始误删因果正常的数据**。换句话说，不加死区时 DA 有一部分是在测"这个关节花了多少时间保持不动"，而不是在测"因果是否完整"。精细操作、双臂协作时单臂等待，都可能落进 50–60% 这一带。

**所以行为不改，但加一个零成本的诊断**：`measurements` 里输出同一维度在几个死区档位下的 DA——

```jsonc
"da_at_deadband": {"0": 0.714, "p25": 0.842, "p50": 0.916}
```

死区取该维度自身 `|diff(state)|` 的分位数，**不引入任何新参数**，也不参与判定。第一批跑完看这三个数：

- 三个数接近 → 保持段不是问题，这个决定成立，以后可以不看了
- `"0"` 明显低于 `"p50"` → 保持段在压低 DA，被删的 episode 里有一部分是误删

这样把一个判断变成一个可以被数据证伪的假设，而不是一直靠推测。

---

## 8. 输出格式

```json
{
  "check": "saa",
  "drop_episode": true,
  "triggered_rules": ["SAA_DA_BELOW_THRESHOLD"],
  "findings": [
    {
      "rule": "SAA_DA_BELOW_THRESHOLD",
      "description": "存在共享关节维度的方向一致性低于阈值 (state 与 action 的因果关系被破坏)",
      "disposition": "drop_episode",
      "dims": ["right_arm_joint_2"],
      "evidence": {
        "criterion": "min(DA over shared dims) >= da_threshold",
        "threshold": 0.70,
        "action_space": "absolute",
        "n_dims_checked": 12,
        "n_dims_failed": 1,
        "per_dim_failed": {
          "right_arm_joint_2": {
            "state_dim": 9, "action_dim": 9,
            "da": 0.412, "lag_frames": 0, "lag_sec": 0.0,
            "cross_corr": 0.31, "n_frames_used": 1487,
            "da_at_deadband": {"0": 0.412, "p25": 0.430, "p50": 0.445}
          }
        },
        "per_dim_da": {
          "left_arm_joint_0": 0.981, "left_arm_joint_1": 0.975,
          "right_arm_joint_2": 0.412
        }
      }
    }
  ],
  "measurements": {
    "n_frames": 1500,
    "fps": 30.0,
    "action_space_declared": "absolute",
    "action_space_measured": "absolute",
    "action_space_evidence": {"da_as_absolute": 0.94, "da_as_delta": 0.49,
                              "std_ratio": 1.02},
    "params": {"...": "本次实际生效的参数"},
    "per_dim": {
      "left_arm_joint_0": {
        "state_dim": 0, "action_dim": 0,
        "da": 0.981, "lag_frames": 2, "lag_sec": 0.067,
        "cross_corr": 0.97, "n_frames_used": 1487,
        "state_std": 0.42, "action_std": 0.43,
        "da_at_deadband": {"0": 0.981, "p25": 0.994, "p50": 0.998}
      }
    }
  }
}
```

`per_dim_da` 把**所有**维度的 DA 都列出来，不只是失败的那个。只列失败维度就没法回答"是这一维特别差，还是整体都在临界线附近晃"——这两种情况的排查方向完全不同。

`lag_frames` 全维度一起看信息量最大：所有维度的 τ 都是 2 说明系统有个稳定的 167ms 控制延迟（正常）；τ 散落在 0–6 之间说明时间戳本身不稳（可疑）。

---

## 9. 第一批跑完要看什么

1. **`action_space_measured` 与 profile 声明的一致率**。有不一致就说明 profile 定错了，先修 profile 再谈数据质量。

2. **通过率是否还是那种 12% vs 91% 的形态**。如果修完 delta 积分和维度选择之后仍然有平台在 20% 以下，那才可以开始怀疑数据本身；在此之前，低通过率的第一嫌疑人始终是检测。

3. **`da_at_deadband` 三档的差距**（§7）。这决定要不要回头加帧级死区。

4. **`SAA_CAUSALITY_VIOLATED` 的触发量**。这条规则旧版等于不存在，是新增的检出能力，头一次跑出来的量需要人工抽查确认不是误报。两种要排除的假阳性：τ* 只有 -2 这种临界值（可能是平滑相位偏移而不是真的因果错乱）；`cross_corr` 很低（lag 在拟合噪声，真正的问题是 action 信号本身，见 §4）。

5. **`DropBudget` 里 SAA 的唯一归因**。SAA 是五个检测里唯一看信号的，它删掉的 episode 很可能和前四个完全不交叠，对总删除率的边际贡献会比较大。
