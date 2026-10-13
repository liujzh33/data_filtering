# stillness — 静止帧检测（重写版）

**文件**：`check_pipeline/stillness.py`
**入口**：`check_stillness(probes, state, params, state_columns=None) → StillnessReport`
**依赖**：`probe.py`（相邻帧像素差）、`errors.py`、`intervals.py`

---

## 1. 它只回答一个问题

**这条 episode 里有多少帧是"什么都没发生"的。** 机器人没动，画面也没变——这些帧对策略学习没有信息增益，典型出现在 episode 首尾（等待开始、做完之后停着）。

---

## 2. 先说一件最重要的事：这个检测和前三个性质不同

corrupt / black / occlusion 找的是**坏数据**。stillness 找的是**没用的数据**。静止帧本身没有任何错误，它只是不提供训练信号。

删除门槛统一取 **10%**，三个检测口径完全对称：

| 检测 | 删除门槛 | 找的是 |
|---|---|---|
| black | 10% | 坏数据 |
| occlusion | 10% | 坏数据 |
| stillness | 10% | 没用的数据 |

统一的好处是结果可以横向比较，`DropBudget` 的交叠统计也才有意义（三条线不一样的话，"哪个检测删得最多"这个问题会被阈值差异污染）。

**但必须清楚这条线对 stillness 意味着什么**：

```
一条 50 秒 (1500 帧 @30fps) 的 episode
只要有连续 5 秒 (150 帧) 没动  →  整条删掉
```

而开头等指令、结尾等复位停 3–5 秒在真实采集里很常见，不是数据质量问题。**这条线大概率会误删一批正常数据**——black/occlusion 的 10% 拦的是"相机坏了 5 秒"，那确实是故障；stillness 的 10% 拦的是"机器人停了 5 秒"，那可能只是操作员在等。

所以跑第一批之后**必须**做这件事：看被 stillness 删掉的 episode，它们的 `longest_still_run_span` 是不是普遍紧贴 `[0, x)` 或 `[n-x, n)`。

- 是 → 拦的是采集停顿而不是异常停滞，往上调 `max_still_run_ratio`
- 否（停滞在轨迹中间）→ 拦对了，是机器人卡住/操作中断

`DropBudget` 的唯一归因能直接给出代价：`unique[stillness]` 就是"只有 stillness 判死、其他三个检测都认为没问题"的 episode 数。这个数如果很大，说明门槛定紧了。

### 放宽的是容忍度，不是判据

这两件事必须分开：

- **判据**（`state_diff_thresh` / `image_diff_thresh`）决定**什么叫静止**。这是物理问题，放宽只会让判定失真——把明明在动的帧算成"在动"本来就是对的。
- **容忍度**（`max_still_run_ratio`）决定**能接受多长的一段停滞**。这是业务决定，本来就该按数据用途调。

放宽的是后者。

### 统计口径：最长连续段，不是总量

看的是最长的那一段静止占了 episode 多大比例，其他零散的静止帧不参与判定。**这一点和 black / occlusion 完全一致**，三个检测口径统一，结果可以横向比较。

对 stillness 来说这个口径还额外贴合语义：PDF 原文要求的就是 "remove **PROLONGED** static segments"——要删的是**持续**的停滞。

代价的反面也很关键：机器人操作过程中本来就有大量**瞬时微停顿**（接近目标点时减速、抓取瞬间的停顿、换手的过渡）。这些帧单独看都满足"state 没动且画面没变"，但它们是任务的一部分。按总量算会把它们累加起来，一条完全正常的精细操作轨迹可能有 40% 的帧落在微停顿里，直接被判死。按最长连续段算，这些一两帧的停顿根本进不了判定。

回归测试里验证了这个行为：50 个**孤立**的静止帧（总量 50%，最长连续段只有 1 帧）不触发；连续 40 帧（40%）触发。

---

## 3. 判据：机器人没动 **且** 这一路的画面没变

```
state 侧:  ‖state[t] - state[t-1]‖  <  state_diff_thresh
image 侧:  frame_diff[t]            <  image_diff_thresh     (逐路各判各的)

静止帧  ⟺  两侧都满足
```

取**与**的理由：

- 只有 state 静止：可能是机器人停着但场景里有人在动（信息还在）
- 只有画面静止：可能是机器人在动，但动作幅度小到画面看不出

都不算冗余。

### `frame_diff` 是什么

`probe.py` 新增的测量：相邻两帧的**平均绝对像素差**（灰度级）。

```
frame_diff[i] = mean(|frame[i] - frame[i-1]|)
```

只在第 `i` 和第 `i-1` 帧都成功解码**且帧号连续**时才有值，否则是 NaN——跨过坏帧比出来的差分数值上偏大且没有物理意义。`frame_diff[0]` 恒为 NaN。

实现上是**边解码边算，只留上一帧**。把全部灰度数组存下来再统一算的话，1500 帧 480×640 float32 就是 1.8 GB。

---

## 4. 规则表：只有一条

| 规则码 | 触发条件 | 处理 |
|---|---|---|
| `STILLNESS_RUN_RATIO_EXCEEDED` | **任意一路**相机的**最长连续**静止段 ≥ `max_still_run_ratio`（默认 **10%**） | `drop_episode` |

结构上和 black / occlusion 完全对称：同样的判据形式、同样的门槛（10%）。分母是该路 `frame_diff` 非 NaN 的帧数（`frame_diff[0]` 恒为 NaN，所以分母通常是 `n_frames - 1`）。这条门槛对正常采集停顿的杀伤力见 §2。

静止帧总量仍然测出来放在 `measurements.views.still_ratio` 和 `evidence.per_view_still_ratio_total` 里，只是不参与判定。`longest_still_run_span` 给出触发判定的那一段的位置——排查时直接看那一段是"开头等指令"还是"中途卡住了"。

---

## 5. 参数

```python
DEFAULT_PARAMS = {
    "state_diff_thresh": None,     # 必传, 无默认值
    "image_diff_thresh": 2.0,      # 相邻帧平均像素差 (灰度级)
    "max_still_run_ratio": 0.10,   # 任一视角最长连续静止段达到它 -> 丢整条
}
```

### 为什么 `state_diff_thresh` 必须自己传

旧版默认 `1e-4`。旧 docstring 自己列了实测数据：

| 平台 | ‖Δstate‖ 的 1% 分位 |
|---|---|
| AgileX | 0.0（离散控制） |
| AgiBotG1 | 0.000171 |
| GalaxeaR1Pro | **0.061** |

GalaxeaR1Pro 连**最静止的 1% 帧**差值都有 0.061，是默认阈值的 **600 倍**——这个平台上 state 侧永远判不出静止，整个检测静默失效。AgileX 的 q1 = 0.0 则意味着至少 1% 的帧完全不变。

同一个数字在不同平台上行为天差地别，因为 state 的单位取决于机型（弧度 / 米 / 归一化值）。**我们不知道就不能猜。**

所以这一版把它设成必传参数，不传直接报 `UnsupportedDataError`。这就是"有错就报错，不要硬筛选"用在阈值上的样子——一个猜出来的阈值不会报错，它只会静默地给出错误答案。

**怎么标定**：拿该机型一批 episode 算 `‖Δstate‖` 的分布，会看到明显的双峰（真静止段一个峰，正常操作一个峰），取静止峰的上沿。

### 为什么 `image_diff_thresh` 可以有默认值，以及为什么是 2.0

单位是**固定的**——0–255 灰度，和机型无关，所以可以给默认值。

**下界由传感器噪声定。** 两帧各带 σ 的独立噪声时，帧间平均绝对差的期望是 `σ·√2·√(2/π) ≈ 1.13σ`：

```
σ=0.5 → frame_diff ≈ 0.56    判得出静止
σ=1.0 → frame_diff ≈ 1.13    阈值取 1.0 就判不出了
σ=2.0 → frame_diff ≈ 2.26
```

上一版取 1.0 只覆盖 σ<0.9 的相机——真静止的画面会因为噪声略超阈值而判不出静止，是**漏检**方向的错。2.0 覆盖到 σ≈1.8，涵盖大多数真实相机。

**上界由正常操作定。** 机械臂占画面 10–30%、移动区域像素变化 20–80 灰度级，全帧平均落在 5–15。2.0 离它有 2 倍以上余量。

### 一个必须知道的耦合：编码方式

**H.264 对静止场景用 skip 宏块，解出来的相邻帧是逐位相同的，`frame_diff` 精确为 0。**

所以在常规 IPPP 编码的数据上，静止段的 frame_diff 是 0 而不是 1.13σ，这个阈值取 1 还是 3 都无所谓。但一旦某个数据集用 all-intra（每帧独立编码），噪声就全暴露出来，同一个阈值的行为完全变了。

换数据集时先看 `measurements.views[cam].frame_diff_percentiles.p1` 再决定要不要调：接近 0 说明是 IPPP，阈值不敏感；明显大于 0 说明噪声没被压缩吃掉，需要按上面的公式重新定。

---

## 6. 修掉了旧版的哪些问题

旧版是 `new/stillness_frame.py`。下面 1–3 是实打实的代码 bug，4–10 是设计问题。

### 6.1 `still_views` 把相机下标当帧下标用 🐛

```python
n_still_cams = np.zeros(n_frames, dtype=int)      # 一维: 每帧有几路静止

still_views = [cameras[v] for v in range(len(cameras))
               if v < n_still_cams[s:e].shape[0] and n_still_cams[s:e][v].any()]
```

`n_still_cams[s:e][v]` 取的是**段内第 v 帧**的静止相机数，却被当成"第 v 个相机是否静止"来决定 views 列表。

索引语义完全错位——拿相机索引去索引帧维度。输出的 `views` 字段基本是随机的：它实际表达的是"段内前 `len(cameras)` 帧里，哪几帧有相机在静止"。

### 6.2 `typical_cams` 和 `k` 算完从没用过 🐛

```python
typical_cams = int(np.median(n_still_cams[s:e]))
k = min(max(typical_cams, 1), 3)                  # 算了
sev = classify_duration(dur_sec, levels)          # 用的是 levels, 不是 levels[k]
```

分档表只定义了 `still_1cam_sec` 一档，"按视角数分级"的意图从来没实现过。两行死代码。

### 6.3 同一份 motion 算了两遍 🐛

L154–164 算了一次 `motion`，L192–202 又一模一样地算了一次（为了填 `n_still_cams`）。纯浪费。

### 6.4 image 静止判据用错了量

```python
motion = |Δgray_mean| + |Δgray_std| < 0.1
```

两个问题。

**一、量纲混合。** 亮度差和纹理差直接相加，得到的数没有物理意义。

**二、更根本的：整帧灰度均值判不出画面有没有变。** 机械臂在画面里移动时它占的像素少，整帧均值几乎不变——于是"机械臂在动"会被判成图像静止。

旧 docstring 自己也自相矛盾：

> "各平台 motion q1=0.003~0.006, q5=0.014~0.017, **0.1 约在 q25-q50 之间**，只抓接近完全相同的帧"

0.1 落在 q25–q50 意味着 **25%–50% 的帧**都会被判图像静止，跟"接近完全相同"差得远。

**新版怎么解决**：改用 `probe.frame_diff`（相邻帧平均绝对像素差），直接回答"画面变了多少灰度级"，机械臂动了就一定反映在像素差上。

### 6.5 "所有相机都静止才算"

```python
image_mask[:N] = is_static_per_view.all(axis=1)
```

三路相机里有一路噪点大一点，整条 episode 就永远判不出静止。

而需求要的恰恰相反——**任意一路**满足占比就删。

**新版怎么解决**：逐路独立统计，`still[cam][t] = state_static[t] and image_static[cam][t]`，每路各算各的占比。回归测试里有这个场景：a 路画面静止、b 路满是噪点，新版正确地只点名 a 路。

### 6.6 `state_diff_thresh` 硬编码，跨平台失效

见 §5。

### 6.7 exclude_mask 一路坏就让整个检测失效 ⚠️

```python
joint_mask &= ~metrics.exclude_mask[:n_frames].any(axis=1)
```

用的是 `any(axis=1)`：只要**任意一路**在某帧被前序排除，这一帧就不判静止。

corrupt 报了一路整路坏（整列 True）的话，`any` 全 True，`joint_mask` 被整个清零，静止检测**一帧都不报**——而输出看起来就是"这条 episode 没有静止"。

这比 black / occlusion 的同款问题更严重：那两个是逐视角相减，只有被排除的那一路失效；这里是一路坏掉、**整个检测失效**。

**新版怎么解决**：没有 `exclude_mask`。坏帧在 probe 里是 NaN，`frame_diff` 在那些位置也是 NaN，天然不计入。

### 6.8 时间轴长度不等时静默截断（或崩溃）

```python
N = min(len(state_mask), len(image_mask))
state_mask[1:1 + len(is_static)] = is_static
```

state 比视频短时直接截断，长的那一段就没了；state 比视频长时那行赋值会抛 `ValueError`。

**新版怎么解决**：长度不等直接报 `UnsupportedDataError`。corrupt 的 `ACTION_VIDEO_LENGTH_MISMATCH` 本来就该先拦下这种数据；截断对齐会悄悄丢掉一端，而"哪一端是对的"我们并不知道。

### 6.9 夹爪豁免做反了

PDF 要求 "task-critical key frames such as **gripper closure are explicitly preserved**"——夹爪闭合的那几帧不能删。

旧版的做法是算 state 差分时把夹爪维度**剔除**：

```python
gripper_idx = _gripper_indices_from_signals(metrics)
keep = [d for d in range(D) if d not in gripper_idx]
state_for_diff = state[:, keep]
```

结果是"手臂不动、只有夹爪在合"的帧，因为剩余维度没变化被判成 state 静止——**恰好把 PDF 点名要保护的帧标成了可删。** 方向反了。

**新版怎么解决**：要让夹爪帧不被删，夹爪一动就必须判为**不静止**，也就是夹爪维度必须**参与**差分。新版保留全部位置维度，不做剔除。

回归测试里有这个场景：手臂全程不动、夹爪在第 5–10 帧闭合，新版正确地把这几帧排除在静止集合之外。

### 6.10 一串静默降级

| 旧版行为 | 新版 |
|---|---|
| `fps = metrics.fps or 30.0` | 拿不到 fps 报错 |
| state 只有导数通道 → 退化成纯 image 检测 | 报错 |
| 无 state 也无 image → 返回空列表（= 干净） | 报错 |
| `has_image` 判据用 `size > 1`（N=1,V=2 时也成立，但 1 帧做不了差分） | 按 `frame_diff` 的 NaN 自然处理 |

"退化成纯 image 检测"尤其危险：没有 state 约束时，"机器人在动但画面变化小"会被当成静止，正好是 §6.4 说的那个误判。

---

## 7. 报错 vs 筛选

| 情况 | 处理 |
|---|---|
| 不传 `state_diff_thresh`，或传了非正数 | `UnsupportedDataError` |
| probe 没带 `frame_diff` | `UnsupportedDataError` |
| state 行数 ≠ 视频帧数 | `UnsupportedDataError`（corrupt 该先拦下） |
| parquet 里认不出 state 列 | `UnsupportedDataError` |
| state 列全是导数通道 | `UnsupportedDataError`（不退化成纯 image） |
| 拿不到 fps / fps 不一致 / 各相机长度不等 | `UnsupportedDataError` |
| 一路可用的相机都没有 | `UnsupportedDataError` |
| numpy / pandas 不可用 | `PipelineEnvironmentError` |

---

## 8. 输出格式

```json
{
  "check": "stillness",
  "has_stillness": true,
  "drop_episode": true,
  "triggered_rules": ["STILLNESS_RUN_RATIO_EXCEEDED"],
  "findings": [
    {
      "rule": "STILLNESS_RUN_RATIO_EXCEEDED",
      "description": "存在相机的最长连续静止段达到阈值占比 (机器人持续没动且画面没变 = 冗余停滞)",
      "disposition": "drop_episode",
      "views": ["cam_high", "cam_left"],
      "frame_ranges": [[1, 560], [1247, 1500]],
      "n_frames_affected": 812,
      "evidence": {
        "criterion": "longest_still_run / n_valid",
        "threshold": 0.10,
        "state_diff_thresh": 0.001,
        "image_diff_thresh": 2.0,
        "n_state_static": 828,
        "n_views_checked": 2,
        "n_views_over": 2,
        "per_view_over": {
          "cam_high": {
            "longest_still_run": 559,
            "longest_still_run_ratio": 0.372915,
            "longest_still_run_span": [1, 560],
            "n_valid": 1499,
            "n_still_total": 812, "still_ratio_total": 0.541695,
            "n_segments": 2, "n_image_static": 880
          }
        },
        "per_view_longest_run_ratio": {"cam_high": 0.372915, "cam_left": 0.368913},
        "per_view_still_ratio_total": {"cam_high": 0.541695, "cam_left": 0.536891}
      }
    }
  ],
  "measurements": {
    "n_frames": 1500,
    "fps": 30.0,
    "n_state_static": 828,
    "state_columns": ["observation.state"],
    "state_dims_used": 14,
    "params": { "...": "本次实际生效的 3 个参数" },
    "views": {
      "cam_high": {
        "n_frames": 1500, "n_valid": 1499,
        "n_still": 812, "still_ratio": 0.541695,
        "still_ranges": [[1, 560], [1247, 1500]],
        "longest_still_run": 559,
        "longest_still_run_ratio": 0.372915,
        "longest_still_run_span": [1, 560],
        "n_image_static": 880,
        "frame_diff_percentiles": {"p1": 0.0, "p50": 6.4, "p99": 21.3}
      }
    }
  }
}
```

排查时最有用的几个数：

- **`longest_still_run_span`**：触发判定的就是这一段。贴在首尾说明是正常的采集停顿（见 §2），落在中间才是异常（比如机器人卡住了）。上面这个例子里 `[1, 560]` 是开头 18 秒没动，属于采集流程问题而不是数据损坏。
- **`still_ratio`（总量） vs `longest_still_run_ratio`（最长段）**：两者接近说明停滞集中成一段；总量远大于最长段说明是密集的微停顿——后者不会触发删除，是正常的精细操作。
- **`n_state_static` vs 各路的 `n_image_static`**：两个数差很多说明某一侧的阈值不合适。state 侧远小于 image 侧，通常是 `state_diff_thresh` 定得太严。
- **`frame_diff_percentiles`**：`p1` 应该明显小于 `image_diff_thresh = 2.0`（真静止的帧，IPPP 编码下通常接近 0），`p50` 应该明显大于它（正常操作的帧，一般 5–15）。如果 `p1` 就比 2.0 大，说明这批数据是 all-intra 编码或噪声偏高，阈值抓不到任何东西——按 §5 的公式重新定。

---

## 9. 用法

```python
from check_pipeline.probe import probe_video
from check_pipeline.stillness import check_stillness, extract_state_matrix
from check_pipeline.errors import CheckError

videos = {"cam_high": open("cam_high.mp4","rb").read(),
          "cam_left": open("cam_left.mp4","rb").read()}
parquet = open("episode_000000.parquet","rb").read()

try:
    probes = {cam: probe_video(b, cam) for cam, b in videos.items()}
    state, cols = extract_state_matrix(parquet)
    report = check_stillness(
        probes, state,
        params={"state_diff_thresh": THRESH_BY_PLATFORM[platform]},   # 必传
        state_columns=cols)
except CheckError as e:
    log_check_failed(episode_id, type(e).__name__, str(e))
else:
    if report.drop_episode:
        ...   # 丢弃整条 episode
    json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
```

`state_diff_thresh` 应该按机型维护一张表，不要写死一个值——这是这个检测里唯一一个必须按数据集标定的参数。
