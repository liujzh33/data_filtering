# black — 黑屏检测（重写版）

**文件**：`check_pipeline/black.py`
**入口**：`check_black(probes, params=None) → BlackReport`
**依赖**：`probe.py`（逐帧灰度）、`errors.py`（异常）、`intervals.py`（区间工具）

---

## 1. 它只回答一个问题

**画面是不是丢了信号。** 相机掉线、镜头被完全挡死、编码器吐黑帧。

它**不**回答"画面是不是偏暗"。暗但有内容的画面标注照样能用，那是曝光问题，不是黑屏。旧版把这两件事混成了一条规则，是它最大的问题。

---

## 2. 判据：又暗**又平**

```
黑帧  ⟺  gray_mean < mean_abs   AND   gray_std < std_abs
```

默认 `mean_abs = 18`、`std_abs = 8.0`。两个条件取**与**，不是或。

### 为什么标准差才是关键

黑屏的物理本质是**没有信号**，表现出来就是所有像素都一样、没有纹理。均值只能说明"暗"，说明不了"没内容"：

| | 平均亮度 | 标准差 |
|---|---|---|
| 真黑屏（相机掉线） | ≈ 0–5 | **≈ 0** |
| 暗场景（夜间、关灯） | 20–40 | 15–30 |
| 正常画面 | 100–150 | 40–60 |

暗场景和黑屏的均值可以很接近，但标准差差一个数量级。旧版**完全没用 std**——而 `gray_std` 其实早就算好了（`decode_gray_stats` 每帧都算，`ViewMetrics.gray_std` 字段就在那里，遮挡检测在用），黑屏检测一个字没提，白解码了。

### 为什么 `mean_abs` 是 18 不是 10

视频有两种取值范围：full range（0–255）和 limited range（Y 16–235）。**limited range 的黑电平正好是 16**，`color_range` 元数据缺失或转换没做对时，真黑屏的灰度均值就落在 16。

旧版 `abs_thr = 10` 完全抓不到这种黑屏。取 16 也不行——判据是严格小于，`16 < 16` 不成立（这个我是在测试里踩到的）。18 留 2 的余量。

配合 std 条件，把 mean 放宽到 18 不会带来误报：亮度 18 以下**且完全没纹理**的画面，只能是黑屏。

`mean_abs` 同时还是 black 和 occlusion 的**分界线**，见下一节。

### 和 occlusion 怎么分工

两个检测用**同一个** `std_abs` 判"画面平不平"，再按 `mean_abs` 分家：

| 测到的 | 归谁 | 含义 |
|---|---|---|
| `std < 8` 且 `mean < 18` | **black** | 又平又暗 = 丢了信号 |
| `std < 8` 且 `mean ≥ 18` | **occlusion** | 平但不暗 = 镜头被挡住/糊住 |
| `std ≥ 8` | 都不管 | 画面有内容 |

这样任何"平"的帧一定落进其中之一，**不留缝隙也不重叠**。

旧版是 black 用 `std_abs = 2`、occlusion 用 `8`，中间 2～8 且暗的帧两边都不管——是个漏检缝隙。这一版把两边统一到 8 后缝隙消失了，测试里穷举了 80 组 `(mean, std)` 组合验证过。

统一到 8 而不是 2，等于放宽了 black 的判据。会不会误报？`mean < 18` 意味着整个画面的动态范围只有 0–18，在这个范围里 `std = 8` 已经是相当大的起伏了——实际上不存在"亮度 18 以下、std 在 2～8 之间、还有可用内容"的画面。

## 3. 规则表：只有一条

| 规则码 | 触发条件 | 处理 |
|---|---|---|
| `BLACK_RUN_RATIO_EXCEEDED` | **任意一路**相机的**最长连续**黑帧段 ≥ `max_black_run_ratio`（默认 10%） | `drop_episode` |

没有删帧这条路。整个 black 检测的结果只有两种：这条 episode 留着，或者整条丢掉。

**为什么是"任意一路"而不是"所有路"**：视角数量各数据集/机型不一样（2 路、3 路、6 路都有），不能要求"所有相机同时黑"。只要有一路瞎了，这条数据就不完整——用它训练出来的策略在那个视角上学到的是噪声。

**分母是该路"有测量值的帧数"**，不是时间轴长度。读不出来的帧在 probe 里是 NaN，既不算黑也不算正常，直接不计入分母——那些是 corrupt 的事（而且 corrupt 现在只要有坏帧就丢整条，走不到 black 这一步）。

**只输出一条 finding，不按相机拆**。处理动作只有一个（丢整条），拆成多条会让下游还得去重。每一路的具体数字在 `evidence.per_view_over` 里，没超标的路也在 `evidence.per_view_longest_run_ratio` 里，对账不缺信息。

### 判据是最长连续段，不是黑帧总量

这一点决定了整条规则的性格：看的是**最长的那一段**黑屏占了 episode 多大比例，其他零散的黑帧完全不参与判定。

一条 1500 帧的 episode：

| 情况 | 黑帧总量 | 最长连续段 | 结果 |
|---|---|---|---|
| 散布 150 个孤立黑帧 | 10% | 1 帧（0.07%） | 不删 |
| 中间连续黑 150 帧 | 10% | 150 帧（10%） | **删整条** |

要拦的是相机掉线/断流这类"画面持续黑了一段时间"的故障，不是"零星黑帧的累计量"。黑帧总量仍然测出来放在 `measurements.per_view.black_ratio` 和 `evidence.per_view_black_ratio_total` 里，只是不参与判定。

`evidence.per_view_over[cam].longest_black_run_span` 直接给出最长那段的 `[start, end)`，命中之后可以直接跳过去看。

**已知取舍：闪烁型故障会被放过。** 一路相机每 10 帧闪一次黑、闪满全片，总量 10% 但最长连续段只有 1 帧，不触发。跑第一批之后对比 `black_ratio`（总量）和 `longest_black_run_ratio`（最长段）这两个数：如果有一批 episode 总量很高但最长段很小，说明闪烁型故障在这批数据里真实存在，需要另加一条按总量判的规则。

同时取消的还有：段长绝对值阈值、首尾边缘豁免、margin、"整路几乎全黑"和"所有相机同时黑"的独立规则。它们服务的都是"该删哪些帧"这个问题，而现在这个问题不存在了。

**副作用**：最长连续段不到 10% 的黑帧会**原样留在数据里**，不再被抠掉。旧版会把它们删掉（`BLACK_RUN_SHORT`）。如果这批数据对零星黑帧敏感，需要另外处理。

### 短 episode 要注意

阈值是比例，所以 episode 越短越容易触发。一条 200 帧的 episode，开头 20 帧的相机启停黑帧就是 10%，直接删掉——而这是正常现象。（改用最长连续段之后这个问题比按总量算时更突出：启停黑帧天然就是连续的。）

跑第一批之后建议看一下：被删的 episode 里，`longest_black_run_span` 是不是都紧贴 `[0, x)` 或 `[n-x, n)`。如果是，说明触发的主要是启停而不是真故障，这时候该调的是 `max_black_run_ratio`（业务上能容忍多少无效帧），而不是去动 `mean_abs`/`std_abs`（那会让判据本身失真）。

---

## 4. 参数：从 21 个降到 3 个

```python
DEFAULT_PARAMS = {
    "mean_abs": 18.0,            # 黑帧判据: 灰度均值上限 (也是与 occlusion 的分界线)
    "std_abs": 8.0,              # 黑帧判据: 灰度标准差上限 (与 occlusion 必须取同值)
    "max_black_run_ratio": 0.10, # 任一视角最长连续黑帧段达到它 -> 丢整条 (唯一的删除规则)
}
```

旧版是 `black_abs` + `black_rel_ratio` + 3×4 的时长分档表 + 4 个 margin + `edge_exempt_sec` = **21 个数字**，没有一个是从数据统计出来的（注释说是"从 archived 移植"）。参数之间的联合行为没人能推理，5.2 的倒挂就是这么来的。

现在三个参数各管一件事，可以单独解释、单独调：

- `mean_abs` / `std_abs` 决定**什么叫黑帧**（物理判据，和数据集无关）
- `max_black_run_ratio` 决定**容忍多长的一段**（业务决定，跟数据集质量和用途有关）

调参时应该只动第三个。前两个是物理量，改它们意味着你认为"黑"的定义变了。

`std_abs` 有个额外约束：**必须和 occlusion 的 `std_abs` 保持一致**，否则两个检测之间会出现缝隙或重叠（见 §2）。

传入未知参数名会直接报错（`UnsupportedDataError`），不会静默忽略。

---

## 5. 修掉了旧版的哪些问题

下面每一条都是在真实代码上跑出来的，不是推测。

### 5.1 判据两个方向都会错

按 `black.py` 旧公式实跑：

```
场景                                        旧版判黑   新版判黑   结论
A 亮度120, 40帧变暗到35 (std=22 有纹理)          40        0    误报已消除
B 全程 gray=30 均匀一片 (std=0.4)                0        0    不判黑 -> 归 occlusion
C 全程 gray=12 (几乎全黑)                        0     1000    漏报已修复
D 真黑屏 gray=2, 连续40帧                       40       40    一致
E 夜间基线25, 中间40帧真黑                        40       40    一致
F limited-range 黑电平=16                       0       40    漏报已修复
```

**A 是误报。** 画面只是变暗（关了灯、机械臂靠近镜头挡了大半），内容还在。但相对阈值 = p75 × 0.3 = 120 × 0.3 = 36，亮度 35 被判成"黑屏"删掉。这条规则实际检测的是"比自己平时暗"，不是"黑屏"。新判据看 std=22（纹理还在），不判黑。

**C 是漏报，而且是自参照基线的通病。** 相机整条 episode 亮度稳定在 12（对人眼已经是黑的）：

- 绝对阈值 10：`12 > 10`，不触发。这是个硬悬崖——亮度 9 判黑并可能删整条，12 完全放行
- 相对阈值：基线是**它自己**的 p75 = 12，阈值 3.6，也不触发

**异常如果是全程的，用自己的分位数当基线就永远发现不了。** 这和 corrupt 里"用首帧尺寸当基准"是同一类错误，但更隐蔽——corrupt 至少还输出了尺寸直方图，这里连痕迹都没有。

**F 是 limited range 的黑电平问题**，见 2.2。

### 5.2 四档表本身是倒挂的

三张表的设计意图是"同时黑的相机越多越严重"。把所有时长扫一遍：

```
帧数  时长     1路黑      2路黑      3路黑(应最重)   是否倒挂
  4  0.133s   moderate       mild       mild      <-- 倒挂
  6  0.200s   moderate   moderate       mild      <-- 倒挂
 10  0.333s     severe   moderate   moderate      <-- 倒挂
 14  0.467s     severe   moderate   moderate      <-- 倒挂

35 个时长里有 11 个出现"路数越多反而判得越轻"
```

在 0.13–0.47 秒区间，**3 路相机同时黑，判得比 1 路黑还轻**。原因是三张表的 `moderate` 阈值定成 0.13 / 0.17 / 0.33——路数越多阈值订得越松，方向反了。这不是调参问题，是 12 个数字之间根本没做过一致性检查。

新版没有这张表：每路相机独立算自己最长的那段黑、独立和 10% 比，判定里根本不出现"几路同时黑"这个量。多路同时黑的信息在 `evidence.per_view_longest_run_ratio` 里（几路的比值都高就是多路同时黑），不需要专门的档位去表达。

### 5.3 一帧之差跨一个档

```
连续 3帧 (0.100s) -> mild      → 删帧
连续 4帧 (0.133s) -> moderate  → 删段+margin
连续 29帧 (0.967s) -> severe   → 什么都不做
连续 30帧 (1.000s) -> critical → 删整条 episode
```

29 帧和 30 帧的差别决定了保留还是整条删掉，而这个差别在物理上不存在。旧版有 **12 个**这样的硬边界。

新版只剩 `max_black_run_ratio` **1 个**。边界本身消不掉（任何阈值都有边界），但降到 1 个之后它至少是可解释、可单独调的。新版量的同样是连续段长度，但它是**相对 episode 长度的比例**而不是绝对帧数：一条 300 帧的短 episode 和一条 3000 帧的长 episode，"黑了多久算严重"本来就不该用同一个绝对帧数去卡。

### 5.4 切段的粒度和分级的粒度不匹配

旧版段是按"**任意**一路黑"切的，级是按"**几路**黑"定的，两个粒度对不上：

```
场景: 帧100-200 三路全黑(系统断了) + 帧200-400 单路黑
   find_runs 切出 1 段 [100,400)  典型路数(中位)=1  -> 按单路档判
```

因为 100–400 全程满足"至少一路黑"，被切成一段；段内 300 帧里只有 100 帧是三路全黑，中位数是 1，于是整段按单路处理。**那 100 帧的系统级全黑在报告里消失了。**

新版不做跨相机切段，也不按"几路黑"分级：每路相机独立算自己最长的那段黑、独立和 10% 比。上面这个场景里三路的最长段分别是 300、100、100 帧，三个数字都完整地出现在 `evidence.per_view_longest_run_ratio` 里，没有任何一路的信息被另一路稀释掉。

`black_ranges` 仍然逐段输出（诊断用），所以"帧 100–200 三路都黑"这件事从三路各自的区间里一眼能看出来。

### 5.5 边缘豁免的判据是"碰到边缘"

```python
# 旧版 base.py
return start < edge_frames or end > n_frames - edge_frames
```

只要段的起点落在开头 0.5 秒内，整段就算"边缘"，不管它有多长：

```
段[10,1400) 长1390帧 -> is_edge=True
```

一个几乎覆盖整条 episode 的黑屏段，因为起点在第 10 帧被标成"相机启停的正常现象"。

新版**没有边缘豁免**这个概念了：判据只看最长连续段占 episode 的比例，和它从第几帧开始无关。`[10,1400)` 这种段占比接近 1，必然触发 `BLACK_RUN_RATIO_EXCEEDED` → `drop_episode`。

代价是首尾的相机启停黑帧也会计入占比，短 episode 上可能造成误删。这个风险和对策见 §3 末尾。

### 5.6 severe 档什么都不做

```python
"severe": "split_or_drop_episode",   # 代码注释: "severe 当前无自动清洗分支 (留人工)"
```

跑十万条 episode 没人会人工看，所以 severe **实际等于静默保留**。这和 corrupt 里的 `record_only` 是同一个病，已经一起去掉了。

### 5.7 帧号是错的（只要有任何一帧解码失败）

旧版 `metrics.decode_gray_stats`：

```python
for frame in frames:
    try:
        means.append(float(gray.mean()))   # 只在成功时 append
    except Exception:
        corrupt.append(idx)                # 失败时不 append
    idx += 1                               # 但计数器照常推进
```

失败的帧不进 `means`，帧号照常推进，于是**数组下标不等于帧号**：第 50 帧失败，`means[50]` 装的是第 51 帧的值，后面全部左移。black 输出的 `frame_range` 就是这个下标，下游拿去删 parquet 对应行——**删的是错行**。

新版由 `probe.py` 按 PTS 定位帧号，坏帧填 `NaN`，数组长度恒等于时间轴长度。实测：

```
25帧含5帧NaN -> n_valid=20 n_black=10 区间=[(15, 25)]   (帧号未左移)
```

### 5.8 帧率是编的

```python
fps = metrics.fps or 30.0
```

拿不到帧率就当 30fps。而旧版**所有**阈值都是秒——12 个分档阈值、4 个 margin、边缘豁免，全靠 fps 换算成帧。一个 10fps 的视频被当成 30fps，等于所有时长阈值收紧 3 倍：本来 1 秒才 critical，现在 0.33 秒就删整条。

新版拿不到 fps 直接抛 `UnsupportedDataError`。

### 5.9 exclude_mask 会让检测静默失效，还会串改档位

```python
black_mask &= ~metrics.exclude_mask
```

`base.py` 的 `issues_to_mask` 对**没有 frame_range 的 Issue 整列置 True**。旧版 corrupt 报"整路损坏"时正是这种 Issue，于是这路相机在 black 里被整列抹掉，**永远不会报黑屏**。

第二重影响更隐蔽：`n_black_cams = black_mask.sum(axis=1)` 是**减完之后**算的。某路被屏蔽，每帧的"同时黑路数"就少 1，本该查 3 路表的段变成查 2 路表——**corrupt 的一条 issue 会悄悄改变 black 对其他相机的分级结果**。这种跨检测耦合没人能预料，出了问题也无从追查。

新版没有 `exclude_mask`：坏帧在 probe 里就是 NaN，天然跳过。**没有任何检测能改写另一个检测的输入。**

### 5.10 检测做不了的时候返回"干净"

```python
if metrics.gray_mean is None or metrics.gray_mean.size == 0:
    return issues      # 空列表 = 没问题
```

灰度没解出来、相机全部解码失败、只跑了 corrupt 没解灰度——所有这些都返回空列表，在报告里等同于"检查过了，没黑屏"。新版一律报错，见第 6 节。

---

## 6. 报错 vs 筛选

沿用 corrupt 定下的三类划分（定义在 `errors.py`，全部检测共用）：

```
数据坏了       → finding，走规则表（删帧 / 删 episode）
运行环境不对   → 抛 PipelineEnvironmentError，不产出任何 finding
数据格式不支持 → 抛 UnsupportedDataError，不产出任何 finding
```

后两类**不捕获**，直接往上抛。调用方必须把这条 episode 记为**"检测失败"**——既不能计入"已检查且干净"，也不能判成"有黑屏"。

本模块的报错项（全部已测）：

| 情况 | 异常 |
|---|---|
| `numpy` 不可用 | `PipelineEnvironmentError` |
| probe 没带逐帧灰度（`gray_mean`/`gray_std` 为 None） | `UnsupportedDataError` |
| `gray_mean` 与 `gray_std` 长度不一致 | `UnsupportedDataError` |
| 拿不到 fps（fps=0） | `UnsupportedDataError` |
| 各相机 fps 不一致 | `UnsupportedDataError` |
| 各相机时间轴长度不一致 | `UnsupportedDataError` |
| 一路可用的相机都没有 | `UnsupportedDataError` |
| 未提供任何 probe | `UnsupportedDataError` |
| 传入未知参数名 | `UnsupportedDataError` |

**"时间轴长度不一致"为什么要报错而不是截断到最短**：旧版 `metrics.compute_view_metrics` 用 `n = min(len(a) for a in per_mean)` 悄悄截断，长的那一路的尾部就此消失，而且不留任何记录。这种情况应该由 corrupt 的 `VIEW_FRAME_COUNT_MISMATCH` 先拦下；走到 black 还不一致，说明上游没处理，此时逐帧比对多视角没有意义。

---

## 7. 输出格式

```json
{
  "check": "black",
  "has_black": true,
  "drop_episode": true,
  "triggered_rules": ["BLACK_RUN_RATIO_EXCEEDED"],
  "findings": [
    {
      "rule": "BLACK_RUN_RATIO_EXCEEDED",
      "description": "存在相机的最长连续黑帧段达到阈值占比 (画面持续又暗又平 = 丢了信号)",
      "disposition": "drop_episode",
      "views": ["cam_high"],
      "frame_ranges": [[600, 780]],
      "n_frames_affected": 180,
      "evidence": {
        "criterion": "longest_black_run / n_valid",
        "threshold": 0.10,
        "mean_abs": 18.0,
        "std_abs": 8.0,
        "n_views_checked": 2,
        "n_views_over": 1,
        "per_view_over": {
          "cam_high": {
            "longest_black_run": 180,
            "longest_black_run_ratio": 0.12,
            "longest_black_run_span": [600, 780],
            "n_valid": 1500,
            "n_black_total": 180, "black_ratio_total": 0.12,
            "n_segments": 1,
            "min_gray_mean": 2.1, "min_gray_std": 0.4
          }
        },
        "per_view_longest_run_ratio": {"cam_high": 0.12, "cam_left": 0.0},
        "per_view_black_ratio_total": {"cam_high": 0.12, "cam_left": 0.0}
      }
    }
  ],
  "measurements": {
    "n_frames": 1500,
    "fps": 30.0,
    "params": { "...": "本次实际生效的 3 个参数" },
    "per_view": {
      "cam_high": {
        "n_frames": 1500, "n_valid": 1500,
        "n_black": 180, "black_ratio": 0.12,
        "n_flat_not_dark": 0, "flat_not_dark_ratio": 0.0,
        "n_segments": 1,
        "longest_black_run": 180,
        "longest_black_run_ratio": 0.12,
        "longest_black_run_span": [600, 780],
        "black_ranges": [[600, 780]],
        "gray_mean_percentiles": {"p1": 2.1, "p5": 2.3, "p25": 119.2, "p50": 120.0, "p75": 120.8, "p95": 122.4},
        "gray_std_percentiles":  {"p1": 0.4, "p5": 0.5, "p25": 44.8, "p50": 45.0, "p75": 45.3, "p95": 46.0},
        "min_gray_mean": 2.1, "min_gray_std": 0.4
      }
    }
  }
}
```

`findings[].frame_ranges` 现在是**诊断信息**——黑在哪里、黑了几段——不是"要删哪些帧"。整条 episode 都要丢，没有"删哪些"这回事。留着它是为了能一眼看出黑帧是集中在一段还是散在全程，这两种情况的排查方向完全不同。真正触发判定的那一段单独放在 `longest_black_run_span`。

`black_ratio`（总量）和 `longest_black_run_ratio`（最长段）这两个数放在一起看信息量最大：两者接近说明黑帧集中成一段（相机断流），差很多说明是闪烁型故障——后者现在不会触发删除，见 3.1 的取舍说明。

`per_view` 里没超标的相机也全都列出来。只列超标的那路，就没法回答"是这一路特别差，还是全都差、只是差一点"。

`n_flat_not_dark` 是"平但不暗"的帧数——这些归 occlusion 管，black 只报数便于对账。某路 `black_ratio` 接近 0 而 `flat_not_dark_ratio` 很高，说明该看 occlusion 的报告。

### `measurements` 为什么重要

`std_abs = 8.0` 和 `mean_abs = 18` 是按物理含义拍的，**没有你们数据的支撑**。`per_view` 里输出了每路的 mean / std 分位数，跑一批真实数据之后就能看着分布把阈值定准，而不是继续沿用来路不明的常数。

建议第一批跑完先看两个数：正常帧的 `gray_std_percentiles.p5`（应该远大于 8），和黑帧的 `min_gray_std`（应该接近 0）。两者之间的间隔越大，说明 `std_abs = 8.0` 越安全。

`black_ratio` 的全库分布也值得看一眼：如果大量 episode 卡在 0.10 附近，说明阈值正好切在数据的密集区，往哪边挪一点结果都会大变——这种情况下这个阈值本身就不是个好的切分点。

---

## 8. 上一版的缝隙已经补上

上一版这里记着一个已知漏洞：**"画面是均匀的一片但不暗"**（镜头被浅色物体挡死、白平衡失控整片发白）black 不管（不够暗），occlusion 也抓不到（它用自参照基线，一路相机全程 std≈0.3 时基线就是 0.3，相对判据找不出任何异常）——这个 case 从两个检测中间漏了过去。

重写 occlusion 时补上了：

- occlusion 去掉了相对阈值，改用纯绝对判据
- 两边统一 `std_abs = 8.0`，按 `mean_abs = 18` 分家

现在任何"平"的帧一定归 black 或 occlusion 其中之一。black 仍然把这类帧数出来（`per_view[cam].n_flat_not_dark`），方便和 occlusion 的报告对账。

**还剩下的真缝隙**：整帧 std 抓不到**半遮挡**（手挡住镜头一半）。这不是 black 的问题，详见 `occlusion.md` §2。

---

## 9. 用法

```python
from check_pipeline.probe import probe_video
from check_pipeline.black import check_black
from check_pipeline.errors import CheckError, PipelineEnvironmentError

videos = {"cam_high": open("cam_high.mp4","rb").read(),
          "cam_left": open("cam_left.mp4","rb").read()}

try:
    # 整条流水线只解码这一次; corrupt / black / occlusion / stillness 共用同一批 probe
    probes = {cam: probe_video(b, cam) for cam, b in videos.items()}
    report = check_black(probes)
except CheckError as e:
    # 检测没能完成。单独归档, 不要当成"干净", 也不要当成"有黑屏"。
    log_check_failed(episode_id, type(e).__name__, str(e))
else:
    if report.drop_episode:
        ...   # 丢弃整条 episode (唯一的处理结果)
    json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
```

**跑全量时的要求**（同 corrupt）：`PipelineEnvironmentError` 的数量应该是 0，不是 0 就说明环境有问题，该批结果全部作废重跑。`UnsupportedDataError` 如果集中出现在某一类数据上，先看清楚再决定是放宽支持范围还是剔除那批数据——**不要为了让流程跑通而把报错改回降级**。

---

## 10. 模块划分

重写 black 时把公共部分拆了出来，后续 5 个检测直接沿用：

| 文件 | 职责 |
|---|---|
| `errors.py` | 异常体系（报错 vs 筛选的三类划分） |
| `intervals.py` | 帧区间工具（逐段切分 / 合并 / 交集 / 最长段） |
| `probe.py` | 单路视频的**唯一一次**解码：PTS 定位帧号、坏帧 NaN 占位、像素格式白名单、逐帧 mean/std/相邻帧像素差 |
| `corrupt.py` | 文件完整性规则 |
| `black.py` | 黑屏规则（平 + 暗） |
| `occlusion.py` | 遮挡规则（平 + 不暗） |
| `stillness.py` | 静止规则（state 不动 + 画面不变） |

`probe.py` 一次遍历同时产出尺寸、坏帧位置和逐帧灰度。旧版 corrupt 走 `decode_frame_stats`（rgb24）、black/occlusion 走 `decode_gray_stats`（gray），**同一个视频被完整解码两遍**。

灰度是从 rgb24 数组按 BT.601 权重算的，不再单独调一次 `to_ndarray(format="gray")`：少一次 swscale 转换，且结果不受源视频 `color_range` 元数据缺失的影响。

---

## 编排与删除率

四个检测的执行顺序、共用解码、以及"总删除率是四者并集而非相加"的统计，见 `pipeline.md`。

注意 stillness 的删除门槛是 **30%** 而不是这里的 10%——静止帧不是坏数据，容忍度不同，理由见 `stillness.md` §2。
