# occlusion — 遮挡检测（重写版）

**文件**：`check_pipeline/occlusion.py`
**入口**：`check_occlusion(probes, params=None) → OcclusionReport`
**依赖**：`probe.py`（逐帧灰度）、`errors.py`（异常）、`intervals.py`（区间工具）

---

## 1. 它只回答一个问题

**镜头前面是不是被东西挡住了。** 手、物料、盖子贴住镜头，或者镜头糊了。画面还有亮度，但没有内容。

---

## 2. 判据：又平**又不暗**

```
遮挡帧  ⟺  gray_std < std_abs   AND   gray_mean >= mean_abs
```

默认 `std_abs = 8.0`、`mean_abs = 18.0`，两个值都**必须**和 black 保持一致。

### 和 black 怎么分工

两个检测用同一个 `std_abs` 判"画面平不平"，再按 `mean_abs` 分家：

| 测到的 | 归谁 | 含义 |
|---|---|---|
| `std < 8` 且 `mean < 18` | **black** | 又平又暗 = 丢了信号 |
| `std < 8` 且 `mean ≥ 18` | **occlusion** | 平但不暗 = 被挡住了 |
| `std ≥ 8` | 都不管 | 画面有内容 |

任何"平"的帧一定落进其中之一，不留缝隙也不重叠。测试里穷举了 80 组 `(mean, std)` 组合验证。

上一版 black 的文档里记着一个已知漏洞——"均匀的一片但不暗"（镜头被浅色物体挡死、白平衡失控整片发白）black 不管、occlusion 也抓不到——就是这次补上的。

### 一个必须说清楚的局限：抓不到半遮挡

旧版 README 和 docstring 都写着：

> "手挡住镜头一半，均值可能正常（不触发黑屏），但 std 骤降（触发遮挡）"

**这句话是错的，而且方向反了。**

手挡住一半时，画面变成**双峰分布**：一半是暗的手，一半是亮的场景。标准差会**升高**，不是骤降。

```
正常场景     像素散布在 60–180        std ≈ 30
挡住一半     一半 ≈ 40, 一半 ≈ 120     std ≈ 40   ← 更大了
```

所以整帧 std 不但抓不到半遮挡，半遮挡还会让 std **更远离**阈值。

整帧 std 能抓到的只有"**整个画面**变成均匀一片"，也就是镜头被完全糊住/贴死。这一版的文档如实这么写。

要真正抓半遮挡，得换个信号——比如把画面切成 N×N 格，看有没有**整块**坍缩成纯色。那是另一个检测，现在没有。与其在文档里声称覆盖了、让人误以为这个场景有人管，不如明确写出来：**半遮挡目前是漏检的。**

---

## 3. 规则表：只有一条

| 规则码 | 触发条件 | 处理 |
|---|---|---|
| `OCCLUSION_RUN_RATIO_EXCEEDED` | **任意一路**相机的**最长连续**遮挡段 ≥ `max_occlusion_run_ratio`（默认 10%） | `drop_episode` |

没有删帧这条路，和 black 完全对称。

**为什么是"任意一路"**：视角数量各数据集/机型不一样，不能要求所有相机同时被挡。只要有一路被挡瞎，这条数据就不完整。

**分母是该路"有测量值的帧数"**，读不出来的帧（NaN）不计入。

**判据是最长连续段，不是遮挡帧总量**——和 black 口径完全一致。看的是最长的那一段遮挡占了 episode 多大比例，其他零散的低纹理帧不参与判定。

这对 occlusion 尤其重要：画面正好扫过一块纯色区域（白墙、桌面、地板）时会产生零星的低纹理帧，这是正常运镜，不是遮挡。按总量算，这些帧会累加起来把正常数据判死；按最长连续段算，只有"镜头被持续挡住"才会触发。

遮挡帧总量仍然测出来放在 `measurements.per_view.occlusion_ratio` 和 `evidence.per_view_occlusion_ratio_total` 里，只是不参与判定。`longest_occluded_run_span` 给出触发判定的那一段的位置。

同时取消的还有段长绝对值阈值、边缘豁免和 margin。

---

## 4. 参数：3 个

```python
DEFAULT_PARAMS = {
    "std_abs": 8.0,                   # 遮挡判据: 灰度标准差上限 (与 black 必须取同值)
    "mean_abs": 18.0,                 # 与 black 的分界线 (与 black 必须取同值)
    "max_occlusion_run_ratio": 0.10,  # 任一视角最长连续遮挡段达到它 -> 丢整条
}
```

旧版是 `occlusion_abs_std` + `occlusion_rel_ratio` + 3×4 分档表 + 4 个 margin + `edge_exempt_sec` = 20 个数字。

`std_abs = 8` 的含义：在 0–255 的灰度上，像素几乎全挤在一个值附近。参考量级：

| | 灰度标准差 |
|---|---|
| 正常室内场景 | 25–50 |
| 低纹理场景（白墙、素色桌面） | 10–20 |
| 真遮挡（手贴住镜头） | 0–5 |

8 落在低纹理场景和真遮挡之间。这个值和旧版一样，是唯一一个从旧版直接沿用的参数——它本身没问题，出问题的是和它取 AND 的那个相对阈值。

传入未知参数名会直接报错，不会静默忽略。

---

## 5. 修掉了旧版的哪些问题

### 5.1 自参照基线 + AND，整路遮挡必然漏检 ⚠️ 最严重

旧判据：

```python
occl_mask[:, v] = (col < abs_thr) & (col < rel_thr)
#                                     rel_thr = median(该视角所有帧的 std) * 0.15
```

一路相机**全程**被挡死，std 恒为 2。它自己的中位数就是 2，相对阈值 = 2 × 0.15 = **0.3**。判据变成：

```
(2 < 8) and (2 < 0.3)  =  True and False  =  False
```

**一帧都不报。**

这是自参照基线的通病——"异常如果是全程的，基线就把它当成了正常"。black 也有这个病，但**black 用的是 OR，绝对阈值还能兜底；occlusion 用 AND，基线一失效整个判据就废了**。

而"整路被挡死"恰恰是最该抓的情况：它意味着这个视角的数据从头到尾都是废的。

基线用的还是**中位数**，所以遮挡只要超过一半时间就开始失效，不需要到 100%。

**新版怎么解决**：直接删掉相对阈值。判"画面有没有内容"用绝对标准就够了——std 是有物理意义的量，不需要和这台相机自己比。

回归测试里有这个场景：全程 `std = 2` 的相机，新版判出 100% 遮挡 → `drop_episode`。

### 5.2 文档描述的能力做不到

见 §2 末尾。旧版声称能抓半遮挡，实际上半遮挡会让 std 升高、更远离阈值。

这个问题比漏检本身更麻烦：**它让人以为这个场景有人管了**，于是不会再去建别的检测。

### 5.3 四档表倒挂

和 black 同款。旧版按"同时遮挡视角数"选分档表：

```python
"occlusion_1cam_sec": {"mild": 0.10, "moderate": 0.20, "severe": 0.50, "critical": 1.0},
"occlusion_2cam_sec": {"mild": 0.10, "moderate": 0.17, "severe": 0.50, "critical": 1.0},
"occlusion_3cam_sec": {"mild": 0.07, "moderate": 0.33, "severe": 0.50, "critical": 1.0},
```

时长落在 0.20–0.33 秒之间时：1 路遮挡判 `moderate`，3 路同时遮挡判 `mild`。**路数越多判得越轻**，和设计意图完全相反。0.17–0.20 秒区间也有类似倒挂。

这 12 个数字之间没做过一致性检查。

**新版怎么解决**：没有分档表。每路独立算自己最长的那段遮挡、独立和 10% 比。多路同时遮挡的信息在 `evidence.per_view_longest_run_ratio` 里（几路的比值都高就是多路同时遮挡），不需要专门的档位。

### 5.4 severe 档什么都不做

```python
"severe": "split_or_drop_episode",
```

和 black 一样，没有对应的自动处理分支，实际等于静默保留。

### 5.5 exclude_mask 会整列抹掉一路相机

```python
if metrics.exclude_mask is not None and metrics.exclude_mask.shape == occl_mask.shape:
    occl_mask &= ~metrics.exclude_mask
```

前序检测报出一条没有帧号的 Issue，掩码整列（甚至整块）被置 True，这一路相机就**永远不再报遮挡**。而报告上看不出任何异常。

**新版怎么解决**：没有 `exclude_mask`。坏帧在 `probe.py` 里就是 NaN，天然不计入分子也不计入分母。

### 5.6 拿不到帧率就假装 30

```python
fps = metrics.fps or 30.0
```

10fps 的视频，所有秒级阈值等于收紧 3 倍，而且从输出里完全看不出来。

**新版怎么解决**：拿不到 fps 直接报 `UnsupportedDataError`。（这一版的判据只看帧数比例、不需要秒换算，但 fps 拿不到说明容器元数据有问题，而且报告里"最长遮挡段折合多少秒"这个最直观的量就没法算了。）

### 5.7 没有数据就返回"干净"

```python
if metrics.gray_std is None or metrics.gray_std.size == 0:
    return issues   # 空列表 = 没有遮挡
```

测都没测就报干净。**新版怎么解决**：报 `UnsupportedDataError`。

---

## 6. 报错 vs 筛选

| 情况 | 处理 |
|---|---|
| probe 没带 `gray_mean` / `gray_std` | `UnsupportedDataError` |
| 拿不到 fps，或各相机 fps 不一致 | `UnsupportedDataError` |
| 各相机时间轴长度不等 | `UnsupportedDataError`（corrupt 该先拦下） |
| 一路可用的相机都没有 | `UnsupportedDataError`（不能返回"干净"） |
| 传入未知参数名 | `UnsupportedDataError` |
| numpy 不可用 | `PipelineEnvironmentError` |

这些异常表示**检测没能完成**，不代表数据好或坏。调用方必须单独归档，既不能计入"干净"，也不能判成"有遮挡"。详见 `errors.py`。

---

## 7. 输出格式

```json
{
  "check": "occlusion",
  "has_occlusion": true,
  "drop_episode": true,
  "triggered_rules": ["OCCLUSION_RUN_RATIO_EXCEEDED"],
  "findings": [
    {
      "rule": "OCCLUSION_RUN_RATIO_EXCEEDED",
      "description": "存在相机的最长连续遮挡段达到阈值占比 (画面持续平但不暗 = 镜头被挡住/糊住)",
      "disposition": "drop_episode",
      "views": ["cam_wrist"],
      "frame_ranges": [[300, 520]],
      "n_frames_affected": 220,
      "evidence": {
        "criterion": "longest_occluded_run / n_valid",
        "threshold": 0.10,
        "std_abs": 8.0,
        "mean_abs": 18.0,
        "n_views_checked": 3,
        "n_views_over": 1,
        "per_view_over": {
          "cam_wrist": {
            "longest_occluded_run": 220,
            "longest_occluded_run_ratio": 0.146667,
            "longest_occluded_run_span": [300, 520],
            "n_valid": 1500,
            "n_occluded_total": 220, "occlusion_ratio_total": 0.146667,
            "n_segments": 1, "min_gray_std": 1.8
          }
        },
        "per_view_longest_run_ratio": {"cam_high": 0.0, "cam_left": 0.0, "cam_wrist": 0.146667},
        "per_view_occlusion_ratio_total": {"cam_high": 0.0, "cam_left": 0.0, "cam_wrist": 0.146667}
      }
    }
  ],
  "measurements": {
    "n_frames": 1500,
    "fps": 30.0,
    "params": { "...": "本次实际生效的 3 个参数" },
    "views": {
      "cam_wrist": {
        "n_frames": 1500, "n_valid": 1500,
        "n_occluded": 220, "occlusion_ratio": 0.146667,
        "occluded_ranges": [[300, 520]],
        "longest_occluded_run": 220,
        "longest_occluded_run_ratio": 0.146667,
        "longest_occluded_run_span": [300, 520],
        "min_gray_std": 1.8,
        "gray_std_percentiles": {"p1": 1.8, "p50": 42.0, "p99": 51.3},
        "n_flat_and_dark": 0
      }
    }
  }
}
```

`findings[].frame_ranges` 是**诊断信息**（遮挡在哪里、几段），不是"要删哪些帧"。

`n_flat_and_dark` 是"又平又暗"的帧数——这些归 black 管，occlusion 只报数便于对账。它和 black 报告里的 `n_black` 应该对得上。

`gray_std_percentiles` 用来事后校准 `std_abs`：正常帧的 `p50` 应该远大于 8，被挡的帧 `min_gray_std` 应该接近 0。两者间隔越大，8 这个阈值越安全。如果某个数据集的 `p1` 就在 8 附近，说明这批数据的场景纹理本来就很低（大片纯色背景），需要单独调。

---

## 8. 用法

```python
from check_pipeline.probe import probe_video
from check_pipeline.black import check_black
from check_pipeline.occlusion import check_occlusion
from check_pipeline.errors import CheckError

videos = {"cam_high": open("cam_high.mp4","rb").read(),
          "cam_wrist": open("cam_wrist.mp4","rb").read()}

try:
    # 整条流水线只解码这一次, 四个检测共用同一批 probe
    probes = {cam: probe_video(b, cam) for cam, b in videos.items()}
    black_report = check_black(probes)
    occl_report = check_occlusion(probes)
except CheckError as e:
    log_check_failed(episode_id, type(e).__name__, str(e))
else:
    if black_report.drop_episode or occl_report.drop_episode:
        ...   # 丢弃整条 episode
    json.dump(occl_report.to_dict(), f, ensure_ascii=False, indent=2)
```

**改 `std_abs` 或 `mean_abs` 时，black 和 occlusion 必须同时改**，否则两者之间会出现缝隙（某些"平"的帧两边都不管）或重叠（同一批帧被报两次）。

---

## 编排与删除率

四个检测的执行顺序、共用解码、以及"总删除率是四者并集而非相加"的统计，见 `pipeline.md`。

注意 stillness 的删除门槛是 **30%** 而不是这里的 10%——静止帧不是坏数据，容忍度不同，理由见 `stillness.md` §2。
