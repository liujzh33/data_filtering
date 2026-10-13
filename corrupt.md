# corrupt — episode 损坏检测（重写版）

> **模块划分**：写 black 时把公共部分拆了出来。异常体系在 `errors.py`，帧区间工具在 `intervals.py`，视频解码在 `probe.py`（corrupt 和 black 共用同一批 probe，只解码一次）。本文件只做规则判定，不碰解码。`corrupt.py` 仍然 re-export 了 `probe_video` / `merge_ranges` 等符号，旧的 import 路径不受影响。

**代码**：`check_pipeline/corrupt.py`
**入口**：`check_corrupt(videos: {相机名: mp4字节}, parquet_bytes) -> CorruptReport`

---

## 1. 它只回答一个问题

**这条 episode 的原始文件本身是否完整可读。**

不判断画面内容好不好（黑屏、遮挡、静止是后续检测的事），也不判断动作数值合不合理。它是流水线的第一道关卡：文件本身都不可信，后面看什么都是白看。

---

## 2. 三条设计原则

### 2.1 不读 info.json，一切现场测量

旧版把 `info.json` 当作真值，拿它声明的相机数量、分辨率、总帧数去和实际文件比对，对不上就报错。但**声明值本身就可能是错的**——info.json 写 480×848、实际视频 800×1280，这种情况下报出来的"分辨率不一致"根本不知道是谁错了，人还得再去查一遍。

新版所有数字都从 episode 自身测量：

| 要用的信息 | 来源 |
|---|---|
| 有几个相机、叫什么 | 调用方传进来的 `videos` 字典（从目录里扫出来的实际文件） |
| 每路视频多少帧 | 解码时按 PTS 数出来 |
| 画面尺寸 | 解码时逐帧统计，取出现帧数最多的那个 |
| 动作数量 | parquet 行数 |
| 哪些列是动作列 | parquet 列名里含 `action`（lerobot 惯例），不看 info.json 的 `action_cols` |

结论只在**实际文件之间**比对（视频 vs 视频、视频 vs parquet），不和任何声明值比。

### 2.2 不用严重度分级，直接写明触发了哪条规则

旧版输出的是 `severity: "severe"` 这样的等级，规则和等级是多对一的——看到 severe 不知道是解码失败、尺寸抖动、还是某个列有 NaN，还得回去翻代码。

新版每条命中输出四样东西：

- `rule`：规则码，例如 `FRAME_DECODE_FAILED`
- `description`：这条规则的中文说明，直接写在 JSON 里，不用查文档
- `disposition`：怎么处理。corrupt 的规则现在**全部**是 `drop_episode`，这个字段保留是为了和 black / occlusion / stillness 的 JSON 结构统一
- `evidence`：**触发时的实测数字**，例如 `{"n_action_rows": 1490, "video_frame_count": 1498, "diff": -8}`

`disposition` 不是严重度阶梯，是最终动作。不需要下游再去查"severe 对应删段还是删整条"这张表。

### 2.3 坏帧位置必须准

见下面第 4 节，这是重写的主要动机之一。

### 2.4 处理不了就报错，不猜默认值继续跑

结果分成三类，**后两类绝不混进第一类**：

| 情况 | 输出 |
|---|---|
| 数据坏了 | finding，走规则表（删帧 / 删 episode） |
| 运行环境不对 | 抛 `PipelineEnvironmentError`，不产出任何 finding |
| 数据格式不支持 | 抛 `UnsupportedDataError`，不产出任何 finding |

后两类都继承自 `CorruptCheckError`，`check_corrupt` **不捕获**，直接往上抛。调用方必须把这条 episode 记为**"检测失败"**——既不能计入"已检查且干净"，也不能判成"数据损坏"。

为什么必须这样，看旧版的实际输出：

```json
{
  "rule": "VIDEO_UNDECODABLE",
  "disposition": "drop_episode",
  "evidence": {"cam": "cam_high", "error": "pyav 不可用: No module named 'av'"}
}
```

pyav 没装，被判成了"视频损坏，删整条"。跑全量的话**一个环境问题足以删光整个数据集**，而且报告上每条 episode 都写着"数据损坏"，事后翻不出真实原因。

任何"猜个默认值继续跑"的降级都有同类风险：结论看上去正常，但已经不可信，而且**不可信这件事本身也丢失了**。所以下列情况一律报错：

| 情况 | 异常 |
|---|---|
| pyav / pandas 不可用 | `PipelineEnvironmentError` |
| 容器没给 fps 或 time_base | `UnsupportedDataError` |
| 有帧拿不到 PTS，或 PTS 换算出的帧号非严格递增 | `UnsupportedDataError` |
| 源像素格式不在白名单，或中途变化 | `UnsupportedDataError` |
| parquet 字节没传（`parquet_bytes=None`） | `UnsupportedDataError` |
| parquet 里认不出动作列 | `UnsupportedDataError` |
| `videos` 为空 | `UnsupportedDataError` |

注意**任一路相机触发即中止整条 episode 的检测**，不会退回"其余相机的部分结果"。这是刻意的：结论要么整条可信，要么没有结论。

反过来，这几种**不报错**，因为它们是数据坏了，属于检测的正常产出：视频打不开（`VIDEO_UNDECODABLE`）、parquet 读不出来（`PARQUET_UNREADABLE`）、视频文件缺失（`VIDEO_FILE_MISSING`）。

#### 白名单为什么要收紧

`frame.to_ndarray(format="rgb24")` 是**强制**转换：源视频哪怕是 `gray`（真灰度），也会被转成三个相同通道后当彩色放行，检测端完全看不出来。所以在转换**之前**先看源格式：

```python
SUPPORTED_PIXEL_FORMATS = frozenset({
    "yuv420p", "yuvj420p", "yuv422p", "yuvj422p",
    "yuv444p", "yuvj444p", "yuv420p10le",
    "nv12", "nv21", "rgb24", "bgr24", "rgba", "bgra",
})
```

先收紧到常见彩色格式，跑过实际数据、确认还有哪些格式之后再逐步放宽。实测到的格式会写进 `measurements.per_video[cam].pix_fmt`。

#### PTS 为什么不能退化

旧版拿不到 PTS 时会退化成顺序计数，只在 `index_source` 字段标个 `"sequential"`。问题是**退化之后照样输出帧区间**，下游拿这些帧号去删 parquet 对应行会**删错行**——而且删完之后动作数和视频帧数依然对得上，事后无法发现。一个只影响"精度"的降级，实际后果是静默的数据错位。现在 `index_source` 字段整个删掉了，因为只剩 PTS 一种可能。

---

## 3. 规则表

| 规则码 | 触发条件 | 处理 |
|---|---|---|
| `VIDEO_FILE_MISSING` | 给了相机名但没有视频字节（文件缺失） | `drop_episode` |
| `VIDEO_UNDECODABLE` | 视频文件打不开，或容器里没有视频流 | `drop_episode` |
| `VIDEO_NO_READABLE_FRAME` | 能打开但一帧都读不出来 | `drop_episode` |
| `FRAME_DECODE_FAILED` | **存在**读不出来的帧（哪怕只有 1 帧） | `drop_episode` |
| `FRAME_SIZE_CHANGED` | 画面 H×W 中途变化 | `drop_episode` |
| `VIEW_FRAME_COUNT_MISMATCH` | 多个视角之间视频帧数不一致 | `drop_episode` |
| `ACTION_VIDEO_LENGTH_MISMATCH` | parquet 动作数量 ≠ 视频帧数 | `drop_episode` |
| `PARQUET_UNREADABLE` | parquet 读不出来，取不到动作数量 | `drop_episode` |

**全部规则都是 `drop_episode`。corrupt 没有删帧这条路。**

`VIDEO_FILE_MISSING` 单独成一条而不是并进 `VIDEO_UNDECODABLE`：文件不存在和文件损坏是两回事，前者多半是采集或传输环节漏了，后者是文件本身的问题，混在一起会掩盖前者。

"parquet 里找不到动作列"曾经是一条 `record_only` 规则，现在改成抛 `UnsupportedDataError`——核对做不了却在报告上显示"干净"，正是 2.4 要杜绝的情况。

**规则之间的关系**：一路视频整个坏掉（前两条）之后，这一路就不再参与后面的坏帧统计和帧数比对了——否则一个 0 帧的相机会连带触发"多视角帧数不一致"，同一个故障被报好几遍。

---

## 3.1 为什么一帧坏也要丢整条

上一版是分档的：零星短段删帧，最长连续段 ≥ 30 帧才丢整条。现在取消了分档，**只要有任何一帧读不出来就丢整条**。

### 分档为什么站不住

删帧的做法是：把坏帧从视频和 parquet 里同时抠掉，剩下的首尾相接。

抠掉一长段之后断口两侧在时间上不连续——第 500 到 545 帧全坏，删完第 499 帧直接接第 546 帧，中间跳过 1.5 秒，**机械臂在轨迹上"瞬移"了**。这段假跳变拿去训练相当于教模型一个物理上不存在的动作，而且后面的 `sudden_change` 会把它当成真实突变报出来，制造一个无法解释的误报。这部分旧的论证没有问题。

站不住的是"零星单帧就安全"这个推论。三个理由：

**一、零星断口会累积。** 单看一个 1 帧的断口确实平滑，但一条 episode 里散布 200 个这样的断口，等于在整条轨迹上洒了 200 个微小的时间跳变。累计效果和一个大断口没有本质区别，只是更难被发现——`sudden_change` 每个都抓不到，因为每个都不够大。

**二、阈值制造了物理上不存在的边界。** 29 帧保留、30 帧删除，而这两条数据的实际质量没有任何差别。

**三、读不出来的帧说明文件本身有问题。** 解码失败意味着这个 mp4 的写入或传输环节出过错。既然如此，"其余部分一定是好的"这个假设就没有依据——我们只知道解码器在那几帧上报错了，不知道它在别处有没有静默地解出错误内容。与其部分挽救，不如整条剔除。

### 代价

**删除率会上升。** 一条 1500 帧的 episode 只要坏 1 帧就整条丢弃。

这个代价是否可接受取决于坏帧在数据集里的分布。跑第一批时看 `measurements.per_video[cam].n_missing` 的分布：

- 如果坏帧集中在少数几条 episode 上（大部分 episode 的 `n_missing = 0`），那么整条丢弃的成本很低，这个规则是划算的。
- 如果大量 episode 都有 1～2 个坏帧，说明问题出在采集或转码的**共性环节**上，该修的是那个环节，而不是靠调检测阈值把它们放过去。

坏帧位置仍然逐段精确输出（见 §6），只是它现在是**诊断信息**——告诉你坏在哪里、坏了几段——不再是"要删哪些帧"。

---

## 3.2 尺寸基准取"多数派"，不是首帧

判断画面尺寸有没有中途变化，需要一个"正常尺寸"作基准。**基准取的是出现帧数最多的那个尺寸**，不是第一个成功解码的帧。

如果拿首帧当基准，遇到"只有首帧异常"的视频（某些编码器会吐出一个尺寸不对的首帧）结论会整个反过来：

| | 首帧作基准 | 多数派作基准 |
|---|---|---|
| 判定的正常尺寸 | 240×320（错） | 480×640（对） |
| 报告的异常帧数 | 1499 | 1 |
| 报告的异常区间 | `[[1, 1500]]` | `[[0, 1]]` |
| 最终处理 | drop_episode | drop_episode |

**最终处理是一样的，但诊断信息完全相反。** 报告会说"这路相机是 240×320、有 1499 帧尺寸不符"，而事实是"这路是 480×640、只有第 0 帧不符"。人拿这份报告排查会被带偏；如果后续想加修复逻辑（把异常帧 resize 回正常尺寸），按错误基准修会把 1499 帧好数据全毁掉。

`evidence` 里输出完整直方图，各尺寸占多少帧一目了然：

```json
"dominant_size": [480, 640],
"size_histogram": {"480x640": 1499, "240x320": 1},
"n_distinct_sizes": 2,
"n_frames_changed": 1
```

并列时（比如正好一半一半）取先出现的那个，结果是确定的，不会因为运行环境不同而变。

---

## 4. 关键实现：帧号怎么定位

这是新旧版差别最大的地方，值得单独说。

### 旧版的做法

```python
for packet in container.demux(stream):
    try:
        frames = packet.decode()
    except Exception:
        corrupt.append(idx)
        idx += 1          # 假设这个坏包本来只装 1 帧
        continue
```

mp4 内部不是一帧一帧独立存的，而是打成一个个 **packet（数据包）**，一个包解开可能出来 1 帧、多帧、或 0 帧。旧版遇到解码失败的包，只把计数器 `idx` 加 1，也就是**假设这个坏包本来只含 1 帧**。

如果坏包本来含 5 帧，从这里往后所有帧的编号就整体偏小 4，而且偏移随坏包数累积。注释里写了"（近似）"，但这个近似的帧号被下游拿去**精确删帧**了。

### 新版的做法

每一帧解出来时都带一个 **PTS（显示时间戳）**，它记录的是这一帧在时间轴上的位置，和解码是否成功无关。用它直接换算真实帧号：

```
帧号 = round((pts - start_time) * time_base * fps)
```

坏包不参与计数，它造成的空缺会在帧号序列里自然显现为一个"洞"。然后：

```
缺帧集合 = {0 .. 时间轴长度-1} - {成功读出的帧号}
```

这一个式子同时覆盖三种情况：转换失败的帧、坏包造成的中间空洞、尾部截断（容器声明 1500 帧但只读到 1000 帧）。

**没有兜底**：容器没写 PTS、帧率未知、time_base 缺失、或 PTS 换算出的帧号非严格递增（可变帧率 / B 帧重排），一律抛 `UnsupportedDataError`。理由见 2.4——退化成顺序计数会让下游删错 parquet 行，且事后查不出来。

---

## 5. 解决了旧版的哪些问题

### 问题 1：一个小问题会让后面 6 个检测全部静默失效 ⚠️ 最严重

旧版流水线有个 `exclude_mask`（帧 × 相机的格子板），前一个检测命中的格子画 ×，后面的检测跳过画 × 的格子。翻译规则是：

```python
# base.py
else:  # 没有 frame_range
    cols = [...] if it.views else list(range(n_views))
    for c in cols:
        mask[:, c] = True
```

**没写帧号 → 该相机整列画 ×；连相机都没写 → 整块板涂满 ×。**

而旧版 corrupt 的第 3、4 段（视角集合、长度一致性、action/state 列检查）产出的 Issue **全都既没有 `frame_range` 也没有 `views`**。于是「视频 1500 帧 != 声明 1501 帧」这种 off-by-one，会把整块掩码刷成 True，后面黑屏、遮挡、静止、极值、突变、趋势对齐全部报不出东西。

最麻烦的是它**是静默的**——报告上这条 episode 只有一条"帧数对不上"，看起来干干净净。用旧版跑出来的检出率统计因此不可信。

**新版怎么解决**：`exclude_mask` 这个机制整个删掉了，连带 `to_frame_delete_mask()` 也不再需要——corrupt 只有 `drop_episode` 一种结果，没有"删几帧然后继续跑后面的检测"这条路。坏帧对下游的影响改由 `probe.py` 表达：读不出来的帧在逐帧数组里是 `NaN`，black / occlusion / stillness 各自把 NaN 排除在分母外，天然跳过，不需要任何一个检测去"通知"另一个。

### 问题 2：两个孤立坏帧删掉整条 episode

旧版：

```python
if corrupt:
    lo = min(corrupt)        # 5
    hi = max(corrupt) + 1    # 1051
    delete_range=(lo, hi)
```

第 5 帧和第 1050 帧各坏一帧 → 删除区间 `[5, 1051)` → **因为 2 个坏帧删掉 1046 帧**。

（新版这两个坏帧会让整条 episode 被丢弃，见 §3.1。但报告里必须如实写出"坏了 2 帧，分别在第 5 和第 1050 帧"，而不是"坏了 1046 帧"——否则没法判断这批数据的真实质量，也没法回答"该不该放宽规则"。）

代码库里本来就有 `find_runs` 在干逐段拆分的事，黑屏和遮挡都在用，只有 corrupt 没用。

**新版怎么解决**：`indices_to_ranges()` 把帧号压成连续段，逐段输出。

```
输入 [5, 1050]      -> [(5, 6), (1050, 1051)]      删 2 帧
输入 [5, 6, 7, 1050] -> [(5, 8), (1050, 1051)]      删 4 帧
```

### 问题 3：同一个故障被报两遍，两遍结论还打架

旧版 `n_readable` 不计入解码失败的帧，所以只要有帧解码失败，必然 `n_readable < n_declared`，于是：

- 1c「解码失败」→ severe → **删段**
- 1d「丢帧」→ 同时触发；失败超 10% 时给 critical → **删整条**

同一条 episode 上挂着两个互相矛盾的建议，下游按哪个执行没有定义。

**新版怎么解决**：这两件事本来就是同一件事——「时间轴上有帧读不出来」。合并成一条判定，`evidence` 里同时给出 `n_declared` / `n_readable` / `n_frames` / `n_missing` / `missing_ratio` / `longest_missing_run`。要删帧还是要丢整条，由**最长连续段**唯一决定（见 3.1），不存在两条规则给出矛盾建议的情况。

按坏帧**比例**升级的策略没有内置（`missing_ratio` 放在 `evidence` 里供调用方自行决定）——比例高但都是零星单帧的数据其实是能救的，见 3.1 的对比表。

### 问题 4：相机打不开会被报成"缺少相机"

旧版：某路 mp4 打不开 → 这个相机不进 `valid_cameras` → 第 3 段一比对发现声明 3 个、实际 2 个 → 又报一条「视角集合不完整：缺少相机」。

措辞是错的：文件在，只是解不开。「文件损坏」和「文件丢失」排查方向完全不同，被混成一句话。

**新版怎么解决**：不读 info.json，就没有"声明了但没找到"这个概念。相机名单以调用方传入的实际文件为准，打不开就是 `VIDEO_UNDECODABLE`，`evidence.error` 里带上原始异常文字（例如 `moov atom not found`）。

### 问题 5：拿"随便某一路"的帧数当基准

旧版：

```python
readable = [info.get("n_readable", 0) for info in di.values() if not info.get("error")]
vframes = readable[0]     # 字典插入序的第一个
```

上一行刚判定完"各路帧数不一致"，下一行还是拿第一个去和 parquet 比。假设 parquet 有 1499 行、相机分别是 1500 和 1498 帧：谁排第一，「信号 < 视频」的结论就翻转。

**新版怎么解决**：参考帧数取各视角 `min`（这个帧数下每个视角都有数据），并且把 `per_view_frame_count` 完整写进 `evidence`，不丢失任何一路的原始数字。

### 问题 6：视频被完整解码两遍

旧版 `decode_gray_stats`（转灰度）和 `decode_frame_stats`（转 RGB）各自完整遍历一次视频，而且组装结果时灰度那遍算出的 `corrupt_indices` 被直接覆盖丢弃。解码是整个流水线最贵的操作。

**新版怎么解决**：`probe_video()` 只解码一次，一趟同时拿到帧号、坏帧、尺寸直方图。

---

## 6. 输出格式

### 零星坏帧 → 丢整条

```json
{
  "check": "corrupt",
  "corrupt": true,
  "drop_episode": true,
  "triggered_rules": ["FRAME_DECODE_FAILED"],
  "findings": [
    {
      "rule": "FRAME_DECODE_FAILED",
      "description": "存在读取失败的帧 (解码报错, 或时间轴上存在丢包空洞)",
      "disposition": "drop_episode",
      "views": ["cam_high"],
      "frame_ranges": [[5, 6], [1050, 1051]],
      "n_frames_affected": 2,
      "evidence": {
        "cam": "cam_high",
        "n_declared": 1500,
        "n_readable": 1498,
        "n_frames": 1500,
        "n_missing": 2,
        "missing_ratio": 0.001333,
        "n_segments": 2,
        "longest_missing_run": 1,
        "packet_errors": 2,
        "first_error": "decode: InvalidDataError: Invalid data found"
      }
    }
  ],
  "measurements": {
    "video_frame_counts": {"cam_high": 1500, "cam_left": 1500},
    "reference_frame_count": 1500,
    "n_action_rows": 1500,
    "action_columns": ["action"],
    "per_video": { "...": "每路的完整 probe 结果" }
  }
}
```

关键点：

- **`frame_ranges` 是 `[[5,6],[1050,1051]]`，不是 `[[5,1051]]`。** 两个孤立坏帧就报两段。这是诊断信息，告诉你坏帧是散的还是集中的。
- **`n_frames_affected` 是 2，不是 1046。**
- **只坏了 2 帧也是 `drop_episode`。** 理由见 §3.1。
- `evidence.missing_ratio` 和 `longest_missing_run` 让你能判断这条 episode 坏得有多厉害。`missing_ratio = 0.0013` 且 `longest_run = 1` 说明是零星丢包；如果全库大量 episode 都长这样，该查的是转码环节。

### 多视角帧数不一致

```json
{
  "check": "corrupt",
  "corrupt": true,
  "drop_episode": true,
  "triggered_rules": ["VIEW_FRAME_COUNT_MISMATCH", "ACTION_VIDEO_LENGTH_MISMATCH"],
  "findings": [
    {
      "rule": "VIEW_FRAME_COUNT_MISMATCH",
      "description": "多个视角之间视频帧数不一致",
      "disposition": "drop_episode",
      "views": ["cam_high", "cam_left"],
      "frame_ranges": [],
      "n_frames_affected": 0,
      "evidence": {
        "per_view_frame_count": {"cam_high": 1500, "cam_left": 1498},
        "min": 1498, "max": 1500, "diff": 2
      }
    }
  ],
  "measurements": { "...": "同上" }
}
```

这类规则**没有 `frame_ranges`**——问题出在文件之间对不上，不属于任何具体的帧。字段留空而不是填一个假区间。

旧版这里正是那个最严重的 bug 的来源：没有帧号的 Issue 被塞进 `exclude_mask`，掩码被整块刷成 True，后面 6 个检测全部静默失效（见 §5 问题 1）。新版没有 `exclude_mask` 这个东西了。

### 比对的是时间轴长度，不是可读帧数

`VIEW_FRAME_COUNT_MISMATCH` 和 `ACTION_VIDEO_LENGTH_MISMATCH` 比的都是 `n_frames`（时间轴长度），**不是** `n_readable`（成功读出的帧数）。

这个区分在上一版（有删帧路径时）很关键：一路相机坏了 2 帧、只读出 1498 帧，但它的时间轴仍然是 1500 帧，和另一路的 1500、动作的 1500 完全对得上，不该触发对齐规则。拿 `n_readable` 去比的话，2 个坏帧会连带触发"多视角不一致"，同一个故障报两遍。

现在坏帧本身就会丢整条，这个区分对最终结论的影响小了，但对**诊断**仍然重要：报告里应该显示"坏了 2 帧"，而不是"坏了 2 帧 + 多视角不一致 + 动作数量不齐"这三条互相矛盾的结论。

---

## 7. 相对旧版删掉的检查

以下是旧版有、新版按需求范围移除的，列在这里以便确认是有意为之：

| 旧检查 | 为什么移除 |
|---|---|
| 分辨率 vs info.json 声明（warning） | 依赖 info.json，声明值不可信 |
| 视角集合 vs info.json 声明（缺/多相机） | 同上；相机名单改以实际文件为准 |
| 视频帧数 vs `episodes.jsonl.length` | 同上；改为视频之间、视频与 parquet 之间互相比对 |
| action/state 列 ragged / NaN / 维度 vs 声明 | 不在本次 5 条需求范围内。NaN 和 ragged 是数值质量问题，建议移到信号侧的检测项，不放在"文件是否可读"这一关 |
| 通道数异常 / 通道数中途变化 | **这条规则在旧版从来没有触发过**。解码用的是 `frame.to_ndarray(format="rgb24")`，rgb24 是强制转换，不管源视频是 yuv420p 还是 gray，转出来一律是 `(H, W, 3)`，所以读到的通道数恒等于 3，判定条件永远为假。一个真正的灰度视频也会被转成三个相同通道后放行。既然 rgb24 转换已经把这个维度抹平、且转换后所有视频都是可用的 3 通道，通道数本身不影响可用性，这条规则直接移除而不是修复 |
| 「信号行数 > 视频帧数不报」的放行逻辑 | 新版一律要求相等。控制频率高于视频帧率的数据集，应该在读取层做重采样对齐后再进检测，而不是在这里放行——放行之后下游是按下标直接对齐的，比的是不同时刻的两个模态 |

---

## 8. 用法

```python
from check_pipeline.corrupt import check_corrupt
from check_pipeline.errors import CheckError, PipelineEnvironmentError, UnsupportedDataError

videos = {"cam_high": open("cam_high.mp4","rb").read(),
          "cam_left": open("cam_left.mp4","rb").read()}
parquet = open("episode_000000.parquet","rb").read()

try:
    report = check_corrupt(videos, parquet)
except CheckError as e:
    # 检测没能完成。单独归档, 不要当成"干净", 也不要当成"损坏"。
    log_check_failed(episode_id, type(e).__name__, str(e))
else:
    if report.drop_episode:
        ...   # 丢弃整条 episode (唯一的处理结果)
    json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
```

**跑全量时的一个要求**：`PipelineEnvironmentError` 的数量应该是 0，不是 0 就说明环境有问题，这一批结果全部作废重跑。`UnsupportedDataError` 如果集中出现在某一类数据上（比如某个采集批次全是 VFR 视频），先看清楚再决定是放宽白名单还是把这批数据剔除——不要为了让流程跑通而把报错改回降级。
