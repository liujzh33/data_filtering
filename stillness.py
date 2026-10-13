"""stillness — 静止帧检测 (重写版)。

只回答一个问题: **这条 episode 里有多少帧是"什么都没发生"的**。机器人没动, 画面也没变 ——
这些帧对策略学习没有信息增益, 典型出现在 episode 首尾 (等待开始 / 做完后停着)。

判据: 静止帧 ⟺ 机器人没动 **且** 这一路的画面没变

    state 侧:  ‖state[t] - state[t-1]‖ <  state_diff_thresh
    image 侧:  frame_diff[t]           <  image_diff_thresh      (逐路各判各的)

两侧取**与**。只有 state 静止可能是机器人停着但场景里有人在动 (信息还在);
只有画面静止可能是机器人在动但动作幅度小到画面看不出 —— 都不算冗余。

--------------------------------------------------------------------------
和前三个检测的性质差异 (重要)
--------------------------------------------------------------------------

corrupt / black / occlusion 找的是**坏数据**, stillness 找的是**没用的数据**。静止帧
本身没有任何错误, 它只是不提供训练信号。

门槛统一取 0.10, 和 black / occlusion 一致 —— 三个检测口径完全对称, 结果可横向比较。

这个门槛对 stillness 是偏严的, 必须知道它的实际含义:

  一条 50 秒 (1500 帧 @30fps) 的 episode, 只要有**连续 5 秒**没动, 整条就删掉。

而开头等指令、结尾等复位停 3-5 秒在真实采集里很常见, 不是数据质量问题。所以这条线
大概率会误删一批正常数据。跑第一批之后**必须**看 longest_still_run_span:
如果被删的 episode 里那一段普遍紧贴 [0, x) 或 [n-x, n), 说明拦的是采集停顿而不是
异常停滞, 这时候该放宽 max_still_run_ratio (业务决定), 而不是去动判据阈值。

判据本身 (什么叫静止) 不能为了少删而放宽 —— 那是物理问题, 放宽只会让判定失真。
能调的是**容忍度** (能接受多长的一段停滞), 那才是业务决定。两者不能混为一谈。

统计口径是**最长连续段**, 不是总量: 看的是最长的那一段静止占了 episode 多大比例,
与其他零散的静止帧无关。这一点和 black / occlusion 完全一致, 三个检测口径统一。

对 stillness 来说这个口径尤其重要, 因为它正好对上 PDF 的 "remove PROLONGED static
segments" —— 要删的是**持续**的停滞。机器人操作过程中本来就有大量瞬时微停顿
(减速接近目标点、抓取瞬间、换手过渡), 这些帧单独看都满足"state 没动且画面没变",
但它们是任务的一部分。按总量算会把它们累加起来, 把正常数据判死。

--------------------------------------------------------------------------
旧版 (new/stillness_frame.py) 的问题
--------------------------------------------------------------------------

1. **`still_views` 计算把相机下标当帧下标用 (真 bug)**

       n_still_cams = np.zeros(n_frames)        # 一维: 每帧有几路静止
       still_views = [cameras[v] for v in range(len(cameras))
                      if v < n_still_cams[s:e].shape[0] and n_still_cams[s:e][v].any()]

   `n_still_cams[s:e][v]` 取的是**段内第 v 帧**的静止相机数, 却被当成"第 v 个相机是否静止"
   用来决定 views 列表。索引语义完全错位, 输出的 views 是随机的。

2. **`typical_cams` 和 `k` 算完从没用过 (死代码)**

       k = min(max(typical_cams, 1), 3)         # 算了
       sev = classify_duration(dur_sec, levels)  # 用的是 levels, 不是 levels[k]

   分级表只有 `still_1cam_sec` 一档, 按视角数分级的意图没实现。

3. **同一份 motion 算了两遍** (L154-164 一遍, L192-202 又一模一样地算一遍)。

4. **image 静止判据用错了量**

       motion = |Δgray_mean| + |Δgray_std| < 0.1

   两个问题。一是量纲混合: 亮度差和纹理差直接相加没有物理意义。二是更根本的 ——
   **整帧灰度均值判不出画面有没有变**。机械臂在画面里移动时它占的像素少, 整帧均值几乎不变,
   于是"机械臂在动"会被判成图像静止。

   旧 docstring 自己也自相矛盾: 说"各平台 motion q1=0.003~0.006, 0.1 约在 q25-q50 之间",
   又说"只抓接近完全相同的帧"。0.1 落在 q25-q50 意味着 **25%~50% 的帧**都会被判图像静止,
   跟"接近完全相同"差得远。

   新版改用 probe.frame_diff = 相邻帧的**平均绝对像素差**, 直接回答"画面变了多少灰度级"。

5. **`image_mask = is_static_per_view.all(axis=1)`: 所有相机都静止才算**

   三路相机里有一路噪点大一点, 整条 episode 就永远判不出静止。而需求要的恰恰相反 ——
   任意一路满足占比就删。新版逐路独立统计。

6. **`state_diff_thresh` 默认 1e-4 在跨平台时没有意义**

   旧 docstring 自己列了实测: AgiBotG1 q1=0.000171, AgileX q1=0.0, GalaxeaR1Pro q1=0.061。
   GalaxeaR1Pro 连最静止的 1% 帧差值都有 0.061, 是默认阈值的 **600 倍** —— 这个平台上
   state 侧永远判不出静止, 整个检测静默失效; AgileX q1=0.0 则至少 1% 的帧完全不变。
   同一个数字在不同平台上行为天差地别。

   state 的单位取决于机型 (弧度 / 米 / 归一化值), 我们不知道就不能猜。**新版把
   state_diff_thresh 设为必传参数**, 不给默认值, 不传直接报错。这就是"有错就报错,
   不要硬筛选"用在阈值上的样子。

7. **exclude_mask 一路坏就让整个检测失效 (比 black/occlusion 更严重)**

       joint_mask &= ~metrics.exclude_mask[:n_frames].any(axis=1)

   用的是 `any(axis=1)`: 只要**任意一路**在某帧被前序排除, 这一帧就不判静止。corrupt 报了
   一路整路坏 (整列 True) 的话, `any` 全 True, joint_mask 被整个清零, 静止检测一帧都不报,
   而输出看起来就是"这条 episode 没有静止" —— 静默失效。新版没有 exclude_mask,
   坏帧在 probe 里是 NaN, 天然不计入。

8. **state 与视频时间轴长度不等时静默截断**。`N = min(...)` 直接截短, 长的那一段就没了;
   反过来 state 比视频长时 `state_mask[1:1+len(is_static)] = is_static` 还会直接抛
   ValueError。新版长度不等就报 UnsupportedDataError (corrupt 的
   ACTION_VIDEO_LENGTH_MISMATCH 本来就该先拦下)。

9. **`fps or 30.0`**、**只有导数通道时静默退化成纯 image**、**无 state 无 image 时返回空列表
   当"干净"** —— 与 black/occlusion 同款的静默降级。新版一律报错。

--------------------------------------------------------------------------
10. **夹爪豁免做反了**

   PDF 要求 "task-critical key frames such as gripper closure are explicitly preserved"
   —— 夹爪闭合的那几帧不能删。旧版的做法是算 state 差分时把夹爪维度**剔除**, 于是
   "手臂不动、只有夹爪在合"的帧因为剩余维度没变化被判成 state 静止, 恰好把 PDF 点名
   要保护的帧标成了可删。方向反了。

   要让夹爪帧不被删, 夹爪一动就必须判为**不静止**, 也就是夹爪维度必须**参与**差分。
   新版保留全部位置维度, 不做剔除。

--------------------------------------------------------------------------
规则表 (rule 字段的取值)。**只有一条, 结果只有丢整条**:

  STILLNESS_RUN_RATIO_EXCEEDED   任意一路相机的**最长连续**静止段
                                 >= max_still_run_ratio (默认 10%) -> drop_episode

**报错 vs 筛选**: 见 errors.py。本模块的报错项:
  - 不传 state_diff_thresh              -> UnsupportedDataError (不猜机型单位)
  - probe 没带 frame_diff               -> UnsupportedDataError
  - state 行数与视频帧数不等             -> UnsupportedDataError (corrupt 该先拦下)
  - state 里没有可用的值信号列           -> UnsupportedDataError (不退化成纯 image)
  - 拿不到 fps / fps 不一致 / 长度不等    -> UnsupportedDataError
  - numpy / pandas 不可用                -> PipelineEnvironmentError

用法:
    from check_pipeline.probe import probe_video
    from check_pipeline.stillness import check_stillness, extract_state_matrix

    probes = {cam: probe_video(b, cam) for cam, b in videos.items()}
    state, cols = extract_state_matrix(parquet_bytes)
    report = check_stillness(probes, state,
                             params={"state_diff_thresh": 1e-3})  # 按机型标定, 必传
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import PipelineEnvironmentError, UnsupportedDataError
from .intervals import (indices_to_ranges, longest_range, longest_range_span,
                        merge_ranges, ranges_length)
from .probe import VideoProbe

logger = logging.getLogger("check_pipeline.stillness")


# =====================================================================
# 参数
# =====================================================================
DEFAULT_PARAMS = {
    "state_diff_thresh": None,
    #   **必传, 没有默认值。** ‖state[t]-state[t-1]‖ 小于它算 state 静止。
    #   state 的单位取决于机型 (弧度 / 米 / 归一化值), 实测同一组数据在不同平台上
    #   最静止 1% 帧的差值能差 600 倍, 定一个通用默认值等于在猜。
    #   标定方法: 拿该机型一批 episode 算 ‖Δstate‖ 的分布, 取真静止段所在的那个峰的上沿。
    "image_diff_thresh": 2.0,
    #   相邻帧平均绝对像素差 (灰度级) 小于它算画面静止。
    #   有默认值是因为它的单位是**固定的** —— 0-255 灰度, 与机型无关。
    #
    #   为什么是 2.0 而不是更小: 两帧各带 σ 的独立传感器噪声时, 帧间平均绝对差的期望是
    #   σ·√2·√(2/π) ≈ 1.13σ。取 1.0 只能覆盖 σ<0.9 的相机 —— 真静止的画面会因为噪声
    #   略超阈值而判不出静止, 是**漏检**方向的错。2.0 覆盖到 σ≈1.8, 涵盖大多数真实相机。
    #   上界由正常操作决定: 机械臂占画面 10-30%、移动区域像素变化 20-80 灰度级,
    #   全帧平均落在 5-15。2.0 离它有 2 倍以上余量。
    #
    #   **一个重要的耦合**: H.264 对静止场景用 skip 宏块, 解出来的相邻帧逐位相同,
    #   frame_diff 精确为 0。所以常规 IPPP 编码的数据上这个阈值取 1 还是 3 都无所谓;
    #   但 all-intra 编码的数据会把噪声全暴露出来, 行为完全不同。
    #   换数据集时先看 measurements 里的 frame_diff_percentiles.p1 再决定要不要调。
    "max_still_run_ratio": 0.10,
    #   **唯一的删除规则**: 任意一路相机的**最长连续**静止段达到这个占比 -> 丢弃整条。
    #
    #   是最长连续段, 不是静止帧总量。这一点和 black / occlusion 口径一致, 而且正好
    #   对上 PDF 的 "remove PROLONGED static segments" —— 要删的是**持续**的停滞,
    #   不是操作过程中零星的瞬时微停顿 (减速接近、抓取瞬间、换手过渡)。
    #   按总量算的话那些微停顿会累加起来, 把正常数据判死。
    #
    #   取值与 black / occlusion 一致 (0.10), 三个检测口径完全对称。
    #
    #   **但要清楚它对 stillness 意味着什么**: 那两个检测找的是**坏数据** (画面丢了
    #   信号 / 被挡住), 静止帧本身**没有任何错误**, 只是不提供训练信号。0.10 换算过来
    #   是"一条 50 秒的 episode 只要连续 5 秒没动就整条删掉", 而开头等指令、结尾等复位
    #   停 3-5 秒在真实采集里很常见。
    #   这条线大概率会误删一批正常数据 —— 跑第一批之后看 longest_still_run_span
    #   是不是普遍紧贴首尾, 是的话就该往上调这个值。
}

# state 列名里含这些词的是**导数**通道 (速度/力矩/加速度), 不是位置量。
# 差分一个速度信号得到的是加加速度, 和"机器人有没有动"不是一回事, 所以排除。
DERIVATIVE_HINTS = ("velocity", "effort", "force", "torque", "imu", "accel", "jerk", "vel_")

# 注: 没有 GRIPPER_HINTS。夹爪维度**不排除**, 要参与差分才能保护夹爪关键帧,
# 见模块 docstring 第 10 条。

# parquet 里认定为 state 列的列名 (小写包含即命中)。
STATE_COLUMN_HINTS = ("state", "observation.state", "qpos")


# =====================================================================
# 规则表
# =====================================================================
DROP_EPISODE = "drop_episode"

RULES: Dict[str, Tuple[str, str]] = {
    "STILLNESS_RUN_RATIO_EXCEEDED": (
        "存在相机的最长连续静止段达到阈值占比 (机器人持续没动且画面没变 = 冗余停滞)",
        DROP_EPISODE),
}


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class Finding:
    rule: str
    description: str
    disposition: str                            # 当前恒为 drop_episode (字段保留以统一 JSON 结构)
    views: List[str] = field(default_factory=list)
    frame_ranges: List[Tuple[int, int]] = field(default_factory=list)
    n_frames_affected: int = 0
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "description": self.description,
            "disposition": self.disposition,
            "views": self.views,
            "frame_ranges": [list(r) for r in self.frame_ranges],
            "n_frames_affected": self.n_frames_affected,
            "evidence": self.evidence,
        }


@dataclass
class ViewStillness:
    """一路相机的静止测量结果 (已经和 state 侧取过与)。"""
    cam: str = ""
    n_frames: int = 0
    n_valid: int = 0                   # state 和 image 两侧都有测量值的帧数
    n_still: int = 0
    still_ratio: float = 0.0
    still_ranges: List[Tuple[int, int]] = field(default_factory=list)
    longest_still_run: int = 0                       # 最长连续静止段的长度 (判定用)
    longest_still_run_ratio: float = 0.0             # 它占 n_valid 的比例 (判定用)
    longest_still_run_span: Optional[Tuple[int, int]] = None   # 它的位置 (排查用)
    n_image_static: int = 0            # 只看画面 (未与 state 取与) 的静止帧数
    frame_diff_percentiles: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cam": self.cam,
            "n_frames": self.n_frames,
            "n_valid": self.n_valid,
            "n_still": self.n_still,
            "still_ratio": round(self.still_ratio, 6),
            "still_ranges": [list(r) for r in self.still_ranges],
            "longest_still_run": self.longest_still_run,
            "longest_still_run_ratio": round(self.longest_still_run_ratio, 6),
            "longest_still_run_span": (list(self.longest_still_run_span)
                                       if self.longest_still_run_span else None),
            "n_image_static": self.n_image_static,
            "frame_diff_percentiles": self.frame_diff_percentiles,
        }


@dataclass
class StillnessReport:
    findings: List[Finding] = field(default_factory=list)
    views: Dict[str, ViewStillness] = field(default_factory=dict)
    n_frames: int = 0
    fps: float = 0.0
    n_state_static: int = 0            # state 侧静止帧数 (全局, 与相机无关)
    state_columns: List[str] = field(default_factory=list)
    state_dims_used: int = 0
    params: dict = field(default_factory=dict)

    @property
    def has_stillness(self) -> bool:
        return bool(self.findings)

    @property
    def drop_episode(self) -> bool:
        return any(f.disposition == DROP_EPISODE for f in self.findings)

    @property
    def triggered_rules(self) -> List[str]:
        seen: List[str] = []
        for f in self.findings:
            if f.rule not in seen:
                seen.append(f.rule)
        return seen

    def to_dict(self) -> dict:
        return {
            "check": "stillness",
            "has_stillness": self.has_stillness,
            "drop_episode": self.drop_episode,
            "triggered_rules": self.triggered_rules,
            "findings": [f.to_dict() for f in self.findings],
            "measurements": {
                "n_frames": self.n_frames,
                "fps": round(self.fps, 3),
                "n_state_static": self.n_state_static,
                "state_columns": self.state_columns,
                "state_dims_used": self.state_dims_used,
                "params": self.params,
                "views": {c: v.to_dict() for c, v in sorted(self.views.items())},
            },
        }


# =====================================================================
# state 提取
# =====================================================================
def extract_state_matrix(parquet_bytes: Optional[bytes]) -> Tuple[Any, List[str]]:
    """从 lerobot parquet 里取出 state 矩阵 (T, D) 和参与的列名。

    不读 info.json: 按列名惯例识别 (小写包含 STATE_COLUMN_HINTS 之一)。
    导数通道 (速度/力矩/...) 排除, 只保留位置量。

    报错情况 (不降级):
      - pandas / numpy 不可用     -> PipelineEnvironmentError
      - parquet 为 None / 读不出   -> UnsupportedDataError
      - 认不出 state 列            -> UnsupportedDataError
      - state 列全是导数通道        -> UnsupportedDataError (不退化成纯 image 检测)
    """
    if parquet_bytes is None:
        raise UnsupportedDataError(
            "未提供 parquet 字节, 取不到 state 信号。静止检测必须 state 和 image 联合判定, "
            "只看画面会把'机器人在动但画面变化小'误判成静止")
    try:
        import numpy as np
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise PipelineEnvironmentError(f"numpy/pandas 不可用: {e}") from e

    try:
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        raise UnsupportedDataError(f"parquet 读取失败: {type(e).__name__}: {e}") from e

    state_cols = [c for c in df.columns
                  if any(h in str(c).lower() for h in STATE_COLUMN_HINTS)]
    if not state_cols:
        raise UnsupportedDataError(
            f"parquet 里认不出 state 列 (列名: {sorted(str(c) for c in df.columns)}; "
            f"识别规则: 列名小写包含 {list(STATE_COLUMN_HINTS)})")

    used_cols: List[str] = []
    blocks = []
    for c in state_cols:
        if any(h in str(c).lower() for h in DERIVATIVE_HINTS):
            continue
        # lerobot 的 state 列通常每格是一个定长向量, 也可能是标量列
        m = np.asarray(np.stack(df[c].to_numpy()) if df[c].dtype == object
                       else df[c].to_numpy(), dtype=np.float64)
        if m.ndim == 1:
            m = m.reshape(-1, 1)
        blocks.append(m)
        used_cols.append(str(c))

    if not blocks:
        raise UnsupportedDataError(
            f"state 列 {sorted(str(c) for c in state_cols)} 全都是导数通道 "
            f"(含 {list(DERIVATIVE_HINTS)} 之一), 没有位置量可用于判静止。"
            f"不退化成纯画面检测 —— 那会把'机器人在动但画面变化小'误判成静止")

    rows = {b.shape[0] for b in blocks}
    if len(rows) > 1:
        raise UnsupportedDataError(
            f"state 各列行数不一致: "
            f"{ {c: b.shape[0] for c, b in zip(used_cols, blocks)} }")
    return np.concatenate(blocks, axis=1), used_cols


# =====================================================================
# 测量
# =====================================================================
def _state_static(state: Any, thresh: float) -> Any:
    """(T, D) 的 state 矩阵 -> (T,) bool: 该帧相对前一帧是否静止。第 0 帧恒 False。

    **夹爪维度保留在差分里**, 不排除。PDF 的 "gripper closure explicitly preserved" 是
    "夹爪闭合这类关键帧不要删掉"的意思 —— 要达到这个效果, 夹爪一动就必须**判为不静止**,
    也就是夹爪维度必须**参与**差分。旧版反过来把夹爪维度从差分里剔除了, 结果是"手臂不动、
    只有夹爪在合"的那几帧被判成静止 —— 恰好把 PDF 点名要保护的帧标成了可删。
    """
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise PipelineEnvironmentError(f"numpy 不可用: {e}") from e

    T = int(state.shape[0])
    out = np.zeros(T, dtype=bool)
    if T < 2:
        return out
    d = np.linalg.norm(np.diff(state, axis=0), axis=1)   # (T-1,)
    # NaN 的 state (采集丢帧) 不算静止 —— 不知道就不判, 不往"可删"的方向猜
    out[1:] = (d < thresh) & ~np.isnan(d)
    return out


def measure_view(probe: VideoProbe, state_static: Any, image_diff_thresh: float) -> ViewStillness:
    """把一路相机的画面静止与 state 静止取与, 得到该路的静止帧。只测量, 不判定。"""
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise PipelineEnvironmentError(f"numpy 不可用: {e}") from e

    if probe.frame_diff is None:
        raise UnsupportedDataError(
            f"相机 {probe.cam!r} 的 probe 没有 frame_diff; "
            f"probe_video(..., collect_gray=True) 才能跑静止检测")

    fd = np.asarray(probe.frame_diff, dtype=np.float64)
    out = ViewStillness(cam=probe.cam or "", n_frames=int(fd.shape[0]))

    # frame_diff 在坏帧处、以及第 0 帧是 NaN。这些帧两侧都没有可信的"变了多少",
    # 既不算静止也不算运动, 直接不计入分母。
    valid = ~np.isnan(fd)
    out.n_valid = int(valid.sum())
    if out.n_valid == 0:
        return out

    image_static = valid & (fd < image_diff_thresh)
    out.n_image_static = int(image_static.sum())

    is_still = image_static & state_static[:out.n_frames]
    out.n_still = int(is_still.sum())
    out.still_ratio = out.n_still / out.n_valid
    out.still_ranges = indices_to_ranges(np.flatnonzero(is_still).tolist())
    out.longest_still_run = longest_range(out.still_ranges)
    out.longest_still_run_ratio = out.longest_still_run / out.n_valid
    out.longest_still_run_span = longest_range_span(out.still_ranges)

    v = fd[valid]
    out.frame_diff_percentiles = {
        "p1": float(np.percentile(v, 1)),
        "p50": float(np.percentile(v, 50)),
        "p99": float(np.percentile(v, 99)),
    }
    return out


# =====================================================================
# 主检测
# =====================================================================
def check_stillness(probes: Dict[str, VideoProbe],
                    state: Any,
                    params: Optional[dict] = None,
                    state_columns: Optional[Sequence[str]] = None,
                    ) -> StillnessReport:
    """对一条 episode 跑静止帧检测。

    **唯一的删除规则**: 任意一路相机的**最长连续**静止段 >= max_still_run_ratio
    (默认 10%) -> 丢弃整条 episode。判据是最长连续段, 不是静止帧总量。
    这条线对正常采集停顿的杀伤力, 见模块 docstring 开头。

    probes: {相机名: probe_video(...) 的结果}, 必须是 collect_gray=True 解出来的
        (需要 frame_diff)。可与 black/occlusion 复用同一批 probe。
    state: (T, D) 的 state 矩阵, 由 extract_state_matrix() 得到。T 必须等于视频帧数。
    params: 必须包含 state_diff_thresh (无默认值, 见 DEFAULT_PARAMS)。

    可能抛 CheckError, 表示**检测没能完成**; 调用方必须记为"检测失败"。
    """
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise PipelineEnvironmentError(f"numpy 不可用: {e}") from e

    p = dict(DEFAULT_PARAMS)
    if params:
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise UnsupportedDataError(
                f"未知参数 {sorted(unknown)}; 可用参数: {sorted(DEFAULT_PARAMS)}")
        p.update(params)

    if p["state_diff_thresh"] is None:
        raise UnsupportedDataError(
            "必须显式传入 state_diff_thresh。state 的单位取决于机型 (弧度/米/归一化值), "
            "实测同一批数据在不同平台上最静止 1% 帧的差值能差 600 倍, 用通用默认值等于在猜: "
            "阈值偏小则整个检测静默失效 (一帧都判不出静止), 偏大则把正常的慢速操作判成静止")
    if not (p["state_diff_thresh"] > 0):
        raise UnsupportedDataError(f"state_diff_thresh 必须为正数, 收到 {p['state_diff_thresh']}")

    if not probes:
        raise UnsupportedDataError("未提供任何相机的 probe, 无法检测")

    usable = {cam: pr for cam, pr in probes.items() if pr.usable}
    if not usable:
        raise UnsupportedDataError(
            f"{len(probes)} 路相机没有一路可用 (缺失/打不开/零可读帧), 无法判断静止。"
            f"这条 episode 应由 corrupt 处理, 不能当成'没有静止'")

    fps_values = {round(pr.fps, 3) for pr in usable.values()}
    if not fps_values or 0.0 in fps_values:
        raise UnsupportedDataError(f"存在拿不到帧率的相机 (fps={sorted(fps_values)})")
    if len(fps_values) > 1:
        raise UnsupportedDataError(
            f"各相机帧率不一致: "
            f"{ {c: round(pr.fps, 3) for c, pr in usable.items()} }, 时间轴无法对齐")

    lengths = {c: pr.n_frames for c, pr in usable.items()}
    if len(set(lengths.values())) > 1:
        raise UnsupportedDataError(
            f"各相机时间轴长度不一致: {lengths}。应由 corrupt 先判 VIEW_FRAME_COUNT_MISMATCH")
    n_frames = next(iter(lengths.values()))

    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] == 0:
        raise UnsupportedDataError(
            f"state 必须是 (T, D) 且 D>0 的矩阵, 收到 shape={tuple(state.shape)}")
    if state.shape[0] != n_frames:
        # 不做截断对齐: 截断会悄悄把长的那一端丢掉, 而"哪一端是对的"我们并不知道。
        raise UnsupportedDataError(
            f"state 行数 {state.shape[0]} != 视频帧数 {n_frames}。"
            f"应由 corrupt 先判 ACTION_VIDEO_LENGTH_MISMATCH; stillness 不做截断对齐")

    report = StillnessReport(n_frames=n_frames, fps=next(iter(fps_values)),
                             state_columns=list(state_columns or []),
                             state_dims_used=int(state.shape[1]),
                             params={k: v for k, v in p.items()})

    # ---- 1) state 侧静止 (全局, 与相机无关) ----
    state_static = _state_static(state, p["state_diff_thresh"])
    report.n_state_static = int(state_static.sum())

    # ---- 2) 逐路把画面静止与 state 静止取与 ----
    for cam, pr in usable.items():
        report.views[cam] = measure_view(pr, state_static, p["image_diff_thresh"])

    # ---- 3) 唯一的规则: 任意一路静止占比达标 -> 丢整条 ----
    over = {c: v for c, v in report.views.items()
            if v.n_valid > 0 and v.longest_still_run_ratio >= p["max_still_run_ratio"]}
    if over:
        report.findings.append(_finding(
            "STILLNESS_RUN_RATIO_EXCEEDED", views=sorted(over),
            frame_ranges=merge_ranges(
                [r for v in over.values() for r in v.still_ranges]),
            evidence={
                "criterion": "longest_still_run / n_valid",
                "threshold": p["max_still_run_ratio"],
                "state_diff_thresh": p["state_diff_thresh"],
                "image_diff_thresh": p["image_diff_thresh"],
                "n_state_static": report.n_state_static,
                "n_views_checked": len(report.views),
                "n_views_over": len(over),
                "per_view_over": {
                    c: {"longest_still_run": v.longest_still_run,
                        "longest_still_run_ratio": round(v.longest_still_run_ratio, 6),
                        "longest_still_run_span": (list(v.longest_still_run_span)
                                                   if v.longest_still_run_span else None),
                        "n_valid": v.n_valid,
                        "n_still_total": v.n_still,
                        "still_ratio_total": round(v.still_ratio, 6),
                        "n_segments": len(v.still_ranges),
                        "n_image_static": v.n_image_static}
                    for c, v in sorted(over.items())},
                "per_view_longest_run_ratio": {
                    c: round(v.longest_still_run_ratio, 6)
                    for c, v in sorted(report.views.items())},
                # 总量只做参考, 不参与判定
                "per_view_still_ratio_total": {
                    c: round(v.still_ratio, 6)
                    for c, v in sorted(report.views.items())}}))

    return report


def _finding(rule: str, views: Optional[List[str]] = None,
             frame_ranges: Optional[List[Tuple[int, int]]] = None,
             evidence: Optional[dict] = None) -> Finding:
    desc, disposition = RULES[rule]
    fr = merge_ranges(frame_ranges or [])
    return Finding(rule=rule, description=desc, disposition=disposition,
                   views=list(views or []), frame_ranges=fr,
                   n_frames_affected=ranges_length(fr), evidence=evidence or {})
