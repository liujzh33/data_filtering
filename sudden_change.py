"""突变检测 (stage 2)。

Qwen 的做法:

  "For each signal dimension we extract a smooth trend via cascaded median filtering and
   Savitzky-Golay smoothing, then compute three complementary deviation signals: the
   absolute residual between raw and smoothed trajectories, the second-order finite
   difference (acceleration), and the third-order finite difference (jerk). A frame is
   flagged when the residual exceeds a threshold AND either the acceleration or the jerk
   also exceeds its own threshold, which suppresses false positives from slow drift while
   retaining sensitivity to abrupt transients."

算法骨架照搬, 旧版 (new/sudden_change.py) 的问题全部出在**阈值怎么定**和**哪些通道
被排除在检测之外**。完整实测证据见 sudden_change.md, 这里只记结论。

--------------------------------------------------------------------------
一个阈值, 单位是"占该通道量程的比例"
--------------------------------------------------------------------------

旧版有 23 个参数, 其中 `k_residual/k_accel/k_jerk` 看着像阈值但**调了不起作用**:
θ = max(k × 1.4826 × MAD, scale_floor) 里 floor 实测是 MAD 项的 448 倍 (残差), max()
永远选 floor。真正生效的只有 scale_floor = max(3×noise_baseline, 0.02×量程, 1e-6),
而它是三项取大, 命中率偏高时你无法判断该动哪一项。

这一版把 R/A/J 归一到同一个物理量 —— "等效单次跳变幅度" —— 于是三者可以和**同一个**
阈值比较:

    R̂ = R / f_R    Â = A / f_A    Ĵ = J / f_J
    θ = jump_frac × range          (range 来自 profile, 见下)
    帧异常 = (R̂ > θ) 且 (Â > θ 或 Ĵ > θ)

f_* 是三个偏差信号对一次单位跳变的响应倍数。它们**不是写死的常数**, 而是运行时把一个
合成单位阶跃推过同一套滤波器测出来的 (_response_factors) —— 换了窗口/阶数自动跟着变,
不会出现常数和配置对不上的情况。

归一之后 θ 的含义很直接: **一次单帧位置跳变, 幅度达到通道量程的 jump_frac 就报**。
单帧尖峰 (跳出去又跳回来 = 两次跳变) 在 θ/2 处就会命中。

顺带修掉了旧版"阶跃比尖峰难检 3 倍"的问题 (实测尖峰门槛 3% 量程、阶跃 10% 量程):
根因是 SG 趋势线几帧内就追上阶跃, 残差被摊薄一半, 而 R 恰好是三者里唯一起约束作用的
那个。归一后三者对同一事件给出一致的读数, "导数确认"不再白白吃掉灵敏度。

--------------------------------------------------------------------------
量程来自 profile, 不是来自这条 episode
--------------------------------------------------------------------------

旧版用本条 episode 的 p99-p1 当尺度。后果是**同一个关节在不同 episode 上阈值不同**:
机器人动得少的那条 episode 量程小, 阈值跟着变紧, 更容易命中 —— 一个没人会注意到的
系统性偏差。极端情况下某维度整条没动, 量程≈0, 阈值≈0, 浮点抖动全部超阈。

关节的物理行程是**机型属性**, 不是 episode 属性。所以 range 由 profile 显式给出
(扫描阶段对全数据集算的 p1/p99), 缺失直接报错, 不退化成用本条 episode 估。

这一条同时让旧版那套"近恒定跳过"逻辑整个消失: 不再需要判断某维度是不是没动, 因为
尺度不再依赖它动没动。

--------------------------------------------------------------------------
不静默跳过任何通道
--------------------------------------------------------------------------

旧版有四道准入过滤, 命中即整条通道不参与检测、不产出 finding、输出里也看不出来:

  max_noise_ratio=0.15   median(R)/std(x) 超标跳过。实测一条**全程随机乱跳**的关节
                         (能想到的最严重的信号损坏) noise_ratio=0.169 -> 被跳过, 零
                         Issue。**越坏越检不出。** 旧注释说这类由 corrupt/stillness
                         负责, 但 corrupt 只看视频解码, stillness 要求帧间差异低
                         (乱跳的信号恰好相反) —— 整条流水线没有一环覆盖。
  近恒定跳过             std_pool = median(各通道 std) 把弧度和毫米混在一起取中位。
                         实测 3 个弧度关节 + 7 个毫米级通道时 std_pool=316, 判据变成
                         rng_p < 3.16, 于是**运动幅度小于 181° 的关节全被当近恒定排除**。
  唯一值占比 < 25%       意图是排除夹爪阶跃指令, 但实测 episode 前 80% 静止 (采集起始
                         等待) 的正常关节唯一值占比 0.20, 整条被排除。
  按名跳过               ("velocit","vel","imu",...) 子串匹配。"vel" 会误伤 level/travel;
                         effort/force/torque 被永久排除, 而 Qwen 原文里 InternData-A1
                         的突变正是物理碰撞造成的。

这一版一条都不保留。**哪些维度参与检测由 profile 显式声明**, 与 dataset_profile.json
里 SAA 用的是同一份文件, 但范围更宽: SAA 只要 state/action 共有的关节, 突变检测可以
包含只在一侧出现的维度、以及夹爪。全程噪声的通道现在会正常命中并判 drop_episode ——
它是最强的损坏证据, 不是该被排除的对象。

--------------------------------------------------------------------------
删帧: 只记录区间, 不动数据
--------------------------------------------------------------------------

这是流水线里第一个会产出 drop_frames 的检测 (corrupt/black/occlusion/stillness/saa
都只删 episode)。有一点必须清楚:

**在轨迹中间删帧会制造出一个假的连续轨迹。** 删掉 10 帧 (30fps 下 0.33 秒) 后若重排
timestamp = frame_index/fps, 时间戳就在说谎 —— 一个真实的 0.33 秒不连续被伪装成相邻帧。
对按连续帧训练的策略, 这等于教模型做一次瞬移。原来的突变至少是可见的, 拼接后反而藏起来了。

所以 finding 的 evidence 里带 `segment_positions` (head/tail/middle), 供下游决定:
头尾 -> 截断 (安全, 不制造不连续, mp4 可在关键帧处无损切);
中间 -> 优先丢整条, 舍不得就**切分成两条 episode** 而不是拼接。

本模块只产出区间, 不改任何数据。实际改写的格式相关坑 (LeRobot 的全局 index 重排、
episodes_stats 重算、H.264 无法抽帧、h5py 不能原地删行) 见 sudden_change.md §6。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------

    from check_pipeline.sudden_change import check_sudden_change, extract_signal_matrices

    signals = extract_signal_matrices(parquet_bytes, columns=["observation.state", "action"])
    report = check_sudden_change(signals, sc_profile=profile["sudden_change"], fps=30.0)
    if report.drop_episode:
        ...
    elif report.drop_frames:
        ranges = report.frames_to_drop          # [(start, end), ...] 左闭右开, 无 margin

profile["sudden_change"] 的结构见 sudden_change.md §5.2。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import PipelineEnvironmentError, UnsupportedDataError
from .intervals import indices_to_ranges, merge_ranges, ranges_length
from .signals import extract_signal_matrices

__all__ = [
    "DEFAULT_PARAMS",
    "RULES",
    "Finding",
    "ChannelMeasurement",
    "SuddenChangeReport",
    "extract_signal_matrices",
    "measure_channel",
    "check_sudden_change",
]


DEFAULT_PARAMS = {
    "jump_frac": 0.02,
    #   **唯一的业务阈值**。含义: 一次单帧位置跳变, 幅度达到该通道量程的这个比例就判异常。
    #   0.02 = 2%。一个行程 3.14 rad 的关节 -> 0.063 rad ≈ 3.6°/帧。
    #   单帧尖峰 (跳出去又跳回来) 在一半幅度处就会命中。
    #   调法: 跑一批看命中率, 高了调大低了调小。单位可解释, 所以这个反调循环才成立 ——
    #   旧版的 max(3×noise_baseline, 0.02×量程, 1e-6) 三项取大, 你不知道该动哪一项。

    "max_frame_drop": 10,
    #   异常帧总数 ≤ 此值 -> drop_frames, 超过 -> drop_episode。
    #   对应 table.txt(46) 的"帧数少直接删帧, 帧数多必须删 episode"。
    #   用**绝对帧数**而非百分比: 1000 帧的 episode 按 10% 算就是 100 帧, 太多了。

    "median_window": 3,
    #   中值滤波窗口, 单位**帧**。用途是去掉采样级的单点脉冲, 这件事的尺度就是"一个采样",
    #   与帧率无关, 所以按帧给而不是按秒。

    "smooth_window_sec": 0.23,
    #   Savitzky-Golay 窗口, 单位**秒**。30fps -> 7 帧 (与旧版默认等价)。
    #   旧版按帧给 (sg_window=7), 于是 30fps 下是 233ms 而 10fps 下是 700ms,
    #   "什么算趋势"的物理含义随数据集漂移。

    "smooth_poly": 2,
    #   SG 多项式阶数。2 阶保住加减速的曲率, 又不至于把噪声也拟合进去。
}

# 没有 k_residual/k_accel/k_jerk: 实测 floor 是 MAD 项的 448 倍, 这三个参数调了不起作用。
# 没有 floor_*/rel_floor/range_floor_frac/rel_const/scale_floor_frac: 归一化之后只需要一个阈值。
# 没有 max_noise_ratio/abs_min_range/abs_min_range_max: 不再静默跳过任何通道。
# 没有 severe_run_sec/critical_anomaly_ratio/sc_margin_sec/edge_exempt_sec/min_segment_sec:
#   四档严重度已统一成规则码 + 单一 disposition; margin 会让记录的区间不等于真实异常区间,
#   与 table.txt(48) "记录真实删除片段"冲突。

DROP_EPISODE = "drop_episode"
DROP_FRAMES = "drop_frames"

RULES: Dict[str, Tuple[str, str]] = {
    "SUDDEN_CHANGE_SHORT": (
        "信号出现突变, 异常帧总数在可删范围内 (关节瞬移 / 传感器毛刺 / 短暂跳变)",
        DROP_FRAMES),
    "SUDDEN_CHANGE_LONG": (
        "信号突变涉及的帧数过多, 已不是可定位的局部异常 (整段失真 / 全程乱跳)",
        DROP_EPISODE),
}


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class Finding:
    rule: str
    description: str
    disposition: str
    dims: List[str] = field(default_factory=list)
    frame_ranges: List[Tuple[int, int]] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "description": self.description,
            "disposition": self.disposition,
            "dims": self.dims,
            "frame_ranges": [list(r) for r in self.frame_ranges],
            "evidence": self.evidence,
        }


@dataclass
class ChannelMeasurement:
    """一条 1-D 信号通道的测量结果。**不含判定**。"""
    name: str = ""
    column: str = ""
    dim: int = -1
    range_used: float = 0.0          # profile 给的量程 (阈值的分母)
    threshold: float = 0.0           # θ = jump_frac × range_used
    max_equiv_jump: float = 0.0      # 全段最大等效跳变幅度 (与 threshold 同量纲, 可直接比)
    n_flagged: int = 0
    frame_ranges: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "column": self.column,
            "dim": self.dim,
            "range_used": round(self.range_used, 9),
            "threshold": round(self.threshold, 9),
            "max_equiv_jump": round(self.max_equiv_jump, 9),
            "max_equiv_jump_frac_of_range": (
                round(self.max_equiv_jump / self.range_used, 6) if self.range_used else None),
            "n_flagged": self.n_flagged,
            "frame_ranges": [list(r) for r in self.frame_ranges],
        }


@dataclass
class SuddenChangeReport:
    findings: List[Finding] = field(default_factory=list)
    channels: Dict[str, ChannelMeasurement] = field(default_factory=dict)
    n_frames: int = 0
    fps: float = 0.0
    anomaly_ranges: List[Tuple[int, int]] = field(default_factory=list)
    response_factors: Dict[str, float] = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def has_anomaly(self) -> bool:
        return bool(self.findings)

    @property
    def drop_episode(self) -> bool:
        return any(f.disposition == DROP_EPISODE for f in self.findings)

    @property
    def drop_frames(self) -> bool:
        return any(f.disposition == DROP_FRAMES for f in self.findings)

    @property
    def frames_to_drop(self) -> List[Tuple[int, int]]:
        """真正要删的帧区间 (左闭右开, **不加 margin**)。drop_episode 时为空。"""
        if self.drop_episode:
            return []
        out: List[Tuple[int, int]] = []
        for f in self.findings:
            if f.disposition == DROP_FRAMES:
                out.extend(f.frame_ranges)
        return merge_ranges(out)

    @property
    def triggered_rules(self) -> List[str]:
        seen: List[str] = []
        for f in self.findings:
            if f.rule not in seen:
                seen.append(f.rule)
        return seen

    @property
    def n_frames_anomalous(self) -> int:
        return ranges_length(self.anomaly_ranges)

    def to_dict(self) -> dict:
        return {
            "check": "sudden_change",
            "has_anomaly": self.has_anomaly,
            "drop_episode": self.drop_episode,
            "drop_frames": self.drop_frames,
            "frames_to_drop": [list(r) for r in self.frames_to_drop],
            "triggered_rules": self.triggered_rules,
            "findings": [f.to_dict() for f in self.findings],
            "measurements": {
                "n_frames": self.n_frames,
                "fps": round(self.fps, 3),
                "n_channels_checked": len(self.channels),
                "n_frames_anomalous": self.n_frames_anomalous,
                "anomaly_ranges": [list(r) for r in self.anomaly_ranges],
                "response_factors": {k: round(v, 6)
                                     for k, v in sorted(self.response_factors.items())},
                "params": self.params,
                "per_channel": {n: c.to_dict() for n, c in sorted(self.channels.items())},
            },
        }


# =====================================================================
# 信号处理
# =====================================================================
def _median_filter(x, window: int):
    """一维中值滤波 (去采样级单点脉冲)。

    scipy.signal.medfilt 是零填充边界。对中值来说这是安全的: 填进去的零只有 w//2 个,
    永远不足半窗, 改变不了中位数 —— 换成边缘填充结果一样。(对滑动均值就不成立了。)
    """
    w = int(window)
    if w <= 1:
        import numpy as np
        return np.asarray(x, dtype=np.float64)
    if w % 2 == 0:
        w += 1
    n = len(x)
    if w > n:
        raise UnsupportedDataError(
            f"episode 只有 {n} 帧, 短于中值滤波窗口 {w}, 无法做突变检测")
    try:
        from scipy.signal import medfilt
    except Exception as e:  # pragma: no cover - 环境问题
        raise PipelineEnvironmentError(f"scipy 不可用, 无法做中值滤波: {e}") from e
    return medfilt(x, w)


def _savgol(x, window: int, poly: int):
    """Savitzky-Golay 平滑。scipy 不可用直接报环境错, **不退化成滑动均值**。

    旧版是 `except Exception:` 退化到滑动均值。滑动均值不保峰不保拐点, 而"保峰"正是
    选 SG 的全部理由 —— 算法悄悄换了, 输出不告知, 残差的含义随之改变。
    """
    try:
        from scipy.signal import savgol_filter
    except Exception as e:  # pragma: no cover - 环境问题
        raise PipelineEnvironmentError(
            f"scipy 不可用, 无法做 Savitzky-Golay 平滑: {e}。这是环境问题, 不能退化成"
            f"滑动均值 —— 那是另一个算法, 不保峰不保拐点, 残差的含义完全不同") from e
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w < poly + 2:
        raise UnsupportedDataError(
            f"Savitzky-Golay 窗口 {w} 对阶数 {poly} 太小 (至少 {poly + 2})")
    if len(x) <= w:
        raise UnsupportedDataError(
            f"episode 只有 {len(x)} 帧, 不足以用窗口 {w} 做 Savitzky-Golay 平滑")
    return savgol_filter(x, w, poly)


def _abs_finite_diff(x, order: int):
    """|n 阶有限差分|, 补零对齐到原长度。

    order=2 (加速度) 的结果中心正好落在整数下标; order=3 (跃度) 的中心在 i+1.5,
    写回整数下标有固定的 0.5 帧偏移 —— 这是奇数阶差分的固有性质, 不影响幅度判定,
    但定位异常段时要知道 J 的贡献可能偏半帧。
    """
    import numpy as np
    n = len(x)
    if n <= order:
        raise UnsupportedDataError(
            f"episode 只有 {n} 帧, 不足以计算 {order} 阶差分")
    d = np.diff(np.asarray(x, dtype=np.float64), n=order)
    out = np.zeros(n, dtype=np.float64)
    out[order // 2: order // 2 + len(d)] = d
    return np.abs(out)


def _sg_window(p: dict, fps: float) -> int:
    """按秒给的平滑窗口换算成帧, 并夹到该阶数下真正能起平滑作用的最小值。

    下限是 poly + 3 (取奇数): 窗口等于 poly + 1 时多项式**精确穿过**所有点, 输出即输入,
    等于没平滑。所以 2 阶最小是 5 帧, 不是 3 帧。

    低帧率下窗口会被下限顶上去 (10fps 时 0.23s 只有 2 帧 -> 夹到 5 帧 = 0.5s), 实际
    时间常数与请求值不符。这件事**必须让人看见**, 所以 check_sudden_change 会把请求值
    和生效值一起写进 measurements —— 悄悄改平滑强度会让残差含义整体变化 (窗口越大趋势
    线越滞后, 残差越大, 检测越敏感)。
    """
    poly = int(p["smooth_poly"])
    win = int(round(float(p["smooth_window_sec"]) * float(fps)))
    win = max(win, poly + 3)
    if win % 2 == 0:
        win += 1
    return win


def _response_factors(median_window: int, window: int, poly: int) -> Dict[str, float]:
    """把一个合成的**单位阶跃**推过同一套滤波器, 量出 R/A/J 各自的响应倍数。

    这三个数是把 R/A/J 归一到"等效单次跳变幅度"的分母。为什么运行时测而不是写死常数:
    R 的响应完全取决于 SG 的窗口和阶数 (趋势线追上阶跃的快慢), 窗口按秒给之后不同帧率
    下窗口帧数不同, 写死的常数一定会和实际配置对不上。A/J 只依赖差分阶数 (解析值 1 和 2),
    一起测保证三者出自同一条口径。

    量的是**阶跃**而不是尖峰: 阶跃是一次跳变, 尖峰是跳出去再跳回来 (两次)。以单次跳变
    为基准, θ 的含义才是"一次跳变多大算异常"; 尖峰因此在 θ/2 附近就会命中, 这是对的 ——
    同样幅度的尖峰比阶跃包含更剧烈的局部变化。
    """
    import numpy as np
    n = max(64, 8 * int(window) + 8)
    x = np.zeros(n, dtype=np.float64)
    x[n // 2:] = 1.0
    xm = _median_filter(x, median_window)
    s = _savgol(xm, window, poly)
    factors = {
        "residual": float(np.max(np.abs(x - s))),
        "accel": float(np.max(_abs_finite_diff(x, 2))),
        "jerk": float(np.max(_abs_finite_diff(x, 3))),
    }
    bad = [k for k, v in factors.items() if not (v > 1e-9)]
    if bad:
        raise PipelineEnvironmentError(
            f"响应倍数标定失败, {bad} 的响应为零。滤波器配置 "
            f"(median_window={median_window}, window={window}, poly={poly}) 不可用")
    return factors


# =====================================================================
# 测量
# =====================================================================
def measure_channel(x, name: str, column: str, dim: int, range_used: float,
                    factors: Dict[str, float], fps: float, p: dict) -> ChannelMeasurement:
    """测一条 1-D 信号通道。**不做任何判定** —— 规则由 check_sudden_change 决定。

    range_used 来自 profile (全数据集的量程), 不是本条 episode 的量程。见模块 docstring。
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        n_bad = int((~np.isfinite(x)).sum())
        raise UnsupportedDataError(
            f"通道 {name} 有 {n_bad} 帧是 NaN/Inf。旧版在这里做线性插值填上, 那是在"
            f"凭空造数据 —— 插出来的值平滑无突变, 恰好让这段看起来是干净的。"
            f"缺失值该由上游先处理掉")

    win = _sg_window(p, fps)
    x_med = _median_filter(x, p["median_window"])
    trend = _savgol(x_med, win, p["smooth_poly"])

    # 三个偏差信号, 各自除以对单位阶跃的响应倍数 -> 统一成"等效单次跳变幅度"
    r_hat = np.abs(x - trend) / factors["residual"]
    a_hat = _abs_finite_diff(x, 2) / factors["accel"]
    j_hat = _abs_finite_diff(x, 3) / factors["jerk"]

    theta = float(p["jump_frac"]) * float(range_used)

    # 残差主导 + 导数确认: 只有残差大而导数小 = 慢漂移, 不算突变;
    # 只有导数大而残差小 = 高频噪声 (中值滤波已处理), 也不算。
    flag = (r_hat > theta) & ((a_hat > theta) | (j_hat > theta))

    out = ChannelMeasurement(name=name, column=column, dim=dim,
                             range_used=float(range_used), threshold=theta)
    # 全段最大等效跳变: 取三者里最保守的那个 (联合判据实际能达到的读数)
    combined = np.minimum(r_hat, np.maximum(a_hat, j_hat))
    out.max_equiv_jump = float(np.max(combined)) if len(combined) else 0.0
    out.n_flagged = int(flag.sum())
    out.frame_ranges = indices_to_ranges(np.flatnonzero(flag).tolist())
    return out


# =====================================================================
# 判定
# =====================================================================
def check_sudden_change(signals: Dict[str, Any],
                        sc_profile: dict,
                        fps: float,
                        params: Optional[dict] = None,
                        ) -> SuddenChangeReport:
    """对一条 episode 跑突变检测。

    **两条规则** (对应 table.txt(46) "帧数少直接删帧, 帧数多必须删 episode"):
      SUDDEN_CHANGE_SHORT   异常帧总数 <= max_frame_drop  -> drop_frames
      SUDDEN_CHANGE_LONG    超过                          -> drop_episode

    异常帧是所有通道的**并集**: 任一维度突变, 该时间步的整帧数据都不可信 (状态/动作/
    画面是同一时刻的记录), 没有理由只删某一维。

    signals: {列名: (T, D) 矩阵}, 由 extract_signal_matrices() 得到。
    sc_profile: profile["sudden_change"], 结构见 sudden_change.md §5.2。
        enabled=False 的数据集**不应该调用本函数** —— 调用方要把"跳过"单独记账,
        否则"零异常"和"根本没跑"在报告里长得一模一样。
    fps: 帧率, 用于把按秒给的平滑窗口换算成帧。

    可能抛 CheckError, 表示**检测没能完成**; 调用方必须记为"检测失败", 不是"通过"。
    """
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - 环境问题
        raise PipelineEnvironmentError(f"numpy 不可用: {e}") from e

    p = dict(DEFAULT_PARAMS)
    if params:
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise UnsupportedDataError(
                f"未知参数 {sorted(unknown)}; 可用参数: {sorted(DEFAULT_PARAMS)}")
        p.update(params)

    if not sc_profile:
        raise UnsupportedDataError(
            "未提供 sudden_change profile。参与检测的维度必须由 profile 显式声明 —— "
            "旧版靠通道名猜 (子串匹配 velocit/vel/effort/...) 的做法已被实测证伪: "
            "'vel' 会误伤 level/travel, 而 d0..d13 这类命名里根本没有语义")
    if sc_profile.get("enabled") is False:
        raise UnsupportedDataError(
            f"profile 标记本数据集不做突变检测 (reason: {sc_profile.get('reason', '未说明')})。"
            f"调用方应把它记为'跳过'并单独计数, 不能当成通过")

    for key in ("jump_frac", "max_frame_drop"):
        if key in sc_profile:
            p[key] = type(DEFAULT_PARAMS[key])(sc_profile[key])
    if not (0.0 < p["jump_frac"] <= 1.0):
        raise UnsupportedDataError(
            f"jump_frac 必须落在 (0, 1], 收到 {p['jump_frac']}。它的含义是"
            f"'占该通道量程的比例'")
    if int(p["max_frame_drop"]) < 0:
        raise UnsupportedDataError(f"max_frame_drop 不能为负, 收到 {p['max_frame_drop']}")

    if not fps or fps <= 0:
        raise UnsupportedDataError(
            f"拿不到有效帧率 (fps={fps}), 按秒给的平滑窗口无法换算成帧")

    raw_dims = sc_profile.get("dims") or []
    if not raw_dims:
        raise UnsupportedDataError(
            "profile 的 dims 为空, 没有可检测的信号维度。本数据集应在 profile 里标"
            "enabled=false 并由调用方记为跳过")

    if not signals:
        raise UnsupportedDataError("未提供任何信号矩阵")
    n_frames_set = {int(np.asarray(m).shape[0]) for m in signals.values()}
    if len(n_frames_set) > 1:
        raise UnsupportedDataError(f"各列行数不一致: {sorted(n_frames_set)}, 不做截断对齐")
    n_frames = n_frames_set.pop()

    # ---- 解析 profile 里的维度声明 ----
    specs: List[Tuple[str, str, int, float]] = []       # (name, column, dim, range)
    seen: set = set()
    for i, d in enumerate(raw_dims):
        try:
            column = str(d["column"])
            dim = int(d["dim"])
        except (KeyError, TypeError, ValueError) as e:
            raise UnsupportedDataError(
                f"dims[{i}] 缺少 column / dim 或类型不对: {d!r}") from e
        name = str(d.get("name") or f"{column}.d{dim}")
        if name in seen:
            raise UnsupportedDataError(f"dims 里维度名重复: {name!r}")
        seen.add(name)
        if column not in signals:
            raise UnsupportedDataError(
                f"dims[{i}] ({name}) 声明的列 {column!r} 不在提供的信号里 "
                f"(实际: {sorted(signals)})")
        mat = np.asarray(signals[column], dtype=np.float64)
        if mat.ndim != 2:
            raise UnsupportedDataError(
                f"列 {column!r} 必须是二维矩阵, 收到 {mat.shape}")
        if not (0 <= dim < mat.shape[1]):
            raise UnsupportedDataError(
                f"dims[{i}] ({name}) 的 dim={dim} 超出列 {column!r} 的实际维度 "
                f"[0, {mat.shape[1]})。profile 与数据不匹配")
        if "range" not in d:
            raise UnsupportedDataError(
                f"dims[{i}] ({name}) 没有给 range。量程必须由 profile 提供 (扫描阶段对"
                f"全数据集算的 p99-p1), 不能退化成用本条 episode 估 —— 那会让同一个关节"
                f"在不同 episode 上阈值不同: 机器人动得少的那条量程小、阈值更紧、更容易"
                f"命中, 是个不会被注意到的系统性偏差")
        try:
            rng = float(d["range"])
        except (TypeError, ValueError) as e:
            raise UnsupportedDataError(f"dims[{i}] ({name}) 的 range 不是数: {d['range']!r}") from e
        if not (rng > 0) or not np.isfinite(rng):
            raise UnsupportedDataError(
                f"dims[{i}] ({name}) 的 range={rng} 非正或非有限。量程为零说明这个维度"
                f"在整个数据集上都没动过, 它不该出现在 dims 里")
        specs.append((name, column, dim, rng))

    sg_win = _sg_window(p, fps)
    factors = _response_factors(int(p["median_window"]), sg_win, int(p["smooth_poly"]))

    reported = dict(p)
    # 生效的平滑窗口单独写出来: 低帧率下它会被下限顶上去, 与请求的秒数不符 (见 _sg_window)
    reported["sg_window_frames"] = sg_win
    reported["sg_window_sec_effective"] = round(sg_win / float(fps), 6)
    report = SuddenChangeReport(n_frames=n_frames, fps=float(fps),
                                response_factors=factors, params=reported)

    # ---- 逐通道测量 ----
    anomaly = np.zeros(n_frames, dtype=bool)
    for name, column, dim, rng in specs:
        mat = np.asarray(signals[column], dtype=np.float64)
        m = measure_channel(mat[:, dim], name, column, dim, rng, factors, float(fps), p)
        report.channels[name] = m
        for a, b in m.frame_ranges:
            anomaly[a:b] = True

    report.anomaly_ranges = indices_to_ranges(np.flatnonzero(anomaly).tolist())
    total = ranges_length(report.anomaly_ranges)
    if total == 0:
        return report

    # ---- 判定 ----
    hit = {n: c for n, c in report.channels.items() if c.n_flagged}
    max_frame_drop = int(p["max_frame_drop"])
    evidence = {
        "criterion": ("异常帧总数 <= max_frame_drop -> drop_frames, 否则 drop_episode; "
                      "帧异常 = (R̂ > θ) 且 (Â > θ 或 Ĵ > θ), θ = jump_frac × range"),
        "jump_frac": p["jump_frac"],
        "max_frame_drop": max_frame_drop,
        "n_frames": n_frames,
        "n_frames_anomalous": total,
        "anomaly_frac": round(total / n_frames, 6) if n_frames else None,
        "n_channels_checked": len(report.channels),
        "n_channels_hit": len(hit),
        "response_factors": {k: round(v, 6) for k, v in sorted(factors.items())},
        # 没命中的通道也全列出来: "只有这一维在跳"和"所有维度一起跳"是两种完全不同的
        # 问题 (前者多半是单个编码器坏了, 后者是整帧数据出问题), 只列命中的就分不出来
        "per_channel_n_flagged": {n: c.n_flagged
                                  for n, c in sorted(report.channels.items())},
        "per_channel_max_equiv_jump_frac": {
            n: (round(c.max_equiv_jump / c.range_used, 6) if c.range_used else None)
            for n, c in sorted(report.channels.items())},
    }

    if total <= max_frame_drop:
        # 异常段落在哪里决定下游能怎么处理: 头尾可以安全截断, 中间删帧会把一个真实的
        # 时间不连续伪装成相邻帧 (见模块 docstring)。这里只如实标注, 不替下游决定。
        # head/tail 的判据不是"贴着边界", 而是"能不能改写成截断而不超预算":
        # 异常段落在第 1-3 帧时截掉 [0,3) 和截掉 [0,2) 一样安全, 严格要求 a==0 会把这种
        # 情况错标成 middle。用已有的 max_frame_drop 当预算, 不额外引入参数 ——
        # 含义正好是"把它变成一次截断, 代价仍在你已经设定的删帧预算内"。
        positions = []
        for a, b in report.anomaly_ranges:
            if b <= max_frame_drop:
                positions.append("head")        # 截断 [0, b) 代价 b 帧
            elif n_frames - a <= max_frame_drop:
                positions.append("tail")        # 截断 [a, n) 代价 n-a 帧
            else:
                positions.append("middle")
        evidence["segment_positions"] = positions
        evidence["has_middle_segment"] = "middle" in positions
        evidence["note_on_middle"] = (
            "middle 段直接删掉会让删除点两侧变成相邻帧, 把一个真实的时间不连续伪装成"
            "连续轨迹。建议头尾截断、中间切分成两条 episode, 而不是拼接")
        report.findings.append(_finding(
            "SUDDEN_CHANGE_SHORT", dims=sorted(hit),
            frame_ranges=report.anomaly_ranges, evidence=evidence))
    else:
        report.findings.append(_finding(
            "SUDDEN_CHANGE_LONG", dims=sorted(hit),
            frame_ranges=report.anomaly_ranges, evidence=evidence))
    return report


def _finding(rule: str, dims: Optional[List[str]] = None,
             frame_ranges: Optional[List[Tuple[int, int]]] = None,
             evidence: Optional[dict] = None) -> Finding:
    desc, disp = RULES[rule]
    return Finding(rule=rule, description=desc, disposition=disp,
                   dims=dims or [], frame_ranges=merge_ranges(frame_ranges or []),
                   evidence=evidence or {})
