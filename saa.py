"""State-Action 趋势对齐 (stage 2)。

PDF §2.4 Stage 2 / Qwen 的做法:

  "In a correctly recorded episode, action commands should temporally lead or coincide
   with resulting state changes, which is a causal invariant violated when timestamps
   are unsynchronized or there is packet loss. For each shared joint dimension, we
   smooth both the state and action trajectories, then estimate the optimal temporal
   lag via cross-correlation, then compute a directional agreement (DA) metric on
   lag-aligned first-order differences. Dimensions with DA below a dataset-specific
   threshold (typically 0.6-0.7) are flagged and their episodes excluded. For datasets
   using delta actions, we first integrate the action sequence to recover absolute
   values before comparison."

前四个检测看的是**画面**, 这个检测第一次去看**信号本身**: 机器人的实际运动是不是真的
由记录下来的指令引起的。它抓时间戳不同步、丢包、state 与 action 来自不同时钟这类问题
—— 这些在画面上完全看不出来, 但会让训练出来的策略学到错误的因果。

完整设计与实测证据见 saa.md。

--------------------------------------------------------------------------
维度从 profile 来, 不在代码里猜 (最重要的一条)
--------------------------------------------------------------------------

旧版 (new/state_action_alignment_frame.py) 试图从列名推断哪些维度该检, 这条路已经被
实测证伪: OXE / GM-100 那种"一个 observation.state 列装 14 维"的布局, 展开后维度名是
d0 ... d13, 名字里没有任何语义。旧版号称"gripper 全豁免", 靠的是列名里找 "gripper"
子串 —— 在 d13 上完全失效, 夹爪照样进了 DA 判定。而夹爪恰恰是最容易 DA 失败的维度
(离散指令 vs 连续机械响应, 还带大滞后)。

所以这一版**不做任何名字匹配、不做任何维度筛选**, 维度配对由 profile 给出显式下标。
profile 缺失或没有可用维度 -> UnsupportedDataError, **不退化成"用列名猜"**。

--------------------------------------------------------------------------
因果方向是判据, 不是对齐手段
--------------------------------------------------------------------------

Qwen 的定义是 action **领先或同时于** state, 即 tau >= 0 才合法; tau < 0 (state 先动)
就是因果被破坏, 正是要检出的目标。

旧版在 [-L, +L] 全域搜最佳 lag 且**负 lag 照样接受**, 然后按它对齐再算 DA。于是一条
时间戳错位、state 领先 action 3 帧的 episode, 会被 tau=-3 对齐得整整齐齐, DA 很高,
顺利通过 —— **检测把自己该报的缺陷修正掉了**。

这一版: 全域搜 tau* (取正相关最大, 不取 abs), tau* 显著为负单独触发
SAA_CAUSALITY_VIOLATED; 算 DA 时用 tau = max(tau*, 0), 不允许用负 lag 去"修好"因果。

--------------------------------------------------------------------------
delta action 必须先积分
--------------------------------------------------------------------------

PDF 那句 "we first integrate the action sequence" 在旧版里只有引文没有实现。后果:
state 是位置、action 是位移, diff(state) 是速度而 diff(action) 是加速度, 比这两个东西
的符号在物理上没有意义, DA 精确落在 0.5 (随机猜的水平)。

实测 (合成数据, 因果完全正常):
    delta action 未积分   -> DA = 0.496  判失败
    delta action 积分后   -> DA = 1.000  判通过

action_space 由 profile 声明, 运行时**复验**: 两种假设各算一次一致率, 与声明不符就报错。
一条 episode 的动作空间不可能和数据集其他部分不同, 真出现了那是数据有问题。

--------------------------------------------------------------------------
关于帧级死区: 不加, 但输出可证伪的诊断
--------------------------------------------------------------------------

DA 不排除小幅差分的帧。理由: 维度选择已经在 profile 层面把"整条 episode 不动的关节"
排除掉了, 选进来的关节在操作过程中大部分时间是在动的。

但这个决定有边界。实测一个**整体在动**、但 episode 内周期性保持不动的关节 (因果正常):

    保持段占比   DA(无死区)   DA(死区=|Δ| 中位数)   判定@0.7
        0%         0.952           1.000            通过
       30%         0.795           0.983            通过
       50%         0.714           0.916            通过 (余量已经很薄)
       60%         0.660           0.836            **误删**

注意 stage 1 的 stillness **挡不住这种情况**: 它算的是 ‖Δstate‖ 全维度范数 + 画面没变,
一个关节保持不动而其他关节在大幅扫动时, 范数很大、画面也在变, 完整放行。两个阶段口径
不同 —— stage 1 问"整个机器人有没有动", SAA 问"这一个关节有没有动"。

所以行为不变, 但每个维度额外输出几个死区档位下的 DA (死区取该维度自身 |diff(state)|
的分位数, **不引入任何新参数**, 不参与判定):

    "da_at_deadband": {"0": 0.714, "p25": 0.842, "p50": 0.916}

三个数接近 -> 保持段不是问题; "0" 明显低于 "p50" -> 被删的里面有一批是误删。
把一个判断变成能被数据证伪的假设。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------

    from check_pipeline.saa import check_saa, extract_saa_matrices

    state, action = extract_saa_matrices(parquet_bytes,
                                         state_column="observation.state",
                                         action_column="action")
    report = check_saa(state, action, saa_profile=profile["saa"], fps=30.0)
    if report.drop_episode:
        ...                      # report.triggered_rules / report.findings

profile["saa"] 的结构见 saa.md §3.1。
"""

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import PipelineEnvironmentError, UnsupportedDataError

__all__ = [
    "DEFAULT_PARAMS",
    "RULES",
    "Finding",
    "DimAlignment",
    "SAAReport",
    "extract_saa_matrices",
    "measure_dim",
    "check_saa",
]


DEFAULT_PARAMS = {
    "da_threshold": 0.70,
    #   方向一致性下限。PDF/Qwen 给的范围是 0.6-0.7, 默认取严的一端。
    #   这是**唯一的业务参数**, 其余都是信号处理参数。按数据集可由 profile 覆盖。

    "max_lag_sec": 0.20,
    #   互相关搜索范围, 单位**秒**。30fps -> ±6 帧, 10fps -> ±2 帧。
    #   旧版按帧给 (max_lag=5), 于是 30fps 下是 167ms 而 10fps 下是 500ms,
    #   物理含义随数据集漂移。控制回路的实际延迟是以时间计的, 不是以帧计的。

    "smooth_window_sec": 0.17,
    #   Savitzky-Golay 窗口, 单位**秒**。30fps -> 5 帧 (与旧版默认等价)。
    #   平滑是为了让一阶差分的符号反映趋势而不是反映噪声。

    "smooth_poly": 2,
    #   Savitzky-Golay 多项式阶数。2 阶保住加减速的曲率, 又不至于把噪声也拟合进去。

    "causality_tol_frames": 1,
    #   tau* 允许的负向容差。留 1 帧是因为平滑和量化本身会带来亚帧级相位偏移,
    #   硬卡 tau* < 0 会把它们全报成因果错乱。
}

# 没有 min_std: 静止维度的问题在 profile 层面解决 (只选运动相关的关节),
# 不在这里靠阈值兜。旧版的 min_std=1e-6 是绝对阈值套在单位未知的量上,
# 实测一个"没动"的关节噪声 std 是 1e-3, 比阈值大三个量级, 照检不误。

DROP_EPISODE = "drop_episode"

RULES: Dict[str, Tuple[str, str]] = {
    "SAA_DA_BELOW_THRESHOLD": (
        "存在共享关节维度的方向一致性低于阈值 (state 与 action 的因果关系被破坏)",
        DROP_EPISODE),
    "SAA_CAUSALITY_VIOLATED": (
        "存在共享关节维度上 state 领先 action (机器人先动, 指令后到 = 时间戳错位/丢包)",
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
    dims: List[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "description": self.description,
            "disposition": self.disposition,
            "dims": self.dims,
            "evidence": self.evidence,
        }


@dataclass
class DimAlignment:
    """一个共享关节维度的对齐测量结果。**不含判定**。"""
    name: str = ""
    state_dim: int = -1
    action_dim: int = -1
    da: float = 0.0                  # 判定用: 在 tau = max(tau*, 0) 处的方向一致性
    lag_frames: int = 0              # 判定用的对齐量 = max(tau*, 0)
    best_lag_frames: int = 0         # tau*: 全域最佳 lag, **可能为负** (因果判定看它)
    lag_sec: float = 0.0
    cross_corr: float = 0.0          # tau* 处的相关系数 (带符号, 不取 abs)
    n_frames_used: int = 0           # 参与 DA 统计的帧数 (两侧符号都非零)
    state_std: float = 0.0
    action_std: float = 0.0
    da_at_deadband: Dict[str, float] = field(default_factory=dict)   # 诊断, 不参与判定

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "da": round(self.da, 6),
            "lag_frames": self.lag_frames,
            "best_lag_frames": self.best_lag_frames,
            "lag_sec": round(self.lag_sec, 6),
            "cross_corr": round(self.cross_corr, 6),
            "n_frames_used": self.n_frames_used,
            "state_std": round(self.state_std, 9),
            "action_std": round(self.action_std, 9),
            "da_at_deadband": {k: round(v, 6) for k, v in self.da_at_deadband.items()},
        }


@dataclass
class SAAReport:
    findings: List[Finding] = field(default_factory=list)
    dims: Dict[str, DimAlignment] = field(default_factory=dict)
    n_frames: int = 0
    fps: float = 0.0
    action_space_declared: str = ""
    action_space_measured: str = ""
    action_space_evidence: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def has_misalignment(self) -> bool:
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

    @property
    def min_da(self) -> float:
        return min((d.da for d in self.dims.values()), default=1.0)

    def to_dict(self) -> dict:
        return {
            "check": "saa",
            "has_misalignment": self.has_misalignment,
            "drop_episode": self.drop_episode,
            "triggered_rules": self.triggered_rules,
            "findings": [f.to_dict() for f in self.findings],
            "measurements": {
                "n_frames": self.n_frames,
                "fps": round(self.fps, 3),
                "n_dims_checked": len(self.dims),
                "min_da": round(self.min_da, 6),
                "action_space_declared": self.action_space_declared,
                "action_space_measured": self.action_space_measured,
                "action_space_evidence": self.action_space_evidence,
                "params": self.params,
                "per_dim": {n: d.to_dict() for n, d in sorted(self.dims.items())},
            },
        }


# =====================================================================
# 信号提取
# =====================================================================
def extract_saa_matrices(parquet_bytes: Optional[bytes],
                         state_column: str,
                         action_column: str) -> Tuple[Any, Any]:
    """从 parquet 里取出 state (T, Ds) 和 action (T, Da)。

    列名由 profile 显式给出, **不做任何猜测**。profile 里的 state_dim / action_dim
    就是这两个矩阵的列下标。

    报错情况 (不降级):
      - pandas / numpy 不可用      -> PipelineEnvironmentError
      - parquet 为 None / 读不出    -> UnsupportedDataError
      - profile 指定的列不存在      -> UnsupportedDataError
      - state 与 action 行数不一致   -> UnsupportedDataError (**不截断到最短**)
    """
    if parquet_bytes is None:
        raise UnsupportedDataError(
            "未提供 parquet 字节, 取不到 state/action 信号, 无法做趋势对齐")
    try:
        import numpy as np
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise PipelineEnvironmentError(f"numpy/pandas 不可用: {e}") from e

    try:
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        raise UnsupportedDataError(f"parquet 读取失败: {type(e).__name__}: {e}") from e

    cols = {str(c) for c in df.columns}
    missing = [c for c in (state_column, action_column) if c not in cols]
    if missing:
        raise UnsupportedDataError(
            f"profile 指定的列 {missing} 不在 parquet 里 (实际列名: {sorted(cols)})。"
            f"profile 与数据不匹配, 不猜其他列")

    def _mat(col: str):
        s = df[col]
        m = np.asarray(np.stack(s.to_numpy()) if s.dtype == object else s.to_numpy(),
                       dtype=np.float64)
        return m.reshape(-1, 1) if m.ndim == 1 else m

    state = _mat(state_column)
    action = _mat(action_column)
    if state.shape[0] != action.shape[0]:
        raise UnsupportedDataError(
            f"state 与 action 行数不一致 ({state.shape[0]} vs {action.shape[0]})。"
            f"不做截断对齐 —— 截短之后逐帧比对的是错位的两条信号, 而且长度依然自洽, "
            f"事后无法发现")
    return state, action


# =====================================================================
# 测量
# =====================================================================
def _savgol(x, window: int, poly: int):
    """Savitzky-Golay 平滑。scipy 不可用直接报环境错, **不静默返回原值**。

    旧版是 `except Exception: return x` —— scipy 缺失时所有维度悄悄改用未平滑数据,
    噪声不去除、DA 系统性偏低、大批误删, 而且没有任何记录。
    """
    try:
        from scipy.signal import savgol_filter
    except Exception as e:  # pragma: no cover - 环境问题
        raise PipelineEnvironmentError(
            f"scipy 不可用, 无法平滑: {e}。这是环境问题, 不能退化成用未平滑数据 —— "
            f"那会让 DA 系统性偏低, 把正常数据大批判成对齐失败") from e
    w = window if window % 2 == 1 else window + 1
    if w < poly + 2:
        # 这不是数据的问题, 是窗口算出来太小 —— 调用方应先经 _sg_window 夹一道。
        # 两种失败必须分开报: 混在一起会让人拿着"episode 太短"去查一条 112 帧的数据。
        raise UnsupportedDataError(
            f"平滑窗口 {w} 帧对 {poly} 阶多项式太小 (至少 {poly + 2} 帧)。"
            f"窗口等于阶数+1 时多项式精确穿过所有点, 输出即输入, 等于没平滑")
    if len(x) <= w:
        raise UnsupportedDataError(
            f"episode 太短, 无法做 Savitzky-Golay 平滑 (长度 {len(x)}, 窗口 {w}, 阶数 {poly})")
    return savgol_filter(x, w, poly)


def _sg_window(p: dict, fps: float) -> int:
    """按秒给的平滑窗口换算成帧, 并夹到该阶数下真正能起平滑作用的最小值。

    下限是 poly + 3 (取奇数): 窗口等于 poly + 1 时多项式**精确穿过**所有点, 输出即输入,
    等于没平滑。所以 2 阶最小是 5 帧, 不是 3 帧。

    不夹的后果是真实的: fps=20 时 0.17s 只有 3.4 帧, 取整成 3, 直接把整个数据集判成
    "无法检测" —— 而 10~30fps 的数据集非常多。夹完之后实际时间常数与请求值不符, 这件事
    **必须让人看见**, 所以 check_saa 会把请求值和生效值一起写进 measurements。
    """
    poly = int(p["smooth_poly"])
    win = int(round(float(p["smooth_window_sec"]) * float(fps)))
    win = max(win, poly + 3)
    if win % 2 == 0:
        win += 1
    return win


def _agreement(a, b, deadband: float = 0.0) -> Tuple[float, int]:
    """符号一致率。返回 (一致率, 参与统计的帧数)。

    deadband > 0 时只统计 |a| > deadband 的帧 —— 仅用于诊断输出, 判定路径恒为 0。
    没有可统计的帧时返回 (nan, 0): **不返回 1.0**。
    旧版在这里 `return 1.0`, 把"没测出来"当成"测过了没问题", 是假阴性。
    """
    import numpy as np
    sa = np.sign(a)
    sb = np.sign(b)
    mask = (sa != 0) & (sb != 0)
    if deadband > 0:
        mask &= np.abs(a) > deadband
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float((sa[mask] == sb[mask]).mean()), n


def _best_lag(s, a, max_lag: int) -> Tuple[int, float]:
    """互相关找最佳滞后 tau。

    tau > 0 表示 state 落后 action tau 帧, 即 **action 领先** (正常)。
    tau < 0 表示 state 领先 action (因果被破坏)。

    取**正相关**最大, 不取 abs。旧版取了 abs, 于是完全反相关 (corr=-1) 被当成最佳对齐,
    符号约定相反的数据集会被引到一个错误的 lag 上。
    """
    import numpy as np
    best_tau, best_c = 0, -2.0
    for tau in range(-max_lag, max_lag + 1):
        if tau >= 0:
            x, y = s[tau:], a[:len(a) - tau]
        else:
            x, y = s[:len(s) + tau], a[-tau:]
        if len(x) < 3 or len(y) < 3 or len(x) != len(y):
            continue
        if np.std(x) < 1e-15 or np.std(y) < 1e-15:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(c):
            continue
        if c > best_c:
            best_tau, best_c = tau, c
    if best_c < -1.5:
        raise UnsupportedDataError(
            "互相关无法计算 (信号过短或方差为零), 无法估计 state/action 的时间滞后")
    return best_tau, best_c


def _integrate(action_dim, anchor: float):
    """delta action 积分恢复绝对量: a_abs[t] = anchor + sum(action[:t])。

    anchor 取 state[0] 只是让曲线可读, 对结果无影响 (互相关会去均值, DA 看的是差分)。

    一个值得知道的恒等式: diff(cumsum(action))[t] == action[t+1]。所以**就 DA 而言**
    "先积分再一阶差分"等价于"直接用 action", 差的一帧偏移会被 lag 搜索吸收 ——
    这正是 _measure_action_space 的判别式能成立的原因。但**估计 lag 时积分是必要的**:
    在绝对轨迹上做互相关比在差分上稳得多。
    """
    import numpy as np
    out = np.empty(len(action_dim), dtype=np.float64)
    out[0] = anchor
    if len(action_dim) > 1:
        out[1:] = anchor + np.cumsum(action_dim[:-1])
    return out


def _measure_action_space(state, action, pairs) -> Tuple[str, dict]:
    """判定 action 是绝对量还是增量。两种假设各算一次一致率, 取高的那个。

    因为 delta action 本身就约等于 diff(state), 所以:
        DA_as_absolute = 一致率( diff(state), diff(action) )
        DA_as_delta    = 一致率( diff(state), action )
    哪个假设成立哪个数就高。再配一个独立证据 std(action)/std(state): 绝对量约为 1,
    增量小两个数量级。实测两个方向都是 0.938 vs 0.498, 幅度比差 100 倍。
    """
    import numpy as np
    abs_scores, delta_scores, ratios = [], [], []
    for _, si, ai in pairs:
        s = state[:, si]
        a = action[:, ai]
        ds = np.diff(s)
        sa, _ = _agreement(ds, np.diff(a))
        sd, _ = _agreement(ds, a[:-1])
        if not np.isnan(sa):
            abs_scores.append(sa)
        if not np.isnan(sd):
            delta_scores.append(sd)
        s_std = float(np.std(s))
        if s_std > 0:
            ratios.append(float(np.std(a)) / s_std)
    if not abs_scores and not delta_scores:
        raise UnsupportedDataError(
            "所有共享维度都无法比较符号 (信号恒定或过短), 判不出 action 是绝对量还是增量")
    da_abs = float(np.mean(abs_scores)) if abs_scores else 0.0
    da_del = float(np.mean(delta_scores)) if delta_scores else 0.0
    verdict = "delta" if da_del > da_abs else "absolute"
    return verdict, {
        "da_as_absolute": round(da_abs, 6),
        "da_as_delta": round(da_del, 6),
        "std_ratio": round(float(np.mean(ratios)), 6) if ratios else None,
        "rule": "取一致率更高的那个假设; std_ratio 作为独立旁证 (绝对量约 1, 增量小两个量级)",
    }


def measure_dim(state_dim, action_dim, name: str, si: int, ai: int,
                fps: float, p: dict) -> DimAlignment:
    """测一个共享关节维度。**不做任何判定** —— 是否触发规则由 check_saa 决定。

    action_dim 传进来时必须已经是**绝对量**(delta 的话由调用方先积分)。
    """
    import numpy as np

    out = DimAlignment(name=name, state_dim=si, action_dim=ai)
    out.state_std = float(np.std(state_dim))
    out.action_std = float(np.std(action_dim))

    win = _sg_window(p, fps)
    s_sm = _savgol(state_dim, win, p["smooth_poly"])
    a_sm = _savgol(action_dim, win, p["smooth_poly"])

    max_lag = max(1, int(round(p["max_lag_sec"] * fps)))
    tau_star, corr = _best_lag(s_sm, a_sm, max_lag)
    out.best_lag_frames = tau_star
    out.cross_corr = corr

    # 因果只允许 action 领先或同时: 算 DA 时不使用负 lag 去"修好"错位
    tau = max(tau_star, 0)
    out.lag_frames = tau
    out.lag_sec = tau / fps if fps else 0.0

    s_diff = np.diff(s_sm)
    a_diff = np.diff(a_sm)
    n = min(len(s_diff) - tau, len(a_diff))
    if n < 3:
        raise UnsupportedDataError(
            f"维度 {name}: 按 lag={tau} 对齐后只剩 {n} 帧, 不足以统计方向一致性")
    s_al = s_diff[tau:tau + n]
    a_al = a_diff[:n]

    da, used = _agreement(s_al, a_al)
    if np.isnan(da):
        raise UnsupportedDataError(
            f"维度 {name}: 没有任何一帧的 state/action 差分同时非零, 无法计算方向一致性。"
            f"这个维度不该出现在 profile 的 shared_dims 里")
    out.da = da
    out.n_frames_used = used

    # 诊断: 同一维度在几个死区档位下的 DA。死区取该维度自身 |diff(state)| 的分位数,
    # 不引入新参数, 不参与判定。见模块 docstring。
    mag = np.abs(s_al)
    out.da_at_deadband = {"0": da}
    for label, q in (("p25", 25), ("p50", 50)):
        d, _n = _agreement(s_al, a_al, deadband=float(np.percentile(mag, q)))
        out.da_at_deadband[label] = da if np.isnan(d) else d
    return out


# =====================================================================
# 判定
# =====================================================================
def check_saa(state: Any,
              action: Any,
              saa_profile: dict,
              fps: float,
              params: Optional[dict] = None,
              ) -> SAAReport:
    """对一条 episode 跑 state-action 趋势对齐。

    **两条删除规则, 都是 episode 级**:
      SAA_DA_BELOW_THRESHOLD   任一共享关节维度 DA < da_threshold
      SAA_CAUSALITY_VIOLATED   任一共享关节维度 state 领先 action

    "任意一个维度"而不是"超过百分之多少的维度": 一个维度不对就说明这条 episode 的时序
    有问题, 没有理由认为其他维度的数据可信。

    state / action: (T, D) 矩阵, 由 extract_saa_matrices() 得到。
    saa_profile: profile["saa"], 结构见 saa.md §3.1。必须含 shared_dims。
        enabled=False 的数据集**不应该调用本函数** —— 调用方要把"跳过"单独记账,
        否则"零失败"和"根本没跑"在报告里长得一模一样。
    fps: 帧率, 用于把按秒给的参数换算成帧。

    可能抛 CheckError, 表示**检测没能完成**; 调用方必须记为"检测失败", 不是"通过"。
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

    if not saa_profile:
        raise UnsupportedDataError(
            "未提供 saa profile。维度配对必须由 profile 显式给出 —— 旧版从列名推断的做法"
            "已被实测证伪 (d0..d13 这类命名里没有语义, 夹爪豁免完全失效)")
    if saa_profile.get("enabled") is False:
        raise UnsupportedDataError(
            f"profile 标记本数据集不做 SAA (reason: {saa_profile.get('reason', '未说明')})。"
            f"调用方应把它记为'跳过'并单独计数, 不能当成通过")

    declared = str(saa_profile.get("action_space", "")).lower()
    if declared not in ("absolute", "delta"):
        raise UnsupportedDataError(
            f"profile 的 action_space 必须是 'absolute' 或 'delta', 收到 {declared!r}。"
            f"搞错会让 DA 掉到 0.5 (实测 delta 未积分时 DA=0.496)")

    if "da_threshold" in saa_profile:
        p["da_threshold"] = float(saa_profile["da_threshold"])
    if not (0.0 < p["da_threshold"] <= 1.0):
        raise UnsupportedDataError(f"da_threshold 必须落在 (0, 1], 收到 {p['da_threshold']}")

    if not fps or fps <= 0:
        raise UnsupportedDataError(
            f"拿不到有效帧率 (fps={fps}), 按秒给的 lag / 平滑窗口无法换算成帧")

    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.ndim != 2 or action.ndim != 2:
        raise UnsupportedDataError(
            f"state / action 必须是二维矩阵, 收到 {state.shape} / {action.shape}")
    if state.shape[0] != action.shape[0]:
        raise UnsupportedDataError(
            f"state 与 action 行数不一致 ({state.shape[0]} vs {action.shape[0]}), 不做截断对齐")

    raw_dims = saa_profile.get("shared_dims") or []
    if not raw_dims:
        raise UnsupportedDataError(
            "profile 的 shared_dims 为空, 没有可检测的共享关节维度。"
            "本数据集应在 profile 里标 enabled=false 并由调用方记为跳过")

    pairs: List[Tuple[str, int, int]] = []
    seen_names = set()
    for i, d in enumerate(raw_dims):
        try:
            si, ai = int(d["state_dim"]), int(d["action_dim"])
        except (KeyError, TypeError, ValueError) as e:
            raise UnsupportedDataError(
                f"shared_dims[{i}] 缺少 state_dim / action_dim 或不是整数: {d!r}") from e
        name = str(d.get("name") or f"dim_{si}_{ai}")
        if not (0 <= si < state.shape[1]):
            raise UnsupportedDataError(
                f"shared_dims[{i}] ({name}) 的 state_dim={si} 超出实际 state 维度 "
                f"[0, {state.shape[1]})。profile 与数据不匹配")
        if not (0 <= ai < action.shape[1]):
            raise UnsupportedDataError(
                f"shared_dims[{i}] ({name}) 的 action_dim={ai} 超出实际 action 维度 "
                f"[0, {action.shape[1]})。profile 与数据不匹配")
        if name in seen_names:
            raise UnsupportedDataError(f"shared_dims 里维度名重复: {name!r}")
        seen_names.add(name)
        pairs.append((name, si, ai))

    T = state.shape[0]
    win = _sg_window(p, fps)
    min_len = max(win + 2, 2 * int(round(p["max_lag_sec"] * fps)) + 4)
    if T < min_len:
        raise UnsupportedDataError(
            f"episode 只有 {T} 帧, 不足以做平滑与 lag 搜索 (至少需要 {min_len} 帧)")

    report = SAAReport(n_frames=T, fps=float(fps),
                       action_space_declared=declared, params=dict(p))
    # 低帧率下窗口会被下限顶上去 (fps=20 时 0.17s 只有 3.4 帧 -> 夹到 5 帧 = 0.25s),
    # 实际时间常数与请求值不符。窗口越大趋势线越平滑, DA 跟着变, 所以两个值都要留痕。
    report.params["smooth_window_frames_effective"] = win
    report.params["smooth_window_sec_effective"] = round(win / float(fps), 4)

    # ---- action_space 复验: profile 声明与实测不符就报错, 不自己改主意 ----
    measured, evidence = _measure_action_space(state, action, pairs)
    report.action_space_measured = measured
    report.action_space_evidence = evidence
    if measured != declared:
        raise UnsupportedDataError(
            f"profile 声明 action_space={declared}, 但实测像 {measured} "
            f"(证据: {evidence})。一条 episode 的动作空间不可能和数据集其他部分不同, "
            f"要么 profile 定错了, 要么这条数据本身有问题 —— 两种都该停下来查, "
            f"不能让检测自己改主意")

    # ---- 逐维度测量 ----
    for name, si, ai in pairs:
        s_dim = state[:, si]
        a_dim = action[:, ai]
        if declared == "delta":
            a_dim = _integrate(a_dim, float(s_dim[0]))
        report.dims[name] = measure_dim(s_dim, a_dim, name, si, ai, float(fps), p)

    # ---- 判定 ----
    tol = int(p["causality_tol_frames"])
    violated = {n: d for n, d in report.dims.items() if d.best_lag_frames < -tol}
    if violated:
        report.findings.append(_finding(
            "SAA_CAUSALITY_VIOLATED", dims=sorted(violated),
            evidence={
                "criterion": "best_lag_frames >= -causality_tol_frames",
                "causality_tol_frames": tol,
                "max_lag_sec": p["max_lag_sec"],
                "n_dims_checked": len(report.dims),
                "n_dims_violated": len(violated),
                "per_dim_violated": {n: d.to_dict() for n, d in sorted(violated.items())},
                "per_dim_best_lag": {n: d.best_lag_frames
                                     for n, d in sorted(report.dims.items())},
                # cross_corr 必须和 best_lag 并排看: 相关系数接近 0 时 lag 是在拟合噪声,
                # "state 领先 N 帧"这个说法就不成立了 —— 那种情况多半是 action 信号本身
                # 与 state 无关 (同时也会触发 SAA_DA_BELOW_THRESHOLD)。
                "per_dim_cross_corr": {n: round(d.cross_corr, 6)
                                       for n, d in sorted(report.dims.items())},
            }))

    failed = {n: d for n, d in report.dims.items() if d.da < p["da_threshold"]}
    if failed:
        report.findings.append(_finding(
            "SAA_DA_BELOW_THRESHOLD", dims=sorted(failed),
            evidence={
                "criterion": "min(DA over shared dims) >= da_threshold",
                "threshold": p["da_threshold"],
                "action_space": declared,
                "n_dims_checked": len(report.dims),
                "n_dims_failed": len(failed),
                "per_dim_failed": {n: d.to_dict() for n, d in sorted(failed.items())},
                # 没超标的维度也全列出来: "只有这一维特别差"和"整体都在阈值附近晃"
                # 是两种完全不同的问题, 只列失败维度就分不出来
                "per_dim_da": {n: round(d.da, 6) for n, d in sorted(report.dims.items())},
            }))

    return report


def _finding(rule: str, dims: Optional[List[str]] = None,
             evidence: Optional[dict] = None) -> Finding:
    desc, disp = RULES[rule]
    return Finding(rule=rule, description=desc, disposition=disp,
                   dims=dims or [], evidence=evidence or {})
