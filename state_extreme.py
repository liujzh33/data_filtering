"""极值过滤 (stage 2)。

Qwen 的做法:

  "Frames with state or action values outside the expected range are removed to prevent
   distortion of the quantile-based normalization ([q01, q99] -> [-1, 1]). Per-dimension
   q1 and q99 percentiles are computed PER EMBODIMENT TYPE, and frames outside the band
   [q1 - alpha*(q99-q1), q99 + alpha*(q99-q1)] are excluded. Gripper dimensions are exempt
   due to their bimodal distributions."

**目的那半句是关键: 防的是"归一化被离群值带偏", 不是修剪正常分布的尾巴。** 旧实现
(new/state_extreme_frame.py) 偏离这个目的很远, 完整实测证据见 state_extreme.md,
这里只记结论。

--------------------------------------------------------------------------
alpha 必须是活参数
--------------------------------------------------------------------------

旧版的百分位文件里存的是**已经按 alpha=3 算好的 lo/hi**, 而代码直接取这两个字段,
`t["alpha"]` 只在退化分支里用得到。于是 table.txt(52) 要求的"测 alpha=0/0.5/1/2/3"
根本做不了 —— 换一个 alpha 得重算整份 json。

这一版 profile 里存 **q1/q99**, band 在运行时按 alpha 算。alpha 是唯一参数。

实测 alpha 的影响 (7 维 action joint, 200 条工作区中心各不相同的正常 episode,
另注入 20 倍量程的真实毛刺):

    alpha   总删帧率   命中>50%的episode   真离群值检出率
      0      13.43%        10.5%             100%
      0.5     0.08%         0.0%             100%
      1       0.08%         0.0%             100%
      3       0.08%         0.0%             100%

alpha 从 0 抬到 0.5, 删帧率掉 170 倍而**真离群值检出率一点没变**。alpha=0 抓的那 13%
全是正常分布的尾巴。校准直觉: 正态下 q1/q99 是 ±2.33σ, alpha=1 的 band 已经是 ±6.98σ,
够得着的只有数量级上的错误 (毛刺 / 单位搞错 / 坐标系错误) —— 那正是要防的东西。

--------------------------------------------------------------------------
"alpha=0 只删 2%" 不成立
--------------------------------------------------------------------------

两个原因叠加:

其一, 判定是各维度的**并集**。单维 2%, 7 维独立时 13.2%, 14 维 24.6%。

其二, 更要命: **q1/q99 是数据集级统计量, 单条 episode 的命中率方差极大**。同一份模拟
数据 (alpha=0): 中位数 episode 命中 2.2%, 但第 90 百分位是 54.5%, 第 99 百分位 94.8%。
10.5% 的 episode 会被删掉一半以上的帧, 而它们数据完全正常 —— 只是在工作区边缘作业。

所以本模块输出 flagged_ratio, 由流水线层的删除预算去发现这类分布异常。**单个检测不
越权改判 episode** (table.txt 53: 只做帧级)。

--------------------------------------------------------------------------
维度由 profile 给出, 手爪是"不进列表"而不是"运行时豁免"
--------------------------------------------------------------------------

旧版靠子串匹配认手爪:

    if any(k in low for k in ("gripper", "finger", "effector", "jaw")):

两个坑: `"effector"` 是 `"end_effector"` 的子串, 任何把末端位姿命名成
`end_effector_pose_*` 的数据集会把末端位姿整体当手爪静默豁免; 而 d0..d13 那种布局
名字里没有语义, 豁免完全失效 (和 SAA 旧版同一个坑)。

这一版只检 profile 里列出的维度 —— 按你的要求是 **action 的关节维度**。手爪、末端
位姿、速度、力矩根本不出现在列表里。豁免变成一个配置事实, 不是一处名字匹配。

profile 还要声明 source (state / action): 旧版模块提取 state, 而百分位文件由
calc_action_percentiles.py 产出 —— **用 action 的分布去卡 state 的值**, 两者隔着
跟踪误差和控制滞后, 分布不重合, 而输出里完全看不出来。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------

    from check_pipeline.signals import extract_signal_matrices
    from check_pipeline.state_extreme import check_state_extreme

    signals = extract_signal_matrices(parquet_bytes, columns=["action"])
    report = check_state_extreme(signals, se_profile=profile["state_extreme"])
    if report.drop_frames:
        ranges = report.frames_to_drop        # [(start, end), ...] 左闭右开, 无 margin

profile["state_extreme"] 的结构见 state_extreme.md §3.1。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import PipelineEnvironmentError, UnsupportedDataError
from .intervals import indices_to_ranges, merge_ranges, ranges_length

__all__ = [
    "DEFAULT_PARAMS",
    "RULES",
    "Finding",
    "DimExtreme",
    "StateExtremeReport",
    "measure_dim",
    "check_state_extreme",
]


DEFAULT_PARAMS = {
    "alpha": 1.0,
    #   **唯一参数**。band = [q1 - alpha*(q99-q1), q99 + alpha*(q99-q1)]。
    #   alpha=0 就是 [q1, q99] 本身, 实测会删掉 13% 的正常帧 (见模块 docstring);
    #   alpha=1 的 band 在正态下相当于 ±6.98σ, 只有数量级上的错误够得着。
    #   调法: 跑一批看 per_dim 的 alpha_needed_to_clear 分布, 想放过哪一档就调到哪。
}

# 没有 lo_q / hi_q: PDF 把 q01/q99 写死了, 做成参数只会让不同批次的结果不可比。
#   真要换定义, 该换的是 profile 里 q1/q99 的算法, 不是这里。
# 没有四档严重度比例阈值 (extreme_mild/moderate/severe/critical_ratio):
#   只做帧级筛选, 不存在分档。

DROP_FRAMES = "drop_frames"

RULES: Dict[str, Tuple[str, str]] = {
    "STATE_EXTREME_FRAMES": (
        "存在维度的值落在 [q1-α·span, q99+α·span] 之外 (离群值会带偏分位数归一化)",
        DROP_FRAMES),
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
class DimExtreme:
    """一个被检维度的测量结果。**不含判定**。"""
    name: str = ""
    column: str = ""
    dim: int = -1
    q1: float = 0.0
    q99: float = 0.0
    lo: float = 0.0
    hi: float = 0.0
    n_flagged: int = 0
    frame_ranges: List[Tuple[int, int]] = field(default_factory=list)
    episode_min: float = 0.0
    episode_max: float = 0.0
    episode_median: float = 0.0
    alpha_needed_to_clear: float = 0.0
    #   把 alpha 调到这个值以上, 本维度在这条 episode 上就不再贡献任何命中。
    #   = max over frames of (超出 [q1,q99] 的距离) / (q99-q1)。
    #   直接支撑"跑一批再调 alpha"的闭环: 看这个数的分布就知道 alpha 该定在哪。

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "column": self.column,
            "dim": self.dim,
            "q1": round(self.q1, 9),
            "q99": round(self.q99, 9),
            "lo": round(self.lo, 9),
            "hi": round(self.hi, 9),
            "n_flagged": self.n_flagged,
            "frame_ranges": [list(r) for r in self.frame_ranges],
            "episode_min": round(self.episode_min, 9),
            "episode_max": round(self.episode_max, 9),
            "episode_median": round(self.episode_median, 9),
            "alpha_needed_to_clear": round(self.alpha_needed_to_clear, 6),
        }


@dataclass
class StateExtremeReport:
    findings: List[Finding] = field(default_factory=list)
    dims: Dict[str, DimExtreme] = field(default_factory=dict)
    n_frames: int = 0
    flagged_ranges: List[Tuple[int, int]] = field(default_factory=list)
    source: str = ""
    percentile_scope: str = ""
    params: dict = field(default_factory=dict)

    @property
    def has_extreme(self) -> bool:
        return bool(self.findings)

    @property
    def drop_frames(self) -> bool:
        return any(f.disposition == DROP_FRAMES for f in self.findings)

    @property
    def drop_episode(self) -> bool:
        """恒为 False。本检测只做帧级筛选 (table.txt 53)。

        字段保留是为了让调用方能用统一的方式处理所有检测的报告; 想拦"整条被删太多"
        的情况应该在流水线的删除预算层做, 见模块 docstring。
        """
        return False

    @property
    def frames_to_drop(self) -> List[Tuple[int, int]]:
        """真正要删的帧区间 (左闭右开, **不加 margin**)。"""
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
    def n_frames_flagged(self) -> int:
        return ranges_length(self.flagged_ranges)

    @property
    def flagged_ratio(self) -> float:
        return self.n_frames_flagged / self.n_frames if self.n_frames else 0.0

    def to_dict(self) -> dict:
        return {
            "check": "state_extreme",
            "has_extreme": self.has_extreme,
            "drop_episode": self.drop_episode,
            "drop_frames": self.drop_frames,
            "frames_to_drop": [list(r) for r in self.frames_to_drop],
            "triggered_rules": self.triggered_rules,
            "findings": [f.to_dict() for f in self.findings],
            "measurements": {
                "n_frames": self.n_frames,
                "n_dims_checked": len(self.dims),
                "n_frames_flagged": self.n_frames_flagged,
                "flagged_ratio": round(self.flagged_ratio, 6),
                "flagged_ranges": [list(r) for r in self.flagged_ranges],
                "source": self.source,
                "percentile_scope": self.percentile_scope,
                "params": self.params,
                "per_dim": {n: d.to_dict() for n, d in sorted(self.dims.items())},
            },
        }


# =====================================================================
# 测量
# =====================================================================
def measure_dim(x, name: str, column: str, dim: int,
                q1: float, q99: float, alpha: float) -> DimExtreme:
    """测一个维度。**不做任何判定** —— 规则由 check_state_extreme 决定。

    q1 / q99 来自 profile (扫描阶段对该数据集该机型算的), 不是这条 episode 自己算的。
    旧版在没有 percentile 时会退化成"用本条 episode 的 q1/q99 卡本条 episode",
    那个分支在数学上是空转: alpha=0 时按定义恒命中 2%, alpha=3 时恒命中 0,
    结果由 alpha 决定而不由数据决定, 一条完美的 episode 和一条充满毛刺的得到同样结论。
    """
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        n_bad = int((~np.isfinite(x)).sum())
        raise UnsupportedDataError(
            f"维度 {name} 有 {n_bad} 帧是 NaN/Inf。旧版把它直接喂给 np.percentile, "
            f"band 变成 NaN 之后所有比较恒为 False —— 该维度一帧都不会被标记, "
            f"而输出里看不出发生过这件事。缺失值该由上游先处理掉")

    span = float(q99) - float(q1)
    lo = float(q1) - float(alpha) * span
    hi = float(q99) + float(alpha) * span

    out = DimExtreme(name=name, column=column, dim=dim,
                     q1=float(q1), q99=float(q99), lo=lo, hi=hi)
    out.episode_min = float(np.min(x))
    out.episode_max = float(np.max(x))
    out.episode_median = float(np.median(x))

    flag = (x < lo) | (x > hi)
    out.n_flagged = int(flag.sum())
    out.frame_ranges = indices_to_ranges(np.flatnonzero(flag).tolist())

    # 每帧越过 [q1, q99] 的距离 (以 span 为单位), 取最大 -> 想清空本维度需要的 alpha
    excess = np.maximum(np.maximum(float(q1) - x, x - float(q99)), 0.0)
    out.alpha_needed_to_clear = float(np.max(excess)) / span if span > 0 else 0.0
    return out


# =====================================================================
# 判定
# =====================================================================
def check_state_extreme(signals: Dict[str, Any],
                        se_profile: dict,
                        params: Optional[dict] = None,
                        ) -> StateExtremeReport:
    """对一条 episode 跑极值过滤。

    **一条规则, 只删帧** (table.txt 53):
      STATE_EXTREME_FRAMES   任一被检维度的值落在 band 外 -> drop_frames

    命中帧是所有维度的**并集**: 某一维度取值离谱时, 该时间步整帧数据都不该进归一化统计。

    signals: {列名: (T, D) 矩阵}, 由 signals.extract_signal_matrices() 得到。
    se_profile: profile["state_extreme"], 结构见 state_extreme.md §3.1。
        enabled=False 的数据集**不应该调用本函数** —— 调用方要把"跳过"单独记账,
        否则"零极值"和"根本没跑"在报告里长得一模一样。

    不需要 fps: 本检测的判据里没有任何按时间定义的量。

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

    if not se_profile:
        raise UnsupportedDataError(
            "未提供 state_extreme profile。被检维度和 q1/q99 都必须由 profile 显式给出 —— "
            "旧版靠子串匹配认手爪 ('effector' 是 'end_effector' 的子串), 且在 d0..d13 "
            "这类命名下完全失效")
    if se_profile.get("enabled") is False:
        raise UnsupportedDataError(
            f"profile 标记本数据集不做极值过滤 (reason: {se_profile.get('reason', '未说明')})。"
            f"调用方应把它记为'跳过'并单独计数, 不能当成通过")

    if "alpha" in se_profile:
        p["alpha"] = float(se_profile["alpha"])
    if not (p["alpha"] >= 0) or not np.isfinite(p["alpha"]):
        raise UnsupportedDataError(
            f"alpha 必须是非负有限数, 收到 {p['alpha']}")

    source = str(se_profile.get("source", "")).lower()
    if source not in ("state", "action"):
        raise UnsupportedDataError(
            f"profile 的 source 必须是 'state' 或 'action', 收到 {source!r}。"
            f"这份 q1/q99 是从哪种信号算出来的必须写清楚 —— 旧版拿 action 的分位数去卡 "
            f"state 的值, 两者隔着跟踪误差和控制滞后, 分布不重合, 而输出里看不出来")
    scope = str(se_profile.get("percentile_scope", ""))
    if not scope:
        raise UnsupportedDataError(
            "profile 缺少 percentile_scope (这份分位数属于哪个数据集/机型)。"
            "传错机型的后果是用错误的 band 卡整条数据, 必须可追溯")

    raw_dims = se_profile.get("dims") or []
    if not raw_dims:
        raise UnsupportedDataError(
            "profile 的 dims 为空, 没有可检测的维度。本数据集应在 profile 里标 "
            "enabled=false 并由调用方记为跳过")

    if not signals:
        raise UnsupportedDataError("未提供任何信号矩阵")
    n_frames_set = {int(np.asarray(m).shape[0]) for m in signals.values()}
    if len(n_frames_set) > 1:
        raise UnsupportedDataError(f"各列行数不一致: {sorted(n_frames_set)}, 不做截断对齐")
    n_frames = n_frames_set.pop()
    if n_frames == 0:
        raise UnsupportedDataError("信号长度为 0")

    # ---- 解析 profile 里的维度声明 ----
    specs: List[Tuple[str, str, int, float, float]] = []
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
            raise UnsupportedDataError(f"列 {column!r} 必须是二维矩阵, 收到 {mat.shape}")
        if not (0 <= dim < mat.shape[1]):
            raise UnsupportedDataError(
                f"dims[{i}] ({name}) 的 dim={dim} 超出列 {column!r} 的实际维度 "
                f"[0, {mat.shape[1]})。profile 与数据不匹配")
        for key in ("q1", "q99"):
            if key not in d:
                raise UnsupportedDataError(
                    f"dims[{i}] ({name}) 缺少 {key}。profile 里要存 q1/q99 而不是算好的 "
                    f"lo/hi —— 存 lo/hi 等于把 alpha 烘死, 换一个 alpha 得重算整份文件, "
                    f"table.txt(52) 要求的 alpha 扫描就做不了了")
        try:
            q1, q99 = float(d["q1"]), float(d["q99"])
        except (TypeError, ValueError) as e:
            raise UnsupportedDataError(
                f"dims[{i}] ({name}) 的 q1/q99 不是数: {d['q1']!r} / {d['q99']!r}") from e
        if not (np.isfinite(q1) and np.isfinite(q99)):
            raise UnsupportedDataError(f"dims[{i}] ({name}) 的 q1/q99 非有限: {q1} / {q99}")
        if not (q99 > q1):
            raise UnsupportedDataError(
                f"dims[{i}] ({name}) 的 q99={q99} 不大于 q1={q1}, band 宽度非正。"
                f"这个维度在整个数据集上都没动过, 它不该出现在 dims 里")
        specs.append((name, column, dim, q1, q99))

    report = StateExtremeReport(n_frames=n_frames, source=source,
                                percentile_scope=scope, params=dict(p))

    # ---- 逐维度测量 ----
    flagged = np.zeros(n_frames, dtype=bool)
    for name, column, dim, q1, q99 in specs:
        mat = np.asarray(signals[column], dtype=np.float64)
        m = measure_dim(mat[:, dim], name, column, dim, q1, q99, p["alpha"])
        report.dims[name] = m
        for a, b in m.frame_ranges:
            flagged[a:b] = True

    # ---- profile 与数据是否对得上 ----
    _cross_check_percentiles(report, scope)

    report.flagged_ranges = indices_to_ranges(np.flatnonzero(flagged).tolist())
    total = ranges_length(report.flagged_ranges)
    if total == 0:
        return report

    hit = {n: d for n, d in report.dims.items() if d.n_flagged}
    report.findings.append(_finding(
        "STATE_EXTREME_FRAMES", dims=sorted(hit),
        frame_ranges=report.flagged_ranges,
        evidence={
            "criterion": "值落在 [q1 - α·(q99-q1), q99 + α·(q99-q1)] 之外",
            "alpha": p["alpha"],
            "source": source,
            "percentile_scope": scope,
            "n_frames": n_frames,
            "n_frames_flagged": total,
            # 流水线的删除预算靠这个数发现"某条 episode 被删太多": q1/q99 是数据集级
            # 统计量, 单条 episode 的命中率方差极大 (实测 alpha=0 时中位数 2.2% 而第 90
            # 百分位 54.5%)。本检测不改判 episode, 但必须把这个数报出去。
            "flagged_ratio": round(total / n_frames, 6),
            "n_dims_checked": len(report.dims),
            "n_dims_hit": len(hit),
            "n_segments": len(report.flagged_ranges),
            # 极值命中通常是散点而不是连续段, 这点和 sudden_change 很不一样。段数多意味着
            # 删完之后 episode 上会出现同样多处时间不连续, 下游据此决定截断/切分/放弃。
            "per_dim_n_flagged": {n: d.n_flagged for n, d in sorted(report.dims.items())},
            # 想让某个维度不再贡献命中, alpha 要调到多大 —— 支撑"跑一批再调"的闭环
            "per_dim_alpha_needed_to_clear": {
                n: round(d.alpha_needed_to_clear, 6)
                for n, d in sorted(report.dims.items())},
            "per_dim_hit": {n: d.to_dict() for n, d in sorted(hit.items())},
        }))
    return report


def _cross_check_percentiles(report: StateExtremeReport, scope: str) -> None:
    """profile 里的 q1/q99 是不是属于当前这批数据。

    传错机型的分位数, 后果是用错误的 band 卡整条数据, 而且看起来一切正常。这里做一道
    很便宜的自检: 正常情况下一条 episode 的**中位数**应该稳稳落在数据集的 [q1, q99]
    区间内 (那个区间覆盖了全数据集 98% 的帧)。大面积落在外面只有一种解释。

    维度数 < 3 时不做这个检查: 那种情况下分不清"分位数拿错了"和"这条 episode 罕见"。
    """
    dims = list(report.dims.values())
    if len(dims) < 3:
        return
    outside = [d for d in dims if d.episode_median < d.q1 or d.episode_median > d.q99]
    if len(outside) * 2 > len(dims):
        detail = {d.name: {"episode_median": round(d.episode_median, 6),
                           "q1": round(d.q1, 6), "q99": round(d.q99, 6)}
                  for d in outside}
        raise UnsupportedDataError(
            f"{len(outside)}/{len(dims)} 个维度的 episode 中位数落在 [q1, q99] 之外 "
            f"(percentile_scope={scope!r})。[q1, q99] 覆盖全数据集 98% 的帧, 一条 episode "
            f"的中位数大面积落在外面, 几乎只能是这份分位数不属于当前机型/数据集。"
            f"先核对 profile 的 percentile_scope 再跑。明细: {detail}")


def _finding(rule: str, dims: Optional[List[str]] = None,
             frame_ranges: Optional[List[Tuple[int, int]]] = None,
             evidence: Optional[dict] = None) -> Finding:
    desc, disp = RULES[rule]
    return Finding(rule=rule, description=desc, disposition=disp,
                   dims=dims or [], frame_ranges=merge_ranges(frame_ranges or []),
                   evidence=evidence or {})
