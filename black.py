"""black — 黑屏检测 (重写版)。

只回答一个问题: **画面是不是丢了信号**(相机掉线 / 镜头被完全挡死 / 编码器吐黑帧)。
不判断"画面偏暗"—— 暗但有内容的画面标注照样能用, 那是曝光问题, 不是黑屏。

判据: 黑帧 ⟺ 又暗**又平**

    gray_mean < mean_abs   AND   gray_std < std_abs

两个条件取**与**, 不是或。理由是黑屏的物理本质是"没有信号", 表现为所有像素都一样,
所以"纹理"(标准差) 才是决定性的:

    真黑屏 (相机掉线)      mean ≈ 0-5    std ≈ 0
    暗场景 (夜间 / 关灯)   mean 20-40    std 15-30
    镜头被完全蒙住          mean 可能 30   std ≈ 0

旧版只看 gray_mean, 判据是 `mean < 10 OR mean < 自己的75%分位 × 0.3`, 两个方向都错:

  误报: 正常亮度 120 的视频, 相对阈值 = 36。画面变暗到 35 (关了灯 / 机械臂靠近镜头)
        内容还在、标注能用, 却被判成黑屏删掉。这条规则实际检测的是"比自己平时暗",
        不是"黑屏", 两件事被混成了一条。

  漏报: 相机整条 episode 亮度稳定在 12 (对人眼已经是黑的)。绝对阈值 10 抓不到
        (12 > 10, 这是个硬悬崖: 9 判黑并可能删整条, 12 完全放行); 相对阈值的基线是
        **它自己**的 75% 分位 = 12, 阈值 3.6, 也抓不到。
        **自参照基线的通病: 异常如果是全程的, 它就变成了"正常"。**
        新判据下 12 < 16 且 std ≈ 0 -> 每帧都是黑帧 -> VIEW_MOSTLY_BLACK -> 丢整条。

用 std 还顺带绕开一个坑: 视频有 limited range (Y 16-235) 和 full range (0-255) 两种,
color_range 元数据缺失或转换没做对时, 真黑屏的均值会落在 16 而不是 0, 旧版 abs=10
完全抓不到。**标准差不受这个偏移影响**, 这也是 mean_abs 取 18 而不是 10 的原因
(必须严格大于 16, 见 DEFAULT_PARAMS 里的详细说明)。

**与 occlusion 的分工**: 两者用**同一个** std_abs 判"画面平不平", 再按 mean_abs 分家:

    std < std_abs  且  mean <  mean_abs   ->  black       (又平又暗 = 丢了信号)
    std < std_abs  且  mean >= mean_abs   ->  occlusion   (平但不暗 = 镜头被挡住/糊住)
    std >= std_abs                        ->  两者都不管   (画面有内容)

这样任何"平"的帧一定落进其中之一, 不留缝隙也不重叠。旧版 black 用 std_abs=2、
occlusion 用 8, 中间 2~8 且暗的帧两边都不管, 是个漏检缝隙。
black 仍然把"平但不暗"的帧数测出来 (ViewBlack.n_flat_not_dark), 便于对账。

规则表 (rule 字段的取值)。**只有一条, 结果只有丢整条**:

  BLACK_RUN_RATIO_EXCEEDED   任意一路相机的**最长连续**黑帧段 >= max_black_run_ratio
                             (默认 10%) -> drop_episode

判据是**最长连续段**, 不是黑帧总量。要拦的是"画面持续黑了一段时间"这种相机掉线/断流,
不是"零星黑帧的累计量":

    1500 帧的 episode, 散布 150 个孤立黑帧 (总量 10%, 最长段 1 帧) -> 不触发
    1500 帧的 episode, 连续黑 150 帧      (总量 10%, 最长段 150)   -> 触发

总量仍然测出来放进 measurements (black_ratio), 但**不参与判定**。两个数差很多就说明
黑帧是零星散布的 —— 那种情况现在会放过, 见下面"已知取舍"。

"任意一路"而不是"所有路": 视角数量各数据集/机型不同 (2 路 / 3 路 / 6 路都有),
只要有一路瞎了, 这条数据就不完整。分母是该路**有测量值的帧数** (不含 NaN)。

**已知取舍**: 闪烁型故障会被放过。一路相机每 10 帧闪一次黑、闪满全片, 总量 10% 但
最长连续段只有 1 帧, 不触发。如果实际数据里这种故障不少, 需要另加一条按总量判的规则。
第一批跑完对比 measurements 里的 black_ratio 和 longest_black_run_ratio 就能看出来。

相比旧版去掉的东西:

  - **四档严重度** (mild/moderate/severe/critical)。旧版 severe 档的 action 是
    "split_or_drop_episode", 而代码注释自己承认"当前无自动清洗分支 (留人工)" ——
    跑十万条没人会人工看, severe 实际等于静默保留。

  - **按"同时黑几路"缩放时长阈值的 3x4 表**。这 12 个数字之间没做过一致性检查:
    实测 35 个时长里有 11 个出现"路数越多反而判得越轻"(0.13-0.47 秒区间, 3 路全黑
    判 moderate, 1 路黑反而判 severe), 与设计意图完全相反。而且档间只差一帧
    (29 帧 severe = 保留, 30 帧 critical = 删整条)。

  - **相对阈值 (自参照分位数基线)**。见上面的漏报分析。

  - **exclude_mask**。坏帧在 probe 里就是 NaN, 天然跳过, 不需要外挂 mask。旧机制有两
    个毛病: 无 frame_range 的 Issue 会把整路相机整列抹掉 (该路永远不再报黑屏);
    而且 mask 减完之后才统计"同时黑几路", 于是 corrupt 的一条 issue 会悄悄改变 black
    对**其他**相机的分级结果。

  - **删帧、段长阈值、首尾边缘豁免、margin**。现在按"整条 episode 的黑帧占比"判定,
    与黑帧分布在哪里无关, 这几样都失去了意义。副作用: 占比不到 10% 的黑帧会**原样保留**
    在数据里, 不再被抠掉。

参数从 21 个降到 3 个 (mean_abs / std_abs / max_black_run_ratio), 每个都有明确物理含义。

**报错 vs 筛选**: 见 errors.py。本模块的报错项:
  - probe 没带 gray_mean / gray_std      -> UnsupportedDataError (新判据必须要 std)
  - 拿不到 fps                            -> UnsupportedDataError (秒级阈值无法换算)
  - 各相机时间轴长度不等                   -> UnsupportedDataError (corrupt 该先拦下)
  - 一路可用的相机都没有                   -> UnsupportedDataError (不能返回"干净")
  - numpy 不可用                          -> PipelineEnvironmentError

用法:
    from check_pipeline.probe import probe_video
    from check_pipeline.black import check_black
    from check_pipeline.errors import CheckError

    probes = {cam: probe_video(b, cam) for cam, b in videos.items()}
    try:
        report = check_black(probes)
    except CheckError as e:
        ...  # 检测失败, 单独记录; 不是"干净"也不是"有黑屏"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .errors import PipelineEnvironmentError, UnsupportedDataError
from .intervals import (indices_to_ranges, longest_range, longest_range_span,
                        merge_ranges, ranges_length)
from .probe import VideoProbe

logger = logging.getLogger("check_pipeline.black")


# =====================================================================
# 参数 (3 个, 每个都有明确物理含义)
# =====================================================================
DEFAULT_PARAMS = {
    # 黑帧判据 (两个条件取"与")
    "mean_abs": 18.0,
    #   灰度均值上限。取 18 而不是旧版的 10: limited range 视频的黑电平**正好是 16**,
    #   color_range 元数据缺失时真黑屏的均值就落在 16, 阈值必须高于它才抓得到
    #   (取 16 也不行 —— 判据是严格小于, 16 < 16 不成立)。18 留 2 的余量。
    #   这个阈值同时是 black 与 occlusion 的**分界线**: 又平又暗 (mean < 18) 归 black,
    #   又平但不暗 (mean >= 18) 归 occlusion。两边互补, 不留缝隙也不重叠。
    "std_abs": 8.0,
    #   灰度标准差上限, 这是"画面没有内容"的**决定性判据**。真黑屏所有像素相同 -> std≈0;
    #   暗场景仍有纹理 -> std 15-30。与 occlusion 用同一个值 (见上一条的分工说明)。

    # 删除判据
    "max_black_run_ratio": 0.10,
    #   **唯一的删除规则**: 任意一路相机的**最长连续**黑帧段达到这个占比 -> 丢弃整条。
    #
    #   注意是**最长连续段**, 不是黑帧总量。要拦的是"画面持续黑了一段时间"这种
    #   相机掉线/断流, 不是"零星黑帧的累计量"。1500 帧的 episode 里散布 150 个孤立
    #   黑帧 (总量 10%) 不触发; 连续黑 150 帧 (最长段 10%) 触发。
    #
    #   "任意一路"而不是"所有路": 视角数量各数据集/机型不同, 只要有一路瞎了这条数据
    #   就不完整。分母是该路**有测量值的帧数** (不含 NaN, 那些是 corrupt 的事)。
}


# =====================================================================
# 规则表: rule -> (中文说明, 处理方式)
# =====================================================================
DROP_EPISODE = "drop_episode"

RULES: Dict[str, Tuple[str, str]] = {
    "BLACK_RUN_RATIO_EXCEEDED": (
        "存在相机的最长连续黑帧段达到阈值占比 (画面持续又暗又平 = 丢了信号)",
        DROP_EPISODE),
}


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class Finding:
    """一条命中的规则。JSON 里每条 issue 就是这个结构。与 corrupt.Finding 同构。"""
    rule: str
    description: str
    disposition: str                            # 当前恒为 drop_episode (字段保留以统一 JSON 结构)
    views: List[str] = field(default_factory=list)
    frame_ranges: List[Tuple[int, int]] = field(default_factory=list)  # [start, end) 逐段
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
class ViewBlack:
    """单路相机的黑帧测量结果。只描述"测到了什么", 不做判定。

    所有字段都是纯 Python 标量/列表, 规则判定不碰 numpy —— 便于单独测试规则逻辑。
    """
    cam: str
    n_frames: int = 0                 # 时间轴长度
    n_valid: int = 0                  # 有测量值的帧数 (非 NaN); NaN 是 corrupt 的坏帧
    n_black: int = 0
    black_ranges: List[Tuple[int, int]] = field(default_factory=list)
    longest_black_run: int = 0                       # 最长连续黑帧段的长度 (判定用)
    longest_black_run_ratio: float = 0.0             # 它占 n_valid 的比例 (判定用)
    longest_black_run_span: Optional[Tuple[int, int]] = None   # 它的位置 (排查用)
    black_ratio: float = 0.0          # n_black / n_valid
    # 只满足 std 条件、不满足 mean 条件的帧: 画面是**均匀的一片**但不暗 (镜头被浅色物体
    # 挡死、白平衡失控整片发白)。按分工这属于 occlusion, black 只测量不判定。
    # 两边用同一个 std_abs, 按 mean_abs 分界, 所以不留缝隙: 任何"平"的帧一定归其中之一。
    n_flat_not_dark: int = 0
    flat_not_dark_ratio: float = 0.0
    # 分布 (用于事后把阈值定准, 而不是继续沿用来路不明的常数)
    gray_mean_percentiles: Dict[str, float] = field(default_factory=dict)
    gray_std_percentiles: Dict[str, float] = field(default_factory=dict)
    min_gray_mean: Optional[float] = None
    min_gray_std: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "cam": self.cam,
            "n_frames": self.n_frames,
            "n_valid": self.n_valid,
            "n_black": self.n_black,
            "black_ratio": round(self.black_ratio, 6),
            "n_flat_not_dark": self.n_flat_not_dark,
            "flat_not_dark_ratio": round(self.flat_not_dark_ratio, 6),
            "longest_black_run_ratio": round(self.longest_black_run_ratio, 6),
            "longest_black_run_span": (list(self.longest_black_run_span)
                                       if self.longest_black_run_span else None),
            "n_segments": len(self.black_ranges),
            "longest_black_run": self.longest_black_run,
            "black_ranges": [list(r) for r in self.black_ranges],
            "gray_mean_percentiles": self.gray_mean_percentiles,
            "gray_std_percentiles": self.gray_std_percentiles,
            "min_gray_mean": self.min_gray_mean,
            "min_gray_std": self.min_gray_std,
        }


@dataclass
class BlackReport:
    """一条 episode 的 black 检测结果。to_dict() 即写进 result.json 的内容。"""
    findings: List[Finding] = field(default_factory=list)
    views: Dict[str, ViewBlack] = field(default_factory=dict)
    n_frames: int = 0
    fps: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def has_black(self) -> bool:
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
            "check": "black",
            "has_black": self.has_black,
            "drop_episode": self.drop_episode,
            "triggered_rules": self.triggered_rules,
            "findings": [f.to_dict() for f in self.findings],
            "measurements": {
                "n_frames": self.n_frames,
                "fps": round(self.fps, 4) if self.fps else 0.0,
                "params": self.params,
                "per_view": {c: v.to_dict() for c, v in self.views.items()},
            },
        }


# =====================================================================
# 单路测量 (唯一用到 numpy 的地方)
# =====================================================================
def measure_view(probe: VideoProbe, mean_abs: float, std_abs: float) -> ViewBlack:
    """从一路 probe 的逐帧灰度算出黑帧位置与分布。不做判定。

    NaN 帧 (probe 解不出来的坏帧) 既不算黑也不算正常: 不进 n_black, 也不进 n_valid。
    那些帧是 corrupt 的职责, black 不重复处理, 也不让它们污染黑帧占比。
    """
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - 环境问题
        raise PipelineEnvironmentError(f"numpy 不可用: {e}") from e

    cam = probe.cam
    if probe.gray_mean is None or probe.gray_std is None:
        raise UnsupportedDataError(
            f"[{cam}] probe 没有带逐帧灰度 (gray_mean/gray_std 为 None)。"
            f"black 的判据必须同时要均值和标准差; 请用 probe_video(..., collect_gray=True)")

    gm = np.asarray(probe.gray_mean, dtype=np.float64)
    gs = np.asarray(probe.gray_std, dtype=np.float64)
    if gm.shape != gs.shape:
        raise UnsupportedDataError(
            f"[{cam}] gray_mean 与 gray_std 长度不一致 ({gm.shape} vs {gs.shape})")

    out = ViewBlack(cam=cam, n_frames=int(gm.size))
    valid = ~(np.isnan(gm) | np.isnan(gs))
    out.n_valid = int(valid.sum())
    if out.n_valid == 0:
        # 一帧测量值都没有。这不是"没有黑屏", 是"测不了" —— 交给调用层判断
        # (整路解不出来的情况 corrupt 已经会丢整条, black 不重复报)。
        return out

    # 黑帧 = 又暗又平, 两个条件取"与"
    is_flat = valid & (gs < std_abs)
    is_black = is_flat & (gm < mean_abs)
    out.n_black = int(is_black.sum())
    out.black_ratio = out.n_black / out.n_valid
    out.black_ranges = indices_to_ranges(np.flatnonzero(is_black).tolist())
    out.longest_black_run = longest_range(out.black_ranges)
    out.longest_black_run_ratio = out.longest_black_run / out.n_valid
    out.longest_black_run_span = longest_range_span(out.black_ranges)
    # 平但不暗 -> occlusion 的地盘, 这里只测量不判定 (见 ViewBlack.n_flat_not_dark)
    out.n_flat_not_dark = int(is_flat.sum()) - out.n_black
    out.flat_not_dark_ratio = out.n_flat_not_dark / out.n_valid

    # 分布: 阈值现在是按物理含义拍的, 把实测分布输出来才能事后定准
    vm, vs = gm[valid], gs[valid]
    qs = (1, 5, 25, 50, 75, 95)
    out.gray_mean_percentiles = {f"p{q}": round(float(np.percentile(vm, q)), 3) for q in qs}
    out.gray_std_percentiles = {f"p{q}": round(float(np.percentile(vs, q)), 3) for q in qs}
    out.min_gray_mean = round(float(vm.min()), 3)
    out.min_gray_std = round(float(vs.min()), 3)
    return out


# =====================================================================
# 主检测
# =====================================================================
def check_black(probes: Dict[str, VideoProbe],
                params: Optional[dict] = None,
                ) -> BlackReport:
    """对一条 episode 跑 black 检测。

    **唯一的删除规则**: 任意一路相机的**最长连续**黑帧段 >= max_black_run_ratio
    (默认 10%) -> 丢弃整条 episode。判据是最长连续段, 不是黑帧总量。没有删帧这条路。

    "任意一路"而不是"所有路": 视角数量各数据集/机型不同, 只要有一路瞎了这条数据就不完整。

    probes: {相机名: probe_video(...) 的结果}。必须是 collect_gray=True 解出来的。
        整路缺失/打不开/一帧读不出来的相机会被跳过 —— 那些 corrupt 已经判了丢整条,
        black 不重复报。
    params: 覆盖 DEFAULT_PARAMS 的部分键。传未知键名会报错, 不会静默忽略。

    返回 BlackReport, 用 .to_dict() 直接写 JSON。

    可能抛 CheckError (PipelineEnvironmentError / UnsupportedDataError), 表示**检测没能
    完成**。这类异常不捕获, 调用方必须把该 episode 记为"检测失败", 既不能计入"干净",
    也不能判成"有黑屏"。
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise UnsupportedDataError(
                f"未知参数 {sorted(unknown)}; 可用参数: {sorted(DEFAULT_PARAMS)}")
        p.update(params)

    if not probes:
        raise UnsupportedDataError("未提供任何相机的 probe, 无法检测")

    # 只看整路可用的相机。整路坏掉的 corrupt 已经判丢整条, 这里不重复报。
    usable = {cam: pr for cam, pr in probes.items() if pr.usable}
    if not usable:
        raise UnsupportedDataError(
            f"{len(probes)} 路相机没有一路可用 (缺失/打不开/零可读帧), 无法判断黑屏。"
            f"这条 episode 应由 corrupt 处理, 不能当成'没有黑屏'")

    # fps: 不做 `or 30.0` 的猜测。虽然当前判据只看占比、不需要秒换算, 但 fps 拿不到
    # 说明容器元数据有问题, 同一批 probe 还要喂给 stillness (那里要按秒判段长)。
    fps_values = {round(pr.fps, 3) for pr in usable.values()}
    if not fps_values or 0.0 in fps_values:
        raise UnsupportedDataError(f"存在拿不到帧率的相机 (fps={sorted(fps_values)})")
    if len(fps_values) > 1:
        raise UnsupportedDataError(
            f"各相机帧率不一致: "
            f"{ {c: round(pr.fps, 3) for c, pr in usable.items()} }, 时间轴无法对齐")

    # 时间轴长度必须一致。不一致说明 corrupt 的 VIEW_FRAME_COUNT_MISMATCH 没被处理,
    # 此时"占该条 episode 的 10%"里的分母都说不清是哪个 —— 与其截断到最短
    # (旧版 metrics 的做法, 会悄悄丢掉长的那一路的尾部), 不如停下来。
    lengths = {c: pr.n_frames for c, pr in usable.items()}
    if len(set(lengths.values())) > 1:
        raise UnsupportedDataError(
            f"各相机时间轴长度不一致: {lengths}。应由 corrupt 先判 VIEW_FRAME_COUNT_MISMATCH; "
            f"black 不做截断对齐")

    report = BlackReport(n_frames=next(iter(lengths.values())),
                         fps=next(iter(fps_values)), params=dict(p))

    # ---- 1) 逐路测量 ----
    for cam, pr in usable.items():
        report.views[cam] = measure_view(pr, p["mean_abs"], p["std_abs"])

    # ---- 2) 唯一的规则: 任意一路黑帧占比达标 -> 丢整条 ----
    over = {c: v for c, v in report.views.items()
            if v.n_valid > 0 and v.longest_black_run_ratio >= p["max_black_run_ratio"]}
    if over:
        # 一条 finding 覆盖所有超标的相机: 处理动作只有一个 (丢整条), 拆成多条没有意义,
        # 反而要下游去重。每路的具体数字在 evidence.per_view_over 里。
        report.findings.append(_finding(
            "BLACK_RUN_RATIO_EXCEEDED", views=sorted(over),
            frame_ranges=merge_ranges(
                [r for v in over.values() for r in v.black_ranges]),
            evidence={
                "criterion": "longest_black_run / n_valid",
                "threshold": p["max_black_run_ratio"],
                "mean_abs": p["mean_abs"],
                "std_abs": p["std_abs"],
                "n_views_checked": len(report.views),
                "n_views_over": len(over),
                "per_view_over": {
                    c: {"longest_black_run": v.longest_black_run,
                        "longest_black_run_ratio": round(v.longest_black_run_ratio, 6),
                        "longest_black_run_span": (list(v.longest_black_run_span)
                                                   if v.longest_black_run_span else None),
                        "n_valid": v.n_valid,
                        "n_black_total": v.n_black,
                        "black_ratio_total": round(v.black_ratio, 6),
                        "n_segments": len(v.black_ranges),
                        "min_gray_mean": v.min_gray_mean,
                        "min_gray_std": v.min_gray_std}
                    for c, v in sorted(over.items())},
                "per_view_longest_run_ratio": {
                    c: round(v.longest_black_run_ratio, 6)
                    for c, v in sorted(report.views.items())},
                # 总量只做参考, 不参与判定 —— 它和最长段差很多说明黑帧是零星散布的
                "per_view_black_ratio_total": {
                    c: round(v.black_ratio, 6)
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


# 注: 曾经有一个 to_frame_delete_mask(), 已随"删帧"路径一起移除。
# black 现在只有 drop_episode 一种结果, 不需要帧掩码。
# finding.frame_ranges 保留下来是**诊断信息** (黑在哪里、黑了几段), 不是"要删哪些帧"。
