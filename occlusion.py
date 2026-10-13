"""occlusion — 遮挡检测 (重写版)。

只回答一个问题: **镜头前面是不是被东西挡住了**(手/物料/盖子贴住镜头, 或镜头糊了)。
画面还有亮度, 但没有内容。

判据: 遮挡帧 ⟺ 又平**又不暗**

    gray_std < std_abs   AND   gray_mean >= mean_abs

和 black 共用同一个 std_abs 判"画面平不平", 再按 mean_abs 分家:

    std < std_abs  且  mean <  mean_abs   ->  black       (又平又暗 = 丢了信号)
    std < std_abs  且  mean >= mean_abs   ->  occlusion   (平但不暗 = 被挡住了)
    std >= std_abs                        ->  两者都不管   (画面有内容)

任何"平"的帧一定落进其中之一, 不留缝隙也不重叠。

--------------------------------------------------------------------------
旧版 (new/occlusion.py) 的问题
--------------------------------------------------------------------------

1. **自参照基线 + AND, 整路遮挡必然漏检 (最严重)**

   旧判据: `(std < 8) AND (std < 该视角 std 的中位数 × 0.15)`

   一路相机全程被挡死, std 恒为 2。它自己的中位数就是 2, 相对阈值 = 2 × 0.15 = 0.3。
   判据变成 `(2 < 8) AND (2 < 0.3)` = False。**一帧都不报。**

   "异常如果是全程的, 自参照基线就把它当成了正常" —— black 也有这个病, 但 black 用的是
   OR, 绝对阈值还能兜底; occlusion 用 AND, 基线一失效**整个判据就废了**。而"整路被挡死"
   恰恰是最该抓的情况。基线用的还是中位数, 遮挡超过一半时间就开始失效。

   新版直接删掉相对阈值。判"画面有没有内容"用绝对标准就够了 —— std 是有物理意义的量,
   不需要和这台相机自己比。

2. **README 说的"抓半遮挡"做不到, 而且方向是反的**

   旧 README/docstring: "手挡住镜头一半, 均值可能正常 (不触发黑屏), 但 std 骤降 (触发遮挡)"。

   这个说法错了。手挡一半时画面是**双峰分布** (一半是暗的手, 一半是亮的场景), 标准差会
   **升高**, 不是骤降。举例: 正常场景像素散布在 60-180, std≈30; 挡一半后一半≈40 一半≈120,
   std≈40。用整帧 std 不但抓不到半遮挡, 半遮挡还会让 std 更远离阈值。

   整帧 std 能抓的只有"**整个画面**变成均匀一片"。新版 docstring 和文档如实这么写。
   真要抓半遮挡得用分块 std (把画面切成 N×N 格, 看有没有整块坍缩), 那是另一个检测,
   现在没有 —— 与其在文档里声称覆盖了, 不如明确写出来。

3. **四档严重度 + 3×4 时长表存在倒挂**, 与 black 同款。实测: 时长落在 0.20-0.33 秒时,
   1 路遮挡判 moderate, 3 路同时遮挡反而判 mild —— 路数越多判得越轻。
   而且 severe 档的 action 是 "split_or_drop_episode", 没有对应的自动处理分支, 等于静默保留。

4. **exclude_mask 会整列抹掉一路相机**, 与 black 同款。见 black.py docstring。

5. **`fps or 30.0`**: 拿不到帧率就假装 30。10fps 的视频等于所有秒级阈值收紧 3 倍,
   而且从输出里完全看不出来。新版拿不到 fps 直接报错。

6. **`gray_std is None` 时静默返回空列表** = "这条 episode 没有遮挡"。测都没测就报干净。
   新版报 UnsupportedDataError。

--------------------------------------------------------------------------
规则表 (rule 字段的取值)。**只有一条, 结果只有丢整条**:

  OCCLUSION_RUN_RATIO_EXCEEDED   任意一路相机的**最长连续**遮挡段
                                 >= max_occlusion_run_ratio (默认 10%) -> drop_episode

判据是**最长连续段**, 不是遮挡帧总量 —— 与 black 口径一致。零星的单帧低纹理 (画面正好
扫过一块纯色区域) 不会累加成删除理由, 只有"镜头被持续挡住"才会。总量仍然测出来放进
measurements, 但不参与判定。

"任意一路"而不是"所有路": 视角数量各数据集/机型不同, 只要有一路被挡瞎, 这条数据就不完整。
分母是该路**有测量值的帧数** (不含 NaN, 那些是 corrupt 的事)。

**报错 vs 筛选**: 见 errors.py。本模块的报错项:
  - probe 没带 gray_mean / gray_std   -> UnsupportedDataError
  - 拿不到 fps / 各相机 fps 不一致     -> UnsupportedDataError
  - 各相机时间轴长度不等               -> UnsupportedDataError (corrupt 该先拦下)
  - 一路可用的相机都没有               -> UnsupportedDataError (不能返回"干净")
  - numpy 不可用                      -> PipelineEnvironmentError

用法:
    from check_pipeline.probe import probe_video
    from check_pipeline.occlusion import check_occlusion

    probes = {cam: probe_video(b, cam) for cam, b in videos.items()}
    report = check_occlusion(probes)    # 与 check_black 复用同一批 probe, 不重复解码
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .errors import PipelineEnvironmentError, UnsupportedDataError
from .intervals import (indices_to_ranges, longest_range, longest_range_span,
                        merge_ranges, ranges_length)
from .probe import VideoProbe

logger = logging.getLogger("check_pipeline.occlusion")


# =====================================================================
# 参数 (3 个)
# =====================================================================
DEFAULT_PARAMS = {
    "std_abs": 8.0,
    #   灰度标准差上限, "画面没有内容"的判据。与 black 的 std_abs **必须取同一个值**,
    #   否则两个检测之间会出现缝隙 (见模块 docstring 的分工表)。
    #   8 的含义: 在 0-255 的灰度上, 像素几乎全挤在一个值附近。正常室内场景 std 25-50,
    #   低纹理场景 (白墙/素色桌面) std 10-20, 真遮挡 (手贴住镜头) std 0-5。
    "mean_abs": 18.0,
    #   灰度均值下限, 与 black 的 mean_abs **必须取同一个值**。低于它归 black (又暗又平),
    #   不低于它才归 occlusion。取值理由见 black.py DEFAULT_PARAMS。
    "max_occlusion_run_ratio": 0.10,
    #   **唯一的删除规则**: 任意一路相机的**最长连续**遮挡段达到这个占比 -> 丢弃整条。
    #   是最长连续段, 不是遮挡帧总量 —— 与 black 口径一致, 理由见 black.py。
}


# =====================================================================
# 规则表
# =====================================================================
DROP_EPISODE = "drop_episode"

RULES: Dict[str, Tuple[str, str]] = {
    "OCCLUSION_RUN_RATIO_EXCEEDED": (
        "存在相机的最长连续遮挡段达到阈值占比 (画面持续平但不暗 = 镜头被挡住/糊住)",
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
class ViewOcclusion:
    """一路相机的遮挡测量结果。只描述测到了什么, 不含判定。"""
    cam: str = ""
    n_frames: int = 0                  # 时间轴长度
    n_valid: int = 0                   # 有测量值的帧数 (非 NaN)
    n_occluded: int = 0
    occlusion_ratio: float = 0.0       # n_occluded / n_valid
    occluded_ranges: List[Tuple[int, int]] = field(default_factory=list)
    longest_occluded_run: int = 0                    # 最长连续遮挡段的长度 (判定用)
    longest_occluded_run_ratio: float = 0.0          # 它占 n_valid 的比例 (判定用)
    longest_occluded_run_span: Optional[Tuple[int, int]] = None   # 它的位置 (排查用)
    min_gray_std: Optional[float] = None
    gray_std_percentiles: Dict[str, float] = field(default_factory=dict)
    # 又平又暗的帧数。这些帧归 black 管, occlusion 不计入 n_occluded, 只报数便于对账。
    n_flat_and_dark: int = 0

    def to_dict(self) -> dict:
        return {
            "cam": self.cam,
            "n_frames": self.n_frames,
            "n_valid": self.n_valid,
            "n_occluded": self.n_occluded,
            "occlusion_ratio": round(self.occlusion_ratio, 6),
            "occluded_ranges": [list(r) for r in self.occluded_ranges],
            "longest_occluded_run": self.longest_occluded_run,
            "longest_occluded_run_ratio": round(self.longest_occluded_run_ratio, 6),
            "longest_occluded_run_span": (list(self.longest_occluded_run_span)
                                          if self.longest_occluded_run_span else None),
            "min_gray_std": self.min_gray_std,
            "gray_std_percentiles": self.gray_std_percentiles,
            "n_flat_and_dark": self.n_flat_and_dark,
        }


@dataclass
class OcclusionReport:
    findings: List[Finding] = field(default_factory=list)
    views: Dict[str, ViewOcclusion] = field(default_factory=dict)
    n_frames: int = 0
    fps: float = 0.0
    params: dict = field(default_factory=dict)

    @property
    def has_occlusion(self) -> bool:
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
            "check": "occlusion",
            "has_occlusion": self.has_occlusion,
            "drop_episode": self.drop_episode,
            "triggered_rules": self.triggered_rules,
            "findings": [f.to_dict() for f in self.findings],
            "measurements": {
                "n_frames": self.n_frames,
                "fps": round(self.fps, 3),
                "params": self.params,
                "views": {c: v.to_dict() for c, v in sorted(self.views.items())},
            },
        }


# =====================================================================
# 测量
# =====================================================================
def measure_view(probe: VideoProbe, std_abs: float, mean_abs: float) -> ViewOcclusion:
    """测量一路相机的遮挡情况。只测量, 不判定。

    NaN (读不出来的帧) 不参与任何统计: 那些是 corrupt 的事, 既不算遮挡也不算正常。
    """
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise PipelineEnvironmentError(f"numpy 不可用: {e}") from e

    if probe.gray_std is None or probe.gray_mean is None:
        raise UnsupportedDataError(
            f"相机 {probe.cam!r} 的 probe 没有灰度数据; "
            f"probe_video(..., collect_gray=True) 才能跑遮挡检测")

    gs = np.asarray(probe.gray_std, dtype=np.float64)
    gm = np.asarray(probe.gray_mean, dtype=np.float64)
    out = ViewOcclusion(cam=probe.cam or "", n_frames=int(gs.shape[0]))

    valid = ~(np.isnan(gs) | np.isnan(gm))
    out.n_valid = int(valid.sum())
    if out.n_valid == 0:
        return out

    # 遮挡 = 平 且 不暗。"平且暗"归 black, 这里排除掉, 两个检测互斥。
    is_flat = valid & (gs < std_abs)
    is_occluded = is_flat & (gm >= mean_abs)
    out.n_occluded = int(is_occluded.sum())
    out.occlusion_ratio = out.n_occluded / out.n_valid
    out.occluded_ranges = indices_to_ranges(np.flatnonzero(is_occluded).tolist())
    out.longest_occluded_run = longest_range(out.occluded_ranges)
    out.longest_occluded_run_ratio = out.longest_occluded_run / out.n_valid
    out.longest_occluded_run_span = longest_range_span(out.occluded_ranges)
    out.n_flat_and_dark = int(is_flat.sum()) - out.n_occluded

    vs = gs[valid]
    out.min_gray_std = float(vs.min())
    out.gray_std_percentiles = {
        "p1": float(np.percentile(vs, 1)),
        "p50": float(np.percentile(vs, 50)),
        "p99": float(np.percentile(vs, 99)),
    }
    return out


# =====================================================================
# 主检测
# =====================================================================
def check_occlusion(probes: Dict[str, VideoProbe],
                    params: Optional[dict] = None,
                    ) -> OcclusionReport:
    """对一条 episode 跑遮挡检测。

    **唯一的删除规则**: 任意一路相机的**最长连续**遮挡段 >= max_occlusion_run_ratio
    (默认 10%) -> 丢弃整条 episode。判据是最长连续段, 不是遮挡帧总量。

    probes: {相机名: probe_video(...) 的结果}, 必须是 collect_gray=True 解出来的。
        可以和 check_black 复用同一批 probe, 不需要重复解码。
        整路缺失/打不开/一帧读不出来的相机会被跳过 —— 那些 corrupt 已经判了丢整条。
    params: 覆盖 DEFAULT_PARAMS 的部分键。传未知键名会报错, 不会静默忽略。

    可能抛 CheckError, 表示**检测没能完成**; 调用方必须记为"检测失败",
    既不能计入"干净", 也不能判成"有遮挡"。
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

    usable = {cam: pr for cam, pr in probes.items() if pr.usable}
    if not usable:
        raise UnsupportedDataError(
            f"{len(probes)} 路相机没有一路可用 (缺失/打不开/零可读帧), 无法判断遮挡。"
            f"这条 episode 应由 corrupt 处理, 不能当成'没有遮挡'")

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

    report = OcclusionReport(n_frames=next(iter(lengths.values())),
                             fps=next(iter(fps_values)), params=dict(p))

    # ---- 1) 逐路测量 ----
    for cam, pr in usable.items():
        report.views[cam] = measure_view(pr, p["std_abs"], p["mean_abs"])

    # ---- 2) 唯一的规则: 任意一路遮挡占比达标 -> 丢整条 ----
    over = {c: v for c, v in report.views.items()
            if v.n_valid > 0
            and v.longest_occluded_run_ratio >= p["max_occlusion_run_ratio"]}
    if over:
        report.findings.append(_finding(
            "OCCLUSION_RUN_RATIO_EXCEEDED", views=sorted(over),
            frame_ranges=merge_ranges(
                [r for v in over.values() for r in v.occluded_ranges]),
            evidence={
                "criterion": "longest_occluded_run / n_valid",
                "threshold": p["max_occlusion_run_ratio"],
                "std_abs": p["std_abs"],
                "mean_abs": p["mean_abs"],
                "n_views_checked": len(report.views),
                "n_views_over": len(over),
                "per_view_over": {
                    c: {"longest_occluded_run": v.longest_occluded_run,
                        "longest_occluded_run_ratio": round(v.longest_occluded_run_ratio, 6),
                        "longest_occluded_run_span": (list(v.longest_occluded_run_span)
                                                      if v.longest_occluded_run_span else None),
                        "n_valid": v.n_valid,
                        "n_occluded_total": v.n_occluded,
                        "occlusion_ratio_total": round(v.occlusion_ratio, 6),
                        "n_segments": len(v.occluded_ranges),
                        "min_gray_std": v.min_gray_std}
                    for c, v in sorted(over.items())},
                "per_view_longest_run_ratio": {
                    c: round(v.longest_occluded_run_ratio, 6)
                    for c, v in sorted(report.views.items())},
                # 总量只做参考, 不参与判定
                "per_view_occlusion_ratio_total": {
                    c: round(v.occlusion_ratio, 6)
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
