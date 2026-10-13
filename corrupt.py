"""corrupt — episode 损坏检测 (重写版)。

只回答一个问题: **这条 episode 的原始文件本身是否完整可读**。不判断画面内容好坏
(黑屏/遮挡/静止那些是后续检测的事), 也不做严重度分级。

与旧版的根本区别:
  1. 不读 info.json。相机数量/分辨率/总帧数一律从 episode 自身的 mp4 与 parquet 现场测量,
     不和任何"声明值"比对 (声明值本身可能是错的, 比对结果不可信)。
  2. 不输出 severity (critical/severe/moderate/...)。每条命中直接写明**触发了哪条规则**
     (rule 字段 = 规则码) + **触发时的实测数字** (evidence 字段)。
  3. 坏帧位置用 PTS (显示时间戳) 定位, 不用"解码计数器"。解码失败的包不会让后续帧号漂移。
  4. 坏帧按**连续段**逐段输出。第 5 帧和第 1050 帧各坏一帧, 输出 [(5,6), (1050,1051)],
     不再输出 (5, 1051)。注意这现在是**诊断信息**(坏在哪里), 不是"要删哪些帧"。
  5. **只有丢弃整条 episode 一种结果, 没有删帧。** 只要有一帧读不出来就整条丢弃,
     不再按坏帧数量/连续长度分档。理由见 RULES 下方的说明。

模块划分 (与 black 等后续检测共用):
  errors.py     异常体系 (报错 vs 筛选的三类划分)
  intervals.py  帧区间工具
  probe.py      单路视频的唯一一次解码 (PTS 定位 / 坏帧 NaN 占位 / 像素格式白名单)
  corrupt.py    本文件: 只做规则判定, 不碰解码

规则表 (rule 字段的取值)。**全部都是 drop_episode**:

  VIDEO_FILE_MISSING            相机的视频文件缺失 (调用方没给字节)
  VIDEO_UNDECODABLE             整路视频无法打开 / 无视频流
  VIDEO_NO_READABLE_FRAME       能打开但一帧都读不出来
  FRAME_DECODE_FAILED           存在读取失败的帧 (哪怕只有 1 帧)
  FRAME_SIZE_CHANGED            画面 H×W 中途变化
  VIEW_FRAME_COUNT_MISMATCH     多视角之间帧数不一致
  ACTION_VIDEO_LENGTH_MISMATCH  动作数量 != 视频帧数
  PARQUET_UNREADABLE            parquet 读不出来 (无法取动作数量)

**报错 vs 筛选** (重要):

  本模块把结果分成三类, 绝不把后两类混进第一类:

    1. 数据坏了           -> 产出 finding, 走上面的规则表 (删帧 / 删 episode)
    2. 运行环境不对       -> 抛 PipelineEnvironmentError, 不产出任何 finding
    3. 数据格式不支持     -> 抛 UnsupportedDataError, 不产出任何 finding

  第 2、3 类**不捕获**, 直接往上抛, 强制调用方显式处理。这条 episode 应记为
  "检测失败", 而不是"已检查且干净", 更不能被判成"数据损坏"。

  为什么必须这样: 旧实现里 pyav 没装会走进"视频整路无法解码"分支, 于是**每一条
  episode 都被判成损坏并建议删除** —— 一个环境问题足以删光整个数据集, 而报告上
  每条都写着"数据损坏", 看不出真实原因。任何"猜个默认值继续跑"的降级都有同类风险:
  结论看上去正常, 但已经不可信了。

  同理, 帧号定位不可信时 (拿不到 PTS) 也必须报错而不是退化成顺序计数 ——
  退化之后仍然会输出帧区间, 下游拿去删 parquet 对应行会**删错行**, 且删完长度
  依然对得上, 事后无法发现。

用法:
    from check_pipeline.corrupt import check_corrupt, CorruptCheckError

    try:
        report = check_corrupt(videos={"cam_high": b"...", "cam_left": b"..."},
                               parquet_bytes=b"...")
    except CorruptCheckError as e:
        ...  # 检测失败, 单独记录, 不要当成"干净"也不要当成"损坏"
    else:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        report.drop_episode      # bool, 是否整条丢弃 (唯一的处理结果)
        report.triggered_rules   # 触发了哪些规则
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import CheckError, PipelineEnvironmentError, UnsupportedDataError
from .intervals import indices_to_ranges, longest_range, merge_ranges, ranges_length  # noqa: F401  (re-export)
from .probe import SUPPORTED_PIXEL_FORMATS, VideoProbe, probe_video

logger = logging.getLogger("check_pipeline.corrupt")

# 兼容旧名字。异常体系已提到 errors.py, 供全部检测共用。
CorruptCheckError = CheckError


# =====================================================================
# 规则表: rule -> (中文说明, 处理方式)
# =====================================================================
DROP_EPISODE = "drop_episode"     # 整条 episode 丢弃 (含其 mp4 / parquet / meta 索引条目)

RULES: Dict[str, Tuple[str, str]] = {
    "VIDEO_FILE_MISSING": (
        "相机的视频文件缺失: 调用方给出了相机名但没有视频字节", DROP_EPISODE),
    "VIDEO_UNDECODABLE": (
        "视频整路无法解码: 文件打不开, 或容器里没有视频流", DROP_EPISODE),
    "VIDEO_NO_READABLE_FRAME": (
        "视频可以打开但一帧都读不出来", DROP_EPISODE),
    "FRAME_DECODE_FAILED": (
        "存在读取失败的帧 (解码报错, 或时间轴上存在丢包空洞)", DROP_EPISODE),
    "FRAME_SIZE_CHANGED": (
        "画面尺寸 (高×宽) 在 episode 中途发生变化", DROP_EPISODE),
    "VIEW_FRAME_COUNT_MISMATCH": (
        "多个视角之间视频帧数不一致", DROP_EPISODE),
    "ACTION_VIDEO_LENGTH_MISMATCH": (
        "parquet 动作数量与视频帧数不一致", DROP_EPISODE),
    "PARQUET_UNREADABLE": (
        "parquet 无法读取, 取不到动作数量", DROP_EPISODE),
}

# parquet 里认定为"动作列"的列名 (小写包含即命中)。
# 不依赖 info.json 的 action_cols 声明, 直接按 lerobot 惯例从列名识别。
ACTION_COLUMN_HINTS = ("action", "actions")

# 说明: corrupt 不再有"删帧"这条路。**只要有任何一帧读不出来, 整条 episode 丢弃。**
#
# 之前的做法是按最长连续坏帧段分档: 零星短段删帧、长段丢整条。取消的理由:
#   1. 删帧是把坏帧从视频和 parquet 里同时抠掉再首尾相接, 断口两侧在时间上不连续,
#      机械臂会"瞬移"过去。这段假跳变既污染训练, 也会被 sudden_change 当成真突变报出来。
#      零星单帧看似安全, 但一条 episode 里散布几十个单帧断口, 累计效果和长断口没区别。
#   2. 分档意味着引入阈值, 而阈值必然产生"29 帧保留 / 30 帧删除"这种物理上不存在的边界。
#   3. 文件层面读不出来的帧, 说明这个文件本身的写入或传输出过问题, 很难保证"其余部分
#      一定是好的"。与其部分挽救, 不如整条剔除。
#
# 代价是删除率会上升: 一条 1500 帧的 episode 只要坏 1 帧就整条丢弃。跑第一批时
# 看 measurements 里的 n_missing 分布, 确认坏帧是不是集中在少数 episode 上。


# =====================================================================
# 数据结构
# =====================================================================
@dataclass
class Finding:
    """一条命中的规则。JSON 里每条 issue 就是这个结构。"""
    rule: str                                   # 规则码, 见 RULES
    description: str                            # 规则中文说明 (从 RULES 取, 便于直接看 JSON)
    disposition: str                            # 当前恒为 drop_episode (字段保留以统一 JSON 结构)
    views: List[str] = field(default_factory=list)   # 涉及的相机; 空=不针对具体相机
    frame_ranges: List[Tuple[int, int]] = field(default_factory=list)  # [start, end) 逐个连续段
    n_frames_affected: int = 0                  # 受影响帧数 (frame_ranges 的总长)
    evidence: dict = field(default_factory=dict)  # 触发时的实测数字 (每条规则自带)

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
class CorruptReport:
    """一条 episode 的 corrupt 检测结果。to_dict() 即写进 result.json 的内容。"""
    findings: List[Finding] = field(default_factory=list)
    videos: Dict[str, VideoProbe] = field(default_factory=dict)
    n_action_rows: Optional[int] = None       # parquet 动作数量 (行数)
    action_columns: List[str] = field(default_factory=list)
    video_frame_counts: Dict[str, int] = field(default_factory=dict)  # {cam: 帧数}
    reference_frame_count: Optional[int] = None  # 用于比对的视频帧数 (各视角取 min)

    @property
    def corrupt(self) -> bool:
        """是否命中了任何规则。所有规则都对应实际处理动作, 没有"只记录"的档位。"""
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
            "check": "corrupt",
            "corrupt": self.corrupt,
            "drop_episode": self.drop_episode,
            "triggered_rules": self.triggered_rules,
            "findings": [f.to_dict() for f in self.findings],
            "measurements": {
                "video_frame_counts": self.video_frame_counts,
                "reference_frame_count": self.reference_frame_count,
                "n_action_rows": self.n_action_rows,
                "action_columns": self.action_columns,
                "per_video": {c: p.to_dict() for c, p in self.videos.items()},
            },
        }


# =====================================================================
# parquet 动作数量 (不读 info.json, 直接看列名)
# =====================================================================
def probe_parquet(parquet_bytes: Optional[bytes]) -> dict:
    """从 parquet 现场测出动作数量。

    不依赖 info.json 的 action_cols 声明: 按 lerobot 惯例, 列名里含 "action" 的即动作列。
    动作数量 = parquet 行数 (每行一个时间步)。

    报错情况 (不降级):
      - pandas 不可用                  -> PipelineEnvironmentError
      - parquet 字节为 None            -> UnsupportedDataError (核对做不了, 不能当"通过")
      - 认不出动作列                    -> UnsupportedDataError (列名不符合支持的约定)
    parquet **读取失败** 不报错 —— 那是数据坏了, 走 PARQUET_UNREADABLE 规则。

    返回 {"ok": bool, "error": str|None, "n_rows": int|None, "action_columns": [...]}
    """
    if parquet_bytes is None:
        raise UnsupportedDataError(
            "未提供 parquet 字节, 无法核对动作数量与视频帧数。"
            "这条核对做不了就不能给出'干净'的结论")
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise PipelineEnvironmentError(f"pandas 不可用, 无法读取 parquet: {e}") from e

    out = {"ok": False, "error": None, "n_rows": None, "action_columns": []}
    try:
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        # 数据坏了, 不是格式不支持 -> 交给 PARQUET_UNREADABLE 规则
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["ok"] = True
    out["n_rows"] = int(len(df))
    out["action_columns"] = [c for c in df.columns
                             if any(h in str(c).lower() for h in ACTION_COLUMN_HINTS)]
    if not out["action_columns"]:
        raise UnsupportedDataError(
            f"parquet 里认不出动作列 (列名: {sorted(str(c) for c in df.columns)}; "
            f"识别规则: 列名小写包含 {list(ACTION_COLUMN_HINTS)})。"
            f"认不出就没法核对动作数量, 不能静默放行")
    return out


# =====================================================================
# 主检测
# =====================================================================
def check_corrupt(videos: Dict[str, Optional[bytes]],
                  parquet_bytes: Optional[bytes],
                  probes: Optional[Dict[str, VideoProbe]] = None,
                  ) -> CorruptReport:
    """对一条 episode 跑 corrupt 检测。命中任何一条规则即丢弃整条 episode。

    videos: {相机名: mp4 字节}。相机名与数量以这个字典为准 (调用方从目录里扫出来的实际文件),
        不和 info.json 声明比对 —— 声明值可能是错的。某个相机的值为 None 表示文件缺失,
        走 VIDEO_FILE_MISSING 规则。
    parquet_bytes: lerobot parquet 字节。**必传** —— 传 None 会报 UnsupportedDataError,
        因为动作数量核对做不了就不能给出"干净"的结论。
    probes: 已经解码好的 {相机名: VideoProbe}。传了就直接用, 不再解码 —— 供 pipeline.py
        让四个检测共用同一次解码。必须覆盖 videos 里的全部相机, 否则报错 (少一路就
        少一路的结论, 不能当成那一路没问题)。

    返回 CorruptReport, 用 .to_dict() 直接写 JSON。

    可能抛 CorruptCheckError (PipelineEnvironmentError / UnsupportedDataError), 表示
    **检测没能完成**。这类异常不捕获, 调用方必须把该 episode 记为"检测失败", 既不能
    计入"干净", 也不能判成"损坏"。注意任一路相机触发即中止整条 episode 的检测 ——
    这是刻意的: 结论要么整条可信, 要么没有结论。
    """
    if not videos:
        raise UnsupportedDataError("未提供任何视频, 无法检测")

    report = CorruptReport()

    # ---- 1) 逐路视频现场测量 (外部已解码则复用) ----
    if probes is None:
        for cam in videos:
            report.videos[cam] = probe_video(videos[cam], cam)
    else:
        missing = sorted(set(videos) - set(probes))
        if missing:
            raise UnsupportedDataError(
                f"传入的 probes 缺少相机 {missing}; 少一路就少一路的结论, "
                f"不能当成那一路没问题")
        for cam in videos:
            report.videos[cam] = probes[cam]

    # ---- 2) 规则 1: 文件缺失 / 整路无法解码 / 一帧都读不出来 ----
    for cam, p in report.videos.items():
        if p.missing_file:
            report.findings.append(_finding(
                "VIDEO_FILE_MISSING", views=[cam], evidence={"cam": cam}))
            continue
        if p.open_error:
            report.findings.append(_finding(
                "VIDEO_UNDECODABLE", views=[cam],
                evidence={"cam": cam, "error": p.open_error}))
            continue
        if p.n_readable == 0:
            report.findings.append(_finding(
                "VIDEO_NO_READABLE_FRAME", views=[cam],
                evidence={"cam": cam,
                          "n_declared": p.n_declared,
                          "n_readable": 0,
                          "packet_errors": p.packet_errors,
                          "first_error": p.first_error}))

    # ---- 3) 规则 2: 存在读取失败的帧 (逐段记录位置, 帧号按 PTS 定位) ----
    # 只要有一帧读不出来就丢整条。frame_ranges 仍然逐段精确输出, 但那是**诊断信息**
    # (坏在哪里、坏了几段), 不再是"要删哪些帧"。
    for cam, p in report.videos.items():
        if p.missing_file or p.open_error or p.n_readable == 0:
            continue  # 已按规则 1 整条处理
        if p.n_missing == 0:
            continue
        report.findings.append(_finding(
            "FRAME_DECODE_FAILED", views=[cam],
            frame_ranges=p.missing_ranges,
            evidence={"cam": cam,
                      "n_declared": p.n_declared,
                      "n_readable": p.n_readable,
                      "n_frames": p.n_frames,
                      "n_missing": p.n_missing,
                      "missing_ratio": round(p.n_missing / p.n_frames, 6) if p.n_frames else None,
                      "n_segments": len(p.missing_ranges),
                      "longest_missing_run": longest_range(p.missing_ranges),
                      "packet_errors": p.packet_errors,
                      "first_error": p.first_error}))

    # ---- 4) 规则 3: 画面尺寸中途变化 (基准 = 出现帧数最多的尺寸) ----
    for cam, p in report.videos.items():
        if p.missing_file or p.open_error or p.n_readable == 0:
            continue
        if p.size_mismatch_ranges:
            report.findings.append(_finding(
                "FRAME_SIZE_CHANGED", views=[cam],
                frame_ranges=p.size_mismatch_ranges,
                evidence={"cam": cam,
                          "dominant_size": list(p.dominant_size) if p.dominant_size else None,
                          "size_histogram": p.size_histogram,
                          "n_distinct_sizes": len(p.size_histogram),
                          "n_frames_changed": ranges_length(p.size_mismatch_ranges),
                          "n_segments": len(p.size_mismatch_ranges)}))

    # ---- 5) 规则 4: 多视角帧数一致 / 动作数量与视频帧数一致 ----
    # 只统计"能读出帧"的视角; 整路坏掉的视角已由规则 1 处理, 不参与帧数比对。
    usable = {cam: p for cam, p in report.videos.items()
              if not p.missing_file and not p.open_error and p.n_readable > 0}
    report.video_frame_counts = {cam: p.n_frames for cam, p in usable.items()}

    if report.video_frame_counts:
        counts = sorted(set(report.video_frame_counts.values()))
        # 参考帧数取各视角最小值 (对齐时能保证每个视角都有这一帧); 旧版取字典第一个,
        # 结论会随相机遍历顺序翻转。
        report.reference_frame_count = min(report.video_frame_counts.values())
        if len(counts) > 1:
            report.findings.append(_finding(
                "VIEW_FRAME_COUNT_MISMATCH",
                views=sorted(usable.keys()),
                evidence={"per_view_frame_count": dict(report.video_frame_counts),
                          "distinct_counts": counts,
                          "min": counts[0], "max": counts[-1],
                          "max_gap": counts[-1] - counts[0]}))

    # parquet 读不出来是数据问题 (走规则); 认不出动作列是格式不支持 (probe_parquet 抛错)
    pq = probe_parquet(parquet_bytes)
    report.n_action_rows = pq["n_rows"]
    report.action_columns = pq["action_columns"]
    if not pq["ok"]:
        report.findings.append(_finding(
            "PARQUET_UNREADABLE",
            evidence={"error": pq["error"]}))
    elif report.reference_frame_count is not None:
        n_act = pq["n_rows"]
        n_vid = report.reference_frame_count
        if n_act != n_vid:
            report.findings.append(_finding(
                "ACTION_VIDEO_LENGTH_MISMATCH",
                views=sorted(usable.keys()),
                evidence={"n_action_rows": n_act,
                          "video_frame_count": n_vid,
                          "diff": n_act - n_vid,
                          "per_view_frame_count": dict(report.video_frame_counts),
                          "action_columns": pq["action_columns"]}))

    return report


def _finding(rule: str, views: Optional[List[str]] = None,
             frame_ranges: Optional[List[Tuple[int, int]]] = None,
             evidence: Optional[dict] = None) -> Finding:
    desc, disposition = RULES[rule]
    frs = list(frame_ranges or [])
    return Finding(rule=rule, description=desc, disposition=disposition,
                   views=list(views or []), frame_ranges=frs,
                   n_frames_affected=ranges_length(frs), evidence=evidence or {})

# 注: 曾经有一个 to_frame_delete_mask(), 已随"删帧"路径一起移除。
# corrupt 现在只有 drop_episode 一种结果, 不需要帧掩码。
