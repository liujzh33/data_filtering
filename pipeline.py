"""pipeline — 七个检测的编排入口, 删帧清单的汇总, 以及删除率预算。

**整条流水线只检测和记录, 不删任何数据。** 帧级命中汇总成一张工单 (manifest.py),
交给下游写出器执行。检测可以重跑一万次不改数据, 删除是不可逆的 —— 这两件事必须分开。

--------------------------------------------------------------------------
执行顺序
--------------------------------------------------------------------------

    [阶段 0]  probe_video()            逐路解码, 每路只解一次 (占单条耗时 95% 以上)
                  |
    [阶段 1]  corrupt                  <- gate: 文件能不能用、各模态是否对齐
                  |
              black / occlusion / stillness    <- 看画面, 三者互不依赖
                  |
    [阶段 2]  saa                      <- 先跑: episode 级, 不过就整条丢
                  |
              sudden_change / state_extreme    <- 帧级, 产出删帧区间

**为什么 corrupt 必须第一个跑。** 它检查的是"第 i 个视频帧 == 第 i 行动作数据"这件事
成不成立。后面所有检测都建立在这个下标映射上: black 说"第 30-50 帧是黑的", 只有当各路
视频和动作表共享同一条时间轴时才有意义。corrupt 的 VIEW_FRAME_COUNT_MISMATCH /
ACTION_VIDEO_LENGTH_MISMATCH 一旦命中, 帧区间就退化成一串没有所指的数字 ——
所以**帧级删除工单在 corrupt 未过时一律标为不可执行** (见 manifest.py)。

**为什么阶段 2 里 saa 排在前面。** saa 是 episode 级的 (DA 不达标就整条丢), 而
sudden_change / state_extreme 是帧级的。整条都不要了, 再算删哪几帧没有意义。顺序
放在前面, 开了短路开关时能省掉后两个检测。

**为什么阶段 1 的三个检测不做并行。** 它们都是在阶段 0 已经解出来的数组上做向量运算,
耗时相对解码可以忽略。在 check 这一层开线程/进程, 调度和序列化开销比省下来的计算还多。
真正该并行的粒度是 **episode 级** —— 多进程各跑各的, 进程内部按上面的顺序串行。

--------------------------------------------------------------------------
默认不短路: 因为现在还在调阈值
--------------------------------------------------------------------------

`short_circuit=False` (默认) 时, 哪怕 corrupt 已经判了整条删除, 后面六个检测照常跑完。

短路能省一点计算, 但会让**交叠统计**永远做不了。而交叠统计回答的是一个很实际的问题:
四个检测各自命中率都是 5%, 总删除率可能是 5% (完全重合) 到 20% (完全不相交) 之间的
任何值。只看各自的命中率, 这两种情况分不出来, 而处置完全相反 —— 完全重合说明有三个
检测是冗余的, 完全不相交说明数据有四类独立问题。

调阈值阶段需要这个数 (DropBudget.per_check.only_rate 就是"关掉这个检测能救回多少数据"),
所以默认跑满。**阈值冻结、进入量产之后把 short_circuit 打开**, 那时统计已经没用了,
省下的是真金白银的机时。

--------------------------------------------------------------------------
报错: 分三种, 各归各的位置
--------------------------------------------------------------------------

检测**从不静默降级**。出问题一律带着具体原因落到 JSON 里, 分三档:

  1. **整条失败** (result.failed)  解码失败, 或某个检测抛 PipelineEnvironmentError
     (scipy 缺失这类)。意味着这条 episode **没有结论** —— 既不是干净也不是坏, 必须
     单独归档, 不能计进任何一边的比率。

  2. **单个检测报错** (checks[x].status == "error")  该检测抛 UnsupportedDataError:
     profile 没配、维度对不上、有 NaN、格式不支持。**只影响这一个检测**, 其他照常出结论。
     错误类型和原文都写进 JSON, 例如 `"数据格式 'rosbag' 未知; 支持的格式: [...]"`,
     而不是笼统的"数据错误"。

  3. **主动跳过** (checks[x].status == "skipped")  profile 里标了 enabled=false,
     或开了短路。**单独记账** —— "零命中"和"根本没跑"在 JSON 里必须长得不一样。

--------------------------------------------------------------------------
用法
--------------------------------------------------------------------------

    from check_pipeline.pipeline import run_episode, DropBudget
    from check_pipeline.manifest import EpisodeAssets

    result = run_episode(
        videos={"head": head_mp4, "left_wrist": lw_mp4},
        parquet_bytes=pq,
        episode_id="ep_000123",
        profile=profile,                    # dataset_profile.json 的内容
        assets=EpisodeAssets(
            data_format="lerobot",
            table_path="data/chunk-000/episode_000123.parquet",
            video_paths={"head": "videos/chunk-000/head/episode_000123.mp4",
                         "left_wrist": "videos/chunk-000/left_wrist/episode_000123.mp4"},
            meta_paths={"episodes": "meta/episodes.jsonl"}),
        params={"stillness": {"state_diff_thresh": 1e-3}})

    json.dump(result.to_dict(), open(f"{result.episode_id}.json", "w"),
              ensure_ascii=False, indent=2)

非 parquet 的数据集 (hdf5 / jsonl) 由格式适配器先把信号取出来, 用 profile 里声明的
column 名作键传进 `signals=`, 流水线本身不关心物理格式。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import CheckError, PipelineEnvironmentError, UnsupportedDataError
from .manifest import DeletionManifest, EpisodeAssets, build_manifest
from .probe import VideoProbe, probe_video

logger = logging.getLogger("check_pipeline.pipeline")

GATE = "corrupt"
STAGE1_VIDEO = ("black", "occlusion", "stillness")
STAGE2_EPISODE = ("saa",)
STAGE2_FRAME = ("sudden_change", "state_extreme")

ALL_CHECKS: Tuple[str, ...] = (GATE,) + STAGE1_VIDEO + STAGE2_EPISODE + STAGE2_FRAME

CHECK_STAGE: Dict[str, int] = {c: (1 if c in (GATE,) + STAGE1_VIDEO else 2)
                               for c in ALL_CHECKS}

# corrupt 的这两条规则一旦命中, "第 i 个视频帧 == 第 i 行" 就不成立, 帧区间失去所指。
# 此时帧级工单必须标为不可执行 —— 照着删只会把错位坐实。
ALIGNMENT_RULES = ("VIEW_FRAME_COUNT_MISMATCH", "ACTION_VIDEO_LENGTH_MISMATCH")


class _Skip(Exception):
    """内部信号: 这个检测被配置主动关掉了, 不是出错。"""


# =====================================================================
# 单个检测的结果
# =====================================================================
@dataclass
class CheckOutcome:
    """一个检测的终态。三种 status 互斥, 调用方必须分开处理。"""
    check: str = ""
    stage: int = 0
    status: str = ""                      # ok / error / skipped
    report: Any = None
    error_type: Optional[str] = None      # UnsupportedDataError / ...
    error_message: Optional[str] = None   # 具体原因原文, 不做二次概括
    skip_reason: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"check": self.check, "stage": self.stage, "status": self.status}
        if self.status == "error":
            # 报什么错必须看得出来: "数据格式 'rosbag' 未知" 和 "profile 缺少 q1"
            # 是完全不同的两件事, 不能都写成"数据错误"
            d["error_type"] = self.error_type
            d["error_message"] = self.error_message
        elif self.status == "skipped":
            d["skip_reason"] = self.skip_reason
        else:
            d["report"] = self.report.to_dict()
        return d


# =====================================================================
# 单条 episode 的结果
# =====================================================================
@dataclass
class EpisodeResult:
    """一条 episode 跑完七个检测的完整结果。

    三种互斥的终态:
      - failed=True       没有结论 (解码失败/环境问题), 必须单独归档
      - drop_episode=True 至少一个检测判整条丢弃, dropped_by 列出是哪些
      - 两者都 False      保留; 若 manifest.actionable 则需按工单删掉若干帧
    """
    episode_id: str = ""
    failed: bool = False
    failed_stage: Optional[str] = None
    failed_error_type: Optional[str] = None
    failed_error: Optional[str] = None

    outcomes: Dict[str, CheckOutcome] = field(default_factory=dict)
    manifest: Optional[DeletionManifest] = None
    n_frames: int = 0
    fps: float = 0.0

    @property
    def reports(self) -> Dict[str, Any]:
        return {c: o.report for c, o in self.outcomes.items() if o.status == "ok"}

    @property
    def drop_episode(self) -> bool:
        return bool(self.dropped_by)

    @property
    def dropped_by(self) -> List[str]:
        """判了整条丢弃的检测名, 按执行顺序。交叠统计基于这个列表。"""
        return [c for c in ALL_CHECKS
                if c in self.reports and self.reports[c].drop_episode]

    @property
    def triggered(self) -> List[dict]:
        """**扁平的命中清单**: 哪个阶段、哪个检测、触发了哪条规则、后果是什么。

        这是给人看 JSON 时最先看的一栏 —— 不用钻进各检测的 report 就能回答
        "这条数据被谁以什么理由处置了"。

        字段一律用 getattr 取: 各检测的 Finding 结构不完全一致 —— 看画面的带 views,
        看信号的带 dims, 而 saa 是纯 episode 级的, 连 frame_ranges 都没有。缺的补空,
        不要求各模块为了这里的统一去加无意义的字段。
        """
        out = []
        for c in ALL_CHECKS:
            o = self.outcomes.get(c)
            if o is None or o.status != "ok":
                continue
            for f in o.report.findings:
                out.append({
                    "stage": o.stage,
                    "check": c,
                    "rule": f.rule,
                    "description": f.description,
                    "disposition": f.disposition,
                    "views": list(getattr(f, "views", None) or []),
                    "dims": list(getattr(f, "dims", None) or []),
                    "frame_ranges": [list(r) for r in
                                     (getattr(f, "frame_ranges", None) or [])],
                })
        return out

    @property
    def triggered_rules(self) -> Dict[str, List[str]]:
        return {c: r.triggered_rules for c, r in self.reports.items() if r.triggered_rules}

    @property
    def errors(self) -> Dict[str, dict]:
        return {c: {"stage": o.stage, "type": o.error_type, "message": o.error_message}
                for c, o in self.outcomes.items() if o.status == "error"}

    @property
    def skipped(self) -> Dict[str, str]:
        return {c: o.skip_reason or "" for c, o in self.outcomes.items()
                if o.status == "skipped"}

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "failed": self.failed,
            "failed_stage": self.failed_stage,
            "failed_error_type": self.failed_error_type,
            "failed_error": self.failed_error,
            "drop_episode": self.drop_episode,
            "dropped_by": self.dropped_by,
            "triggered": self.triggered,
            "triggered_rules": self.triggered_rules,
            "errors": self.errors,
            "skipped": self.skipped,
            "frame_deletion": self.manifest.to_dict() if self.manifest else None,
            "checks": {c: o.to_dict() for c, o in self.outcomes.items()},
        }


# =====================================================================
# 辅助
# =====================================================================
def _fps_of(probes: Dict[str, VideoProbe]) -> float:
    """取整条 episode 的 fps。各路不一致就报错。

    saa / sudden_change 的窗口都是按秒给的, fps 错了整套阈值跟着错; 各路 fps 不同则
    "第 i 帧"在不同相机上根本不是同一时刻。这件事不能猜, 也不能取平均。
    """
    vals = {c: float(p.fps) for c, p in probes.items() if p.fps}
    if not vals:
        raise UnsupportedDataError("所有视频都没有可用的 fps, 按秒定义的窗口无从换算")
    uniq = sorted(set(round(v, 3) for v in vals.values()))
    if len(uniq) > 1:
        raise UnsupportedDataError(
            f"各路视频 fps 不一致: {vals}。'第 i 帧'在不同相机上不是同一时刻, "
            f"跨模态的帧区间失去意义")
    return uniq[0]


def _n_frames_of(probes: Dict[str, VideoProbe]) -> int:
    vals = {c: int(p.n_frames) for c, p in probes.items()}
    uniq = sorted(set(vals.values()))
    if len(uniq) > 1:
        raise UnsupportedDataError(
            f"各路视频帧数不一致: {vals}。corrupt 应已拦下, 这里不做截断对齐")
    return uniq[0]


def _profile_columns(section: dict) -> List[str]:
    return sorted({str(d["column"]) for d in (section.get("dims") or []) if "column" in d})


# =====================================================================
# 单条 episode 的编排
# =====================================================================
def run_episode(videos: Dict[str, Optional[bytes]],
                parquet_bytes: Optional[bytes] = None,
                episode_id: str = "",
                profile: Optional[dict] = None,
                assets: Optional[EpisodeAssets] = None,
                signals: Optional[Dict[str, Any]] = None,
                params: Optional[Dict[str, dict]] = None,
                checks: Sequence[str] = ALL_CHECKS,
                short_circuit: bool = False,
                ) -> EpisodeResult:
    """对一条 episode 跑完整流水线。**不抛异常, 不删数据。**

    videos:  {相机名: mp4 字节}, 值为 None 表示文件缺失 (corrupt 会报 VIDEO_FILE_MISSING)。
    parquet_bytes: lerobot parquet 字节。非 parquet 数据集改用 signals=。
    profile: dataset_profile.json 的内容, 需要含 saa / sudden_change / state_extreme 三节。
             缺哪节, 对应检测就报错并写明缺什么 —— 不猜维度。
    assets:  该 episode 的全部文件路径。**产生删帧时必须提供**, 否则工单列不全目标,
             会标成不可执行。
    signals: 预先取好的信号矩阵 {column: (T,D)}, 键名用 profile 里声明的 column。
             hdf5 / jsonl 数据集由格式适配器填这个参数。不给则从 parquet 取。
    short_circuit: 判定整条删除后是否跳过剩余检测。默认 False (跑满, 保留交叠统计),
             阈值冻结后再打开, 见模块说明。

    为什么这个函数不抛异常: 批量跑十万条时, 让每条自己带着结论回来, 比让调用方在循环里
    try/except 更不容易出错 —— 后者很容易被写成 `except: continue`, 于是检测失败被
    静默当成通过。失败信息在 failed_* 和 checks[x].error_* 里, 是结构化的。
    """
    params = params or {}
    profile = profile or {}
    result = EpisodeResult(episode_id=episode_id)

    if not videos:
        result.failed = True
        result.failed_stage = "probe"
        result.failed_error_type = "UnsupportedDataError"
        result.failed_error = "未提供任何视频"
        return result

    # ---- 阶段 0: 解码。整条流水线只在这里碰视频字节 ----
    try:
        probes: Dict[str, VideoProbe] = {
            cam: probe_video(b, cam, collect_gray=True) for cam, b in videos.items()}
    except CheckError as e:
        result.failed = True
        result.failed_stage = "probe"
        result.failed_error_type = type(e).__name__
        result.failed_error = str(e)
        return result

    def _record(name: str, fn) -> None:
        """跑一个检测, 把三种终态归位。异常带着具体原因进 JSON, 不做二次概括。"""
        stage = CHECK_STAGE[name]
        if short_circuit and result.drop_episode:
            result.outcomes[name] = CheckOutcome(
                check=name, stage=stage, status="skipped",
                skip_reason=f"short_circuit: 已被 {'+'.join(result.dropped_by)} 判整条删除")
            return
        try:
            result.outcomes[name] = CheckOutcome(
                check=name, stage=stage, status="ok", report=fn())
        except _Skip as e:
            result.outcomes[name] = CheckOutcome(
                check=name, stage=stage, status="skipped", skip_reason=str(e))
        except UnsupportedDataError as e:
            # 只影响这一个检测。原文照抄 —— "数据格式 'rosbag' 未知" 和 "profile 缺少 q1"
            # 是完全不同的问题, 归并成"数据错误"就没法定位了
            result.outcomes[name] = CheckOutcome(
                check=name, stage=stage, status="error",
                error_type=type(e).__name__, error_message=str(e))

    # ---- 阶段 1: corrupt (gate) ----
    if GATE in checks:
        from .corrupt import check_corrupt
        try:
            result.outcomes[GATE] = CheckOutcome(
                check=GATE, stage=1, status="ok",
                report=check_corrupt(videos, parquet_bytes, probes=probes))
        except CheckError as e:
            # gate 失败 = 后面全都没有立足点, 整条记为无结论
            result.failed = True
            result.failed_stage = GATE
            result.failed_error_type = type(e).__name__
            result.failed_error = str(e)
            return result

    try:
        result.n_frames = _n_frames_of(probes)
    except UnsupportedDataError:
        result.n_frames = max((int(p.n_frames) for p in probes.values()), default=0)

    # ---- 阶段 1: black / occlusion / stillness ----
    state, state_cols = None, None
    for name in STAGE1_VIDEO:
        if name not in checks:
            continue
        if name == "black":
            from .black import check_black
            _record(name, lambda: check_black(probes, params.get("black")))
        elif name == "occlusion":
            from .occlusion import check_occlusion
            _record(name, lambda: check_occlusion(probes, params.get("occlusion")))
        else:
            from .stillness import check_stillness, extract_state_matrix

            def _stillness():
                nonlocal state, state_cols
                if state is None:
                    state, state_cols = extract_state_matrix(parquet_bytes)
                return check_stillness(probes, state, params.get("stillness"),
                                       state_columns=state_cols)
            _record(name, _stillness)

    # ---- 阶段 2 的公共前提: fps 与信号矩阵 ----
    try:
        result.fps = _fps_of(probes)
        fps_error = None
    except UnsupportedDataError as e:
        fps_error = str(e)

    sig_cache: Dict[str, Any] = dict(signals) if signals else {}

    def _section(name: str) -> dict:
        """取出某个检测的 profile 段。缺失是错误, 主动关掉是跳过 —— 两者不能混为一谈。"""
        section = profile.get(name)
        if not section:
            raise UnsupportedDataError(
                f"profile 缺少 {name!r} 段。该检测要用哪些维度、阈值是多少, "
                f"必须由 profile 显式给出 —— 靠维度名猜是旧版一系列误判的根源")
        if section.get("enabled") is False:
            # 配置里明确关掉的, 归"主动跳过"。它和"跑了但零命中"必须在 JSON 里长得不一样,
            # 否则一个配错的 enabled=false 会被当成这批数据很干净。
            raise _Skip(f"profile 标记本数据集不做 {name} "
                        f"(reason: {section.get('reason', '未说明')})")
        return section

    def _need(section_name: str) -> Tuple[dict, Dict[str, Any]]:
        """取出某个检测的 profile 段和它需要的信号矩阵。缺什么报什么。"""
        section = _section(section_name)
        cols = _profile_columns(section)
        missing = [c for c in cols if c not in sig_cache]
        if missing:
            if signals is not None:
                raise UnsupportedDataError(
                    f"profile 声明的列 {missing} 不在传入的 signals 里 "
                    f"(实际: {sorted(sig_cache)})。非 parquet 数据集由格式适配器负责"
                    f"按 profile 的 column 名把信号填进来")
            from .signals import extract_signal_matrices
            sig_cache.update(extract_signal_matrices(parquet_bytes, missing))
        return section, {c: sig_cache[c] for c in cols}

    # ---- 阶段 2: saa (episode 级, 先跑) ----
    if "saa" in checks:
        from .saa import check_saa, extract_saa_matrices

        def _saa():
            if fps_error:
                raise UnsupportedDataError(fps_error)
            section = _section("saa")
            missing = [k for k in ("state_column", "action_column") if k not in section]
            if missing:
                raise UnsupportedDataError(
                    f"profile 的 saa 段缺少 {missing}。哪一列是 state、哪一列是 action, "
                    f"必须由人确认后写明 —— 靠列名猜正是 §1.8 那种错配的来源")
            st, ac = extract_saa_matrices(parquet_bytes,
                                          section["state_column"], section["action_column"])
            return check_saa(st, ac, section, result.fps, params.get("saa"))
        _record("saa", _saa)

    # ---- 阶段 2: sudden_change / state_extreme (帧级) ----
    if "sudden_change" in checks:
        from .sudden_change import check_sudden_change

        def _sc():
            if fps_error:
                raise UnsupportedDataError(fps_error)
            section, sig = _need("sudden_change")
            return check_sudden_change(sig, section, result.fps, params.get("sudden_change"))
        _record("sudden_change", _sc)

    if "state_extreme" in checks:
        from .state_extreme import check_state_extreme

        def _se():
            section, sig = _need("state_extreme")
            return check_state_extreme(sig, section, params.get("state_extreme"))
        _record("state_extreme", _se)

    # ---- 汇总删帧工单 ----
    result.manifest = _build_manifest(result, probes, assets)
    return result


def _build_manifest(result: EpisodeResult,
                    probes: Dict[str, VideoProbe],
                    assets: Optional[EpisodeAssets]) -> DeletionManifest:
    """把帧级检测的区间汇总成工单, 并判断现在能不能执行。"""
    sources: Dict[str, List[Tuple[int, int]]] = {}
    for name in STAGE2_FRAME:
        o = result.outcomes.get(name)
        if o is not None and o.status == "ok" and o.report.frames_to_drop:
            sources[name] = list(o.report.frames_to_drop)

    blocked = None
    if result.drop_episode:
        # 整条都不要了, 删哪几帧没有意义。区间仍然记下来, 便于回答"这条为什么被删"
        blocked = f"episode 已被 {'+'.join(result.dropped_by)} 判整条删除, 无需再删帧"
    else:
        corrupt = result.reports.get(GATE)
        if corrupt is not None:
            hit = [r for r in corrupt.triggered_rules if r in ALIGNMENT_RULES]
            if hit:
                blocked = (
                    f"corrupt 命中 {hit}: 各模态帧数对不上, '第 i 个视频帧 == 第 i 行' "
                    f"不成立, 帧区间没有确定的所指。照着删只会把错位坐实")

    return build_manifest(episode_id=result.episode_id, frame_sources=sources,
                          n_frames_total=result.n_frames, cameras=sorted(probes),
                          assets=assets, blocked_reason=blocked)


# =====================================================================
# 删除率预算: 交叠与唯一归因
# =====================================================================
@dataclass
class DropBudget:
    """一批 episode 的删除率预算。逐条 add(), 最后 format_report()。

    回答三个只看各检测命中率回答不了的问题:

      1. **总删除率** —— 七个检测的并集, 不是七个数相加。
      2. **两两交叠** —— 两个检测总是同时命中, 说明其中一个可能冗余。
      3. **唯一归因** —— 只有它命中、其他都没命中的条数。这个数才是"关掉这个检测能
         救回多少数据", 也是这个检测存在的真正理由。

    帧级删除另算 (frame_* 字段): state_extreme 的命中通常是散点, 总删帧量不大但会把
    episode 打成筛子。所以除了删帧率, 还统计**段数分布** —— 删 5 帧分散在 5 处, 是给
    轨迹开了 5 个口子; 删 5 帧集中在开头, 只是把 episode 截短了一点。
    """
    n_total: int = 0
    n_failed: int = 0
    n_dropped: int = 0
    per_check_hit: Dict[str, int] = field(default_factory=dict)
    per_check_only: Dict[str, int] = field(default_factory=dict)
    pair_overlap: Dict[str, int] = field(default_factory=dict)
    combo_counts: Dict[str, int] = field(default_factory=dict)
    rule_counts: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    skipped_counts: Dict[str, int] = field(default_factory=dict)
    failed_stages: Dict[str, int] = field(default_factory=dict)

    n_frame_edited: int = 0                  # 有帧要删的 episode 数
    n_frames_deleted: int = 0
    n_frames_seen: int = 0
    n_blocked_manifest: int = 0              # 有帧要删但现在不能删的
    segment_hist: Dict[str, int] = field(default_factory=dict)   # 段数 -> 条数

    def add(self, result: EpisodeResult) -> None:
        self.n_total += 1

        if result.failed:
            # 失败的既不算通过也不算删除。分母也不含它们 —— 把失败算成"通过"
            # 会让删除率显得比真实情况好看。
            self.n_failed += 1
            k = result.failed_stage or "?"
            self.failed_stages[k] = self.failed_stages.get(k, 0) + 1
            return

        for c, info in result.errors.items():
            k = f"{c}:{info['type']}"
            self.error_counts[k] = self.error_counts.get(k, 0) + 1
        for c in result.skipped:
            self.skipped_counts[c] = self.skipped_counts.get(c, 0) + 1
        for c, rules in result.triggered_rules.items():
            for r in rules:
                k = f"{c}.{r}"
                self.rule_counts[k] = self.rule_counts.get(k, 0) + 1

        self.n_frames_seen += result.n_frames
        m = result.manifest
        if m is not None and m.n_frames_deleted:
            if m.actionable:
                self.n_frame_edited += 1
                self.n_frames_deleted += m.n_frames_deleted
                k = str(min(m.n_segments, 10)) + ("+" if m.n_segments > 10 else "")
                self.segment_hist[k] = self.segment_hist.get(k, 0) + 1
            else:
                self.n_blocked_manifest += 1

        hits = result.dropped_by
        if not hits:
            return

        self.n_dropped += 1
        combo = "+".join(hits)
        self.combo_counts[combo] = self.combo_counts.get(combo, 0) + 1
        for c in hits:
            self.per_check_hit[c] = self.per_check_hit.get(c, 0) + 1
        if len(hits) == 1:
            self.per_check_only[hits[0]] = self.per_check_only.get(hits[0], 0) + 1
        for i, a in enumerate(hits):
            for b in hits[i + 1:]:
                self.pair_overlap[f"{a}&{b}"] = self.pair_overlap.get(f"{a}&{b}", 0) + 1

    @property
    def n_judged(self) -> int:
        """分母: 真正给出了结论的条数 (不含检测失败的)。"""
        return self.n_total - self.n_failed

    def to_dict(self) -> dict:
        d = max(self.n_judged, 1)
        return {
            "n_total": self.n_total,
            "n_failed": self.n_failed,
            "n_judged": self.n_judged,
            "n_dropped": self.n_dropped,
            "n_kept": self.n_judged - self.n_dropped,
            "drop_rate": round(self.n_dropped / d, 6),
            "per_check": {
                c: {"hit": self.per_check_hit.get(c, 0),
                    "hit_rate": round(self.per_check_hit.get(c, 0) / d, 6),
                    "only": self.per_check_only.get(c, 0),
                    # 关掉这个检测能救回多少数据 —— 它存在的真正理由
                    "only_rate": round(self.per_check_only.get(c, 0) / d, 6)}
                for c in ALL_CHECKS if c in self.per_check_hit},
            "pair_overlap": dict(sorted(self.pair_overlap.items())),
            "combo_counts": dict(sorted(self.combo_counts.items(), key=lambda kv: -kv[1])),
            "rule_counts": dict(sorted(self.rule_counts.items(), key=lambda kv: -kv[1])),
            "frame_deletion": {
                "n_episodes_edited": self.n_frame_edited,
                "n_frames_deleted": self.n_frames_deleted,
                "n_frames_seen": self.n_frames_seen,
                "frame_drop_rate": round(
                    self.n_frames_deleted / max(self.n_frames_seen, 1), 6),
                "n_blocked": self.n_blocked_manifest,
                # 删完留下多少处时间不连续。散点多说明阈值定得太紧, 该回去调参数
                # 而不是硬删。
                "segment_hist": dict(sorted(self.segment_hist.items())),
            },
            "error_counts": dict(sorted(self.error_counts.items(), key=lambda kv: -kv[1])),
            "skipped_counts": dict(sorted(self.skipped_counts.items())),
            "failed_stages": dict(sorted(self.failed_stages.items())),
        }

    def format_report(self) -> str:
        """给人看的文本摘要。跑完一批直接 print。"""
        d = max(self.n_judged, 1)
        L = [f"总计 {self.n_total} 条 | 检测失败 {self.n_failed} 条 (不计入分母) "
             f"| 有结论 {self.n_judged} 条",
             f"整条删除 {self.n_dropped} 条 = {self.n_dropped / d:.2%}   "
             f"保留 {self.n_judged - self.n_dropped} 条", ""]
        L.append(f"{'检测':<16}{'阶段':>4}{'命中':>8}{'命中率':>10}{'唯一归因':>10}{'关掉可救回':>12}")
        for c in ALL_CHECKS:
            if c not in self.per_check_hit:
                continue
            hit, only = self.per_check_hit.get(c, 0), self.per_check_only.get(c, 0)
            L.append(f"{c:<16}{CHECK_STAGE[c]:>4}{hit:>8}{hit / d:>9.2%}"
                     f"{only:>10}{only / d:>11.2%}")
        if self.pair_overlap:
            L.append("")
            L.append("两两交叠 (同时命中的条数):")
            for k, v in sorted(self.pair_overlap.items(), key=lambda kv: -kv[1]):
                L.append(f"  {k:<34}{v:>6}  {v / d:>7.2%}")
        if self.n_frame_edited or self.n_blocked_manifest:
            L.append("")
            L.append(f"帧级删除: {self.n_frame_edited} 条需要删帧, 共 "
                     f"{self.n_frames_deleted} 帧 / {self.n_frames_seen} 帧 = "
                     f"{self.n_frames_deleted / max(self.n_frames_seen, 1):.2%}")
            if self.segment_hist:
                L.append(f"  每条的不连续处数分布: {dict(sorted(self.segment_hist.items()))}")
            if self.n_blocked_manifest:
                L.append(f"  其中 {self.n_blocked_manifest} 条现在不能删 "
                         f"(corrupt 未过 / 整条已删)")
        if self.rule_counts:
            L.append("")
            L.append("触发规则分布:")
            for k, v in sorted(self.rule_counts.items(), key=lambda kv: -kv[1]):
                L.append(f"  {k:<44}{v:>6}")
        if self.error_counts:
            L.append("")
            L.append("检测报错分布 (该检测无结论, 其余照常):")
            for k, v in sorted(self.error_counts.items(), key=lambda kv: -kv[1]):
                L.append(f"  {k:<44}{v:>6}")
        if self.failed_stages:
            L.append("")
            L.append(f"整条失败分布: {dict(self.failed_stages)}")
        if self.skipped_counts:
            L.append(f"主动跳过分布: {dict(self.skipped_counts)}")
        return "\n".join(L)
