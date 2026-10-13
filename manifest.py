"""删帧清单: 记录"这条 episode 的哪些帧要删、要从哪些文件里删"。

**本模块不删任何东西。** 整条流水线只做检测和记录, 清单是交给下游写出器的工单。
这样分开有个很实际的理由: 检测可以重跑一万次不改数据, 而删除是不可逆的。把两件事
放进同一个函数, 意味着每次调阈值都在拿真实数据冒险。

--------------------------------------------------------------------------
为什么删一帧要牵动这么多文件
--------------------------------------------------------------------------

一条 episode 的第 i 帧不是一个东西, 是**一组**东西:

    videos/head/episode_000012.mp4        第 i 个视频帧
    videos/left_wrist/episode_000012.mp4  第 i 个视频帧
    videos/right_wrist/episode_000012.mp4 第 i 个视频帧
    data/chunk-000/episode_000012.parquet 第 i 行 (state / action / timestamp / ...)

它们靠**下标对齐**, 没有别的锚点。所以删第 i 帧必须四个地方同时删, 少删一个, 从第 i 帧
往后所有模态就整体错位一帧 —— 而且错位之后每个文件自己看都是自洽的 (长度合法、时间戳
连续), 没有任何检测能事后发现。**一次失败的删帧, 等于给数据集注入一个 SAA 级别的损坏。**

正因如此, 清单必须**逐个列出**所有目标, 而不是只写一句"删第 i 帧"。写出器照单执行,
执行完逐项核对。

--------------------------------------------------------------------------
下标对齐的前提: corrupt 必须先过
--------------------------------------------------------------------------

"第 i 个视频帧对应第 i 行"这件事本身是 corrupt 检测的结论, 不是天然成立的。各路视频
帧数不一致、或视频帧数与表格行数对不上时, 下标映射根本不存在, 此时任何帧区间都是
无意义的。所以清单带一个 actionable 标志:

    corrupt 未通过 (尤其是 VIEW_FRAME_COUNT_MISMATCH / ACTION_VIDEO_LENGTH_MISMATCH)
        -> actionable=False, 区间只作诊断

    episode 已被判整条删除
        -> actionable=False, 整条都不要了, 删帧没意义

只有两个条件都满足, 清单才是一张可执行的工单。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import UnsupportedDataError
from .intervals import merge_ranges, ranges_length

__all__ = ["DATA_FORMATS", "EpisodeAssets", "DeletionManifest", "build_manifest"]


# 表格类数据的格式。删帧的物理代价在这三者之间差别很大, 写出器必须知道自己在处理哪种,
# 所以这里不接受"未知"。
DATA_FORMATS = {
    "lerobot": (
        "LeRobot (parquet + 独立 mp4)。删行之外还要重算本条的 frame_index/timestamp、"
        "全局 index、meta/episodes.jsonl 的 length、以及 episodes_stats 里的统计量 —— "
        "全局 index 是跨 episode 连续的, 少一行会让**后面所有 episode** 的索引整体前移"),
    "parquet": (
        "裸 parquet。删行本身容易 (重写整个文件), 但 timestamp/frame_index 这类自增列 "
        "要跟着重算, 否则时间轴上会留下一个看不见的跳变"),
    "hdf5": (
        "HDF5。h5py **不能原地删行**, 必须把每个 dataset 读出来、切片、重新写一个文件。"
        "固定长度的 dataset 要重建, 属性和软链接要一并搬过去"),
}


@dataclass
class EpisodeAssets:
    """一条 episode 在磁盘/对象存储上的全部文件。删帧清单靠它列出目标。

    **必须显式提供。** 让检测自己去猜文件在哪, 等于把"删哪个文件"这种不可逆的决定
    交给字符串拼接。
    """
    data_format: str = ""                                   # lerobot / parquet / hdf5
    table_path: str = ""                                    # 动作数据文件
    video_paths: Dict[str, str] = field(default_factory=dict)   # 相机名 -> mp4 路径
    meta_paths: Dict[str, str] = field(default_factory=dict)    # 需要连带更新的元数据文件

    def validate(self, cameras: Sequence[str]) -> None:
        """校验清单目标与实际检测的对象一致。

        cameras: 本次真正跑了检测的相机名。两边必须**完全一致** —— 少一路意味着有一路
        视频不会被删帧 (直接制造错位), 多一路意味着 profile 和数据对不上。
        """
        if self.data_format not in DATA_FORMATS:
            raise UnsupportedDataError(
                f"数据格式 {self.data_format!r} 未知; 支持的格式: {sorted(DATA_FORMATS)}。"
                f"删帧的物理代价在这几种格式之间差别很大, 写出器必须知道自己在处理哪种")
        if not self.table_path:
            raise UnsupportedDataError(
                f"缺少 table_path (动作数据文件路径)。删视频帧却不删对应的动作行, "
                f"会让这条 episode 从该帧起整体错位一帧")
        have, want = set(self.video_paths), set(cameras)
        if have != want:
            raise UnsupportedDataError(
                f"video_paths 的相机集合与实际检测的不一致: "
                f"多出 {sorted(have - want)}, 缺少 {sorted(want - have)}。"
                f"缺的那一路不会被删帧, 从该帧起它就和其他模态错开了")


@dataclass
class DeletionManifest:
    """删帧工单。**不含任何删除动作**, 只说明删什么、从哪删、现在能不能删。"""
    episode_id: str = ""
    actionable: bool = False
    blocked_reason: Optional[str] = None
    ranges: List[Tuple[int, int]] = field(default_factory=list)   # 左闭右开, 已合并
    sources: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)  # check -> 区间
    n_frames_total: int = 0
    assets: Optional[EpisodeAssets] = None

    @property
    def n_frames_deleted(self) -> int:
        return ranges_length(self.ranges)

    @property
    def delete_ratio(self) -> float:
        return self.n_frames_deleted / self.n_frames_total if self.n_frames_total else 0.0

    @property
    def n_segments(self) -> int:
        """删完之后 episode 上会出现多少处时间不连续。

        这个数比"删了多少帧"更能说明后果: 删 5 帧但分散在 5 处, 是给轨迹开了 5 个口子;
        删 5 帧集中在开头, 只是把 episode 截短了一点。
        """
        return len(self.ranges)

    def targets(self) -> List[dict]:
        """逐个列出必须同步删除的对象。写出器照单执行, 执行完逐项核对。"""
        if self.assets is None:
            return []
        out = [{"kind": "video", "camera": cam, "path": p,
                "action": "删除这些帧后重新编码", "frame_ranges": [list(r) for r in self.ranges]}
               for cam, p in sorted(self.assets.video_paths.items())]
        out.append({"kind": "table", "format": self.assets.data_format,
                    "path": self.assets.table_path,
                    "action": "删除对应行 " + DATA_FORMATS[self.assets.data_format],
                    "frame_ranges": [list(r) for r in self.ranges]})
        out += [{"kind": "meta", "name": n, "path": p,
                 "action": "连带更新 (长度/索引/统计量)"}
                for n, p in sorted(self.assets.meta_paths.items())]
        return out

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "actionable": self.actionable,
            "blocked_reason": self.blocked_reason,
            "frame_ranges": [list(r) for r in self.ranges],
            "n_frames_deleted": self.n_frames_deleted,
            "n_frames_total": self.n_frames_total,
            "delete_ratio": round(self.delete_ratio, 6),
            # 删完会留下多少处时间不连续 —— 比总帧数更能说明后果
            "n_segments": self.n_segments,
            "sources": {k: [list(r) for r in v] for k, v in sorted(self.sources.items())},
            "data_format": self.assets.data_format if self.assets else None,
            # 同一 episode 里必须**一起**删的全部对象。少删一个就是制造整体错位,
            # 而错位之后每个文件自己看都是自洽的, 没有任何检测能事后发现。
            "targets": self.targets(),
        }


def build_manifest(episode_id: str,
                   frame_sources: Dict[str, List[Tuple[int, int]]],
                   n_frames_total: int,
                   cameras: Sequence[str],
                   assets: Optional[EpisodeAssets] = None,
                   blocked_reason: Optional[str] = None) -> DeletionManifest:
    """汇总各检测的删帧区间, 生成工单。

    frame_sources: {check 名: 该检测要求删除的区间}。**取并集** —— 同一帧被两个检测
        命中只删一次, 但两个来源都记下来, 事后能回答"这一帧是因为什么被删的"。
    blocked_reason: 非空表示现在不能删 (corrupt 未过 / 整条已被删), 区间仅作诊断。
    """
    merged = merge_ranges([r for rs in frame_sources.values() for r in rs])

    for name, rs in frame_sources.items():
        for a, b in rs:
            if not (0 <= a < b <= n_frames_total):
                raise UnsupportedDataError(
                    f"{name} 给出的区间 [{a}, {b}) 超出 episode 范围 [0, {n_frames_total})。"
                    f"区间是要拿去删真实文件的, 越界必须当场停下")

    m = DeletionManifest(episode_id=episode_id, ranges=merged,
                         sources={k: merge_ranges(v) for k, v in frame_sources.items() if v},
                         n_frames_total=n_frames_total,
                         blocked_reason=blocked_reason)

    if not merged:
        m.actionable = False
        m.blocked_reason = blocked_reason      # 没有要删的, 谈不上可不可执行
        return m
    if blocked_reason:
        m.actionable = False
        return m

    if assets is None:
        m.actionable = False
        m.blocked_reason = (
            "未提供 EpisodeAssets, 无法列出要删的文件。删帧必须同步作用于该 episode 的"
            "所有视频和动作数据, 清单里列不全就不能执行")
        return m

    assets.validate(cameras)
    m.assets = assets
    m.actionable = True
    return m
