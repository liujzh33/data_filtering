"""单路视频的**唯一一次**解码, 产出所有检测需要的逐帧测量值。

设计目标:

  1. **只解码一次**。旧版 corrupt 走 decode_frame_stats (rgb24), black/occlusion 走
     decode_gray_stats (gray), 同一个视频被完整解码两遍。这里一次遍历同时产出
     尺寸/坏帧位置和逐帧灰度均值/标准差。

  2. **帧号可信**。用每帧的 PTS (显示时间戳) 换算真实帧号:

         帧号 = round((pts - start_time) * time_base * fps)

     即使中间有包解码失败, 后续帧的帧号仍然落在原始时间轴的正确位置上。
     旧版 metrics.decode_gray_stats 是"解码成功才 append, 但计数器照常 +1", 于是
     **数组下标不等于帧号** —— 第 50 帧失败, means[50] 装的是第 51 帧的值, 后面全部
     左移。下游拿这个下标当帧号去删 parquet 行, 删的是错行。

  3. **坏帧用 NaN 占位, 不用外挂 mask**。gray_mean / gray_std 的长度恒等于时间轴长度
     n_frames, 解不出来的帧填 NaN。于是:
       - 下标恒等于帧号
       - "这帧没有测量值"由 NaN 直接表达, 后续检测跳过 NaN 即可
       - 不再需要 ViewMetrics.exclude_mask。旧的 mask 机制有两个毛病: 无 frame_range 的
         Issue 会把整路相机整列抹掉 (该路永远不再报黑屏); 而且 mask 减完之后才统计
         "同时黑几路", 于是 corrupt 的一条 issue 会悄悄改变 black 对**其他**相机的分级。

  4. **处理不了就报错**。见 errors.py。本模块的报错项:
       - pyav 不可用 / numpy 不可用     -> PipelineEnvironmentError
       - 容器没给 fps 或 time_base      -> UnsupportedDataError
       - 有帧拿不到 PTS / 帧号非严格递增 -> UnsupportedDataError
       - 源像素格式不在白名单 / 中途变化 -> UnsupportedDataError
     "视频打不开"**不报错** —— 那是数据坏了, 记进 open_error 交给规则处理。
"""

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .errors import PipelineEnvironmentError, UnsupportedDataError
from .intervals import indices_to_ranges

__all__ = ["VideoProbe", "probe_video", "SUPPORTED_PIXEL_FORMATS", "LUMA_WEIGHTS"]


# 支持的源像素格式白名单。
# 不在表里的一律报错, 不做静默转换 —— frame.to_ndarray(format="rgb24") 是**强制**转换,
# 源视频哪怕是 gray (真灰度) 也会被转成三个相同通道后当彩色放行, 检测端看不出来。
# 白名单先收紧到常见彩色格式, 跑过实际数据、确认还有哪些格式之后再逐步放宽。
SUPPORTED_PIXEL_FORMATS = frozenset({
    "yuv420p", "yuvj420p",      # H.264 最常见 (yuvj = full range)
    "yuv422p", "yuvj422p",
    "yuv444p", "yuvj444p",
    "yuv420p10le",              # 10-bit
    "nv12", "nv21",
    "rgb24", "bgr24", "rgba", "bgra",
})

# BT.601 亮度权重 (R, G, B)。
# 从 rgb24 数组自己算灰度, 而不是再调一次 to_ndarray(format="gray"):
# 少一次 swscale 转换, 且结果不受源视频 color_range 元数据缺失的影响 (rgb24 转换时
# 已经做过 range 归一化)。
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


@dataclass
class VideoProbe:
    """单路视频一次解码产出的全部测量值。只描述"测到了什么", 不做任何判定。"""

    cam: str

    # ---- 整路状态 ----
    missing_file: bool = False            # True = 调用方没给视频字节 (文件缺失)
    open_error: Optional[str] = None      # 非 None = 整路打不开 (数据坏了, 不是报错项)

    # ---- 容器属性 ----
    fps: float = 0.0
    pix_fmt: Optional[str] = None         # 源像素格式; 不在白名单会直接报错
    n_declared: int = 0                   # 容器自称的帧数 (stream.frames); 0 = 容器没写

    # ---- 帧计数 ----
    n_readable: int = 0                   # 实际成功读出的帧数
    n_frames: int = 0                     # 时间轴长度 = max(n_declared, 最大帧号 + 1)
    missing_ranges: List[Tuple[int, int]] = field(default_factory=list)  # 读不出来的帧, 逐段
    n_missing: int = 0

    # ---- 画面尺寸 ----
    dominant_size: Optional[Tuple[int, int]] = None  # (h, w) 出现帧数最多的尺寸 (多数派, 非首帧)
    size_histogram: Dict[str, int] = field(default_factory=dict)
    size_mismatch_ranges: List[Tuple[int, int]] = field(default_factory=list)

    # ---- 解码错误 ----
    packet_errors: int = 0
    first_error: Optional[str] = None

    # ---- 逐帧灰度 (长度恒 = n_frames, 坏帧为 NaN); collect_gray=False 时为 None ----
    gray_mean: Optional[Any] = None       # np.ndarray (n_frames,) float32
    gray_std: Optional[Any] = None        # np.ndarray (n_frames,) float32
    # 与**前一帧**的平均绝对像素差 (灰度级)。frame_diff[i] = mean(|frame[i] - frame[i-1]|)。
    # 只在 i 和 i-1 都成功解码**且帧号连续**时才有值, 否则 NaN (差分跨过空洞没有意义)。
    # frame_diff[0] 恒为 NaN。
    #
    # 为什么需要它: 判"画面有没有变"必须比像素, 不能比整帧灰度均值。机械臂在画面里移动时,
    # 它占的像素少, 整帧均值几乎不变 —— 旧版 stillness 用 |Δmean|+|Δstd| 判图像静止,
    # 会把"机械臂在动"误判成静止。
    frame_diff: Optional[Any] = None      # np.ndarray (n_frames,) float32

    @property
    def usable(self) -> bool:
        """这一路是否产出了可用的逐帧测量值 (文件在、能打开、至少读出一帧)。"""
        return not self.missing_file and not self.open_error and self.n_readable > 0

    def to_dict(self) -> dict:
        return {
            "cam": self.cam,
            "missing_file": self.missing_file,
            "open_error": self.open_error,
            "fps": round(self.fps, 4) if self.fps else 0.0,
            "pix_fmt": self.pix_fmt,
            "n_declared": self.n_declared,
            "n_readable": self.n_readable,
            "n_frames": self.n_frames,
            "n_missing": self.n_missing,
            "missing_ranges": [list(r) for r in self.missing_ranges],
            "dominant_size": list(self.dominant_size) if self.dominant_size else None,
            "size_histogram": self.size_histogram,
            "size_mismatch_ranges": [list(r) for r in self.size_mismatch_ranges],
            "packet_errors": self.packet_errors,
            "first_error": self.first_error,
        }


def probe_video(video_bytes: Optional[bytes], cam: str = "",
                collect_gray: bool = True) -> VideoProbe:
    """解码一路 mp4, 测出帧数 / 坏帧位置 / 尺寸变化 / 逐帧灰度。整条流水线只在这里解码。

    collect_gray: True 时额外产出 gray_mean / gray_std (black / occlusion 需要)。
        只跑 corrupt 时可以关掉省一点计算 —— 但解码本身才是大头, 一般不必关。

    返回 VideoProbe。**不做任何判定** —— 是不是"坏"、要不要删, 由各检测的规则决定。

    报错项见模块 docstring。特别注意 PTS: 退化成顺序计数之后仍然会输出帧区间, 下游拿去
    删 parquet 对应行会**删错行**, 且删完长度依然对得上, 事后无法发现。宁可停下来。
    """
    probe = VideoProbe(cam=cam)
    if not video_bytes:
        # 文件缺失是一种数据状况 (走规则), 不是"文件损坏", 更不是环境问题
        probe.missing_file = True
        return probe

    try:
        import av
    except Exception as e:  # pragma: no cover - 环境问题
        raise PipelineEnvironmentError(
            f"pyav 不可用, 无法解码视频: {e}。这是环境问题, 与数据无关; "
            f"绝不能把它判成'视频损坏'") from e
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - 环境问题
        raise PipelineEnvironmentError(f"numpy 不可用: {e}") from e

    try:
        container = av.open(io.BytesIO(video_bytes))
    except Exception as e:
        # 打不开是数据坏了, 交给规则 (VIDEO_UNDECODABLE), 不抛异常
        probe.open_error = f"{type(e).__name__}: {e}"
        return probe

    luma = np.array(LUMA_WEIGHTS, dtype=np.float32)
    seq: List[dict] = []
    prev_gray = None          # 上一个成功解码帧的灰度图 (只留一帧, 算完差分即丢)
    prev_pos = None           # 它在 seq 里的下标, 用于事后检查帧号是否连续
    try:
        streams = container.streams.video
        if not streams:
            probe.open_error = "容器中没有视频流"
            return probe
        stream = streams[0]

        # fps / time_base 是按 PTS 定位帧号的必要条件, 缺一不可 -> 缺就报错。
        # 旧版 black 用的是 `fps = metrics.fps or 30.0`: 拿不到就当 30fps, 而该检测**所有**
        # 阈值都是秒, 10fps 的视频等于全部阈值收紧 3 倍, 且完全看不出来。
        try:
            rate = stream.average_rate
        except Exception as e:
            raise UnsupportedDataError(f"[{cam}] 读取 average_rate 失败: {e}") from e
        if not rate or float(rate) <= 0:
            raise UnsupportedDataError(
                f"[{cam}] 容器未提供有效帧率 (average_rate={rate}), 无法按 PTS 定位帧号, "
                f"也无法把秒级阈值换算成帧")
        probe.fps = float(rate)

        time_base = getattr(stream, "time_base", None)
        if not time_base:
            raise UnsupportedDataError(f"[{cam}] 容器未提供 time_base, 无法按 PTS 定位帧号")
        start_time = getattr(stream, "start_time", None) or 0

        # stream.frames 允许为 0 (很多 mp4 不写这个字段), 时间轴长度会由 PTS 推出来。
        # 这不是"猜默认值": PTS 给出的帧号本身就是准的, 只是尾部截断可能测不到。
        try:
            probe.n_declared = int(stream.frames) if stream.frames and stream.frames > 0 else 0
        except Exception:
            probe.n_declared = 0

        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except Exception as e:
                probe.packet_errors += 1
                if probe.first_error is None:
                    probe.first_error = f"packet.decode: {type(e).__name__}: {e}"
                continue  # 不手工推进帧号, 空洞由 PTS 比对自然显现
            for frame in frames:
                # 源像素格式校验必须在 to_ndarray 之前: rgb24 是**强制**转换, 源视频是
                # gray 也会被转成三个相同通道后当彩色放行。
                pix_fmt = getattr(getattr(frame, "format", None), "name", None)
                if pix_fmt is None:
                    raise UnsupportedDataError(f"[{cam}] 读不到源像素格式")
                if pix_fmt not in SUPPORTED_PIXEL_FORMATS:
                    raise UnsupportedDataError(
                        f"[{cam}] 源像素格式 {pix_fmt!r} 不在支持白名单 "
                        f"{sorted(SUPPORTED_PIXEL_FORMATS)}。不做静默转换; "
                        f"确认这种格式该怎么处理之后再把它加进白名单")
                if probe.pix_fmt is None:
                    probe.pix_fmt = pix_fmt
                elif pix_fmt != probe.pix_fmt:
                    raise UnsupportedDataError(
                        f"[{cam}] 源像素格式中途从 {probe.pix_fmt!r} 变为 {pix_fmt!r}")

                rec = {"pts": getattr(frame, "pts", None), "ok": False,
                       "h": 0, "w": 0, "gm": None, "gs": None,
                       "fd": None, "fd_prev": None}
                try:
                    arr = frame.to_ndarray(format="rgb24")   # (H, W, 3) uint8
                    rec["ok"] = True
                    rec["h"] = int(arr.shape[0])
                    rec["w"] = int(arr.shape[1])
                    if collect_gray:
                        gray = arr @ luma                    # (H, W) float32
                        rec["gm"] = float(gray.mean())
                        rec["gs"] = float(gray.std())
                        # 与上一个成功解码帧的平均绝对像素差。边解码边算, 只留一帧 ——
                        # 把全部灰度数组存下来再统一算的话, 1500 帧 480x640 float32 就是 1.8 GB。
                        # PTS 已被要求严格递增, 所以"解码顺序的上一帧"就是"显示顺序的上一帧"。
                        if prev_gray is not None and prev_gray.shape == gray.shape:
                            rec["fd"] = float(np.abs(gray - prev_gray).mean())
                            rec["fd_prev"] = prev_pos
                        prev_gray = gray
                        prev_pos = len(seq)
                except Exception as e:
                    if probe.first_error is None:
                        probe.first_error = f"to_ndarray: {type(e).__name__}: {e}"
                seq.append(rec)
    finally:
        try:
            container.close()
        except Exception:
            pass

    if not seq:
        # 能 open 但一帧都没解出来
        probe.n_readable = 0
        probe.n_frames = probe.n_declared
        if probe.n_declared > 0:
            probe.missing_ranges = [(0, probe.n_declared)]
            probe.n_missing = probe.n_declared
        if collect_gray:
            probe.gray_mean = np.full(probe.n_frames, np.nan, dtype=np.float32)
            probe.gray_std = np.full(probe.n_frames, np.nan, dtype=np.float32)
            probe.frame_diff = np.full(probe.n_frames, np.nan, dtype=np.float32)
        return probe

    _assign_frame_indices(probe, seq, start_time, time_base)

    readable_idx = [r["idx"] for r in seq if r["ok"]]
    probe.n_readable = len(readable_idx)
    probe.n_frames = max(probe.n_declared, max(r["idx"] for r in seq) + 1)

    # 缺帧 = 时间轴上所有帧 - 成功读出的帧。
    # 同时覆盖三种情况: 转换失败的帧 / 坏包造成的中间空洞 / 尾部截断 (声明比实读多)。
    readable_set = set(readable_idx)
    missing = [i for i in range(probe.n_frames) if i not in readable_set]
    probe.missing_ranges = indices_to_ranges(missing)
    probe.n_missing = len(missing)

    _assign_dominant_size(probe, seq)

    if collect_gray:
        gm = np.full(probe.n_frames, np.nan, dtype=np.float32)
        gs = np.full(probe.n_frames, np.nan, dtype=np.float32)
        fd = np.full(probe.n_frames, np.nan, dtype=np.float32)
        for r in seq:
            if not r["ok"] or r["gm"] is None:
                continue
            gm[r["idx"]] = r["gm"]
            gs[r["idx"]] = r["gs"]
            # 只有帧号真正相邻才保留差分。中间隔了坏帧 (帧号跳了) 的话, 这个差分是
            # 跨空洞比出来的, 数值上偏大且没有物理意义 -> 留 NaN。
            if r["fd"] is not None and seq[r["fd_prev"]]["idx"] == r["idx"] - 1:
                fd[r["idx"]] = r["fd"]
        probe.gray_mean = gm
        probe.gray_std = gs
        probe.frame_diff = fd

    return probe


def _assign_frame_indices(probe: VideoProbe, seq: List[dict],
                          start_time: int, time_base) -> None:
    """按 PTS 给每帧算真实帧号, 写进 rec["idx"]。

    任何一个前提不满足都报错, 不退化为顺序计数 —— 退化之后帧号不再对应原始时间轴,
    下游会按错误帧号去删 parquet 行, 而且删完动作数和帧数依然对得上, 事后查不出来。
    """
    cam = probe.cam
    n_no_pts = sum(1 for r in seq if r["pts"] is None)
    if n_no_pts:
        raise UnsupportedDataError(
            f"[{cam}] {n_no_pts}/{len(seq)} 帧没有 PTS, 帧号无法可信定位")

    for r in seq:
        sec = float((r["pts"] - start_time) * time_base)
        r["idx"] = int(round(sec * probe.fps))

    idxs = [r["idx"] for r in seq]
    if idxs[0] < 0:
        raise UnsupportedDataError(
            f"[{cam}] 首帧 PTS 换算出负帧号 ({idxs[0]}), start_time={start_time} 可能不对")
    bad = [(a, b) for a, b in zip(idxs, idxs[1:]) if b <= a]
    if bad:
        raise UnsupportedDataError(
            f"[{cam}] PTS 换算出的帧号非严格递增 (共 {len(bad)} 处, 首处 {bad[0]}), "
            f"可能是可变帧率或 B 帧重排; 帧号无法可信定位")


def _assign_dominant_size(probe: VideoProbe, seq: List[dict]) -> None:
    """以**出现帧数最多**的 (h, w) 为基准尺寸, 不是首帧。

    首帧作基准的话, 遇到"只有首帧异常"的视频 (编码器吐出一个异常首帧) 会把结论整个反
    过来: 基准变成那个异常尺寸, 其余 1499 帧全被标成不符, 报告说这路是 240x320、坏了
    1499 帧, 而实际是 480x640、只坏了 1 帧。最终动作 (丢整条) 虽然一样, 诊断信息是错的。
    """
    counts: Dict[Tuple[int, int], int] = {}
    for r in seq:
        if r["ok"]:
            key = (r["h"], r["w"])
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return
    # 并列时取先出现的 (dict 保序 + max 返回首个最大值), 结果确定
    base = max(counts.items(), key=lambda kv: kv[1])[0]
    probe.dominant_size = base
    probe.size_histogram = {f"{h}x{w}": n for (h, w), n in
                            sorted(counts.items(), key=lambda kv: -kv[1])}
    probe.size_mismatch_ranges = indices_to_ranges(
        [r["idx"] for r in seq if r["ok"] and (r["h"], r["w"]) != base])
