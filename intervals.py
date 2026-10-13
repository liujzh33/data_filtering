"""帧区间 [start, end) 的工具函数。各检测共用。

约定: 区间一律左闭右开, 即 (3, 7) 表示第 3/4/5/6 帧, 共 4 帧。

为什么单独成模块: 旧版把"一批坏帧"压缩成 (min, max) 一个大区间, 于是第 5 帧和第 1050 帧
两个孤立坏帧会被记成 (5, 1050) —— 1045 帧全被删掉。这里的 indices_to_ranges 逐段切分,
坏多少删多少。
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "indices_to_ranges",
    "merge_ranges",
    "ranges_length",
    "intersect_ranges",
    "intersect_many",
    "longest_range",
    "longest_range_span",
    "ranges_to_indices",
]


def indices_to_ranges(indices: Iterable[int]) -> List[Tuple[int, int]]:
    """把散列的帧号压成**逐个连续段** [start, end)。

    [5, 6, 7, 100, 101] -> [(5, 8), (100, 102)]

    注意不是 (min, max): 上例绝不能压成 (5, 102), 那会连带删掉中间 92 帧好数据。
    """
    idx = sorted({int(i) for i in indices})
    if not idx:
        return []
    out: List[Tuple[int, int]] = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            out.append((start, prev + 1))
            start = prev = i
    out.append((start, prev + 1))
    return out


def merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """合并重叠或相邻的区间, 返回排序后的不相交区间列表。空/非法区间被丢弃。"""
    rs = sorted((int(a), int(b)) for a, b in ranges if b > a)
    if not rs:
        return []
    out: List[List[int]] = [list(rs[0])]
    for a, b in rs[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def ranges_length(ranges: Sequence[Tuple[int, int]]) -> int:
    """区间总长度 (帧数)。假定区间不重叠; 不确定时先 merge_ranges。"""
    return sum(b - a for a, b in ranges if b > a)


def longest_range(ranges: Sequence[Tuple[int, int]]) -> int:
    """最长单个区间的长度。空列表返回 0。

    black / occlusion / stillness 的删除判据都用这个值 (而非总长度) 去比阈值:
    要拦的是"持续一段时间的异常", 不是"零星异常的累计量"。
    """
    return max((b - a for a, b in ranges if b > a), default=0)


def longest_range_span(ranges: Sequence[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """最长单个区间本身 (start, end)。空列表返回 None。

    和 longest_range 配套: 后者给长度用于判定, 这个给位置用于排查 ——
    命中之后第一件事通常是去看那一段到底发生了什么。
    """
    valid = [(a, b) for a, b in ranges if b > a]
    if not valid:
        return None
    return max(valid, key=lambda r: r[1] - r[0])


def intersect_ranges(a: Sequence[Tuple[int, int]],
                     b: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """两组区间的交集 (双指针, O(n+m))。输入会先各自 merge。"""
    ra, rb = merge_ranges(a), merge_ranges(b)
    out: List[Tuple[int, int]] = []
    i = j = 0
    while i < len(ra) and j < len(rb):
        lo = max(ra[i][0], rb[j][0])
        hi = min(ra[i][1], rb[j][1])
        if lo < hi:
            out.append((lo, hi))
        # 先结束的那个往前走
        if ra[i][1] < rb[j][1]:
            i += 1
        else:
            j += 1
    return out


def intersect_many(groups: Sequence[Sequence[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    """多组区间的交集。任一组为空则结果为空。groups 为空时返回空。

    用途: "所有相机同时黑"的帧区间 = 各相机黑帧区间的交集。
    """
    if not groups:
        return []
    acc = merge_ranges(groups[0])
    for g in groups[1:]:
        if not acc:
            return []
        acc = intersect_ranges(acc, g)
    return acc


def ranges_to_indices(ranges: Sequence[Tuple[int, int]]) -> List[int]:
    """展开成帧号列表。只在区间总长可控时使用 (调试/小规模)。"""
    out: List[int] = []
    for a, b in merge_ranges(ranges):
        out.extend(range(a, b))
    return out
