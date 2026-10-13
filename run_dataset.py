"""在一个 LeRobot 数据集上跑完整流水线。

    python -m check_pipeline.run_dataset \
        --root    /path/to/dataset \
        --profile dataset_profile.json \
        --out     qc_out/ \
        --limit   50                     # 先跑 50 条看看

**不删任何数据。** 每条 episode 产出一个 result.json (含删帧工单), 最后一份 summary.json
和一份人看的报告。真正执行删除的写出器是另一个模块, 还没写。

--------------------------------------------------------------------------
路径一律从 meta/info.json 的模板解析
--------------------------------------------------------------------------

**不硬编码 chunk-000。** LeRobot 用 `chunks_size` 把 episode 分桶, 超过一个 chunk 的
数据集如果只读 chunk-000, 会静默漏掉后面全部数据 —— 而且跑出来的报告看起来一切正常,
只是条数少。同理相机名取自 info.json 里 dtype == "video" 的 feature, 不从目录名猜。

--------------------------------------------------------------------------
换数据格式要改的是这里, 不是检测模块
--------------------------------------------------------------------------

七个检测都是 profile 驱动的, 换数据集只换 profile。真正和格式耦合的只有本文件的
读取层 (`read_bytes` / `iter_episodes`) 和非 parquet 数据集的信号提取。

hdf5 / jsonl 数据集需要自己写一个同构的入口: 把信号读成 {profile 里声明的 column 名:
(T, D) 矩阵}, 用 `signals=` 传给 run_episode。检测模块一行都不用改。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .manifest import EpisodeAssets
from .pipeline import DropBudget, run_episode


# =====================================================================
# 读取层: 换存储换这里
# =====================================================================
def read_bytes(path: str) -> Optional[bytes]:
    """读一个文件成 bytes。文件不存在返回 None (交给 corrupt 报 VIDEO_FILE_MISSING)。"""
    if path.startswith("obs://"):
        import moxing as mox                     # 只在真用到时才 import
        if not mox.file.exists(path):
            return None
        with mox.file.File(path, "rb") as f:
            return f.read()
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def read_json(path: str) -> dict:
    b = read_bytes(path)
    if b is None:
        raise FileNotFoundError(path)
    return json.loads(b.decode("utf-8"))


def read_jsonl(path: str) -> List[dict]:
    b = read_bytes(path)
    if b is None:
        return []
    return [json.loads(x) for x in b.decode("utf-8").splitlines() if x.strip()]


# =====================================================================
# 数据集布局
# =====================================================================
class Layout:
    """从 meta/info.json 解析出来的布局。所有路径都由模板生成, 不拼字符串。"""

    def __init__(self, root: str):
        self.root = root.rstrip("/")
        self.info = read_json(f"{self.root}/meta/info.json")

        for key in ("data_path", "video_path", "fps", "total_episodes"):
            if key not in self.info:
                raise SystemExit(
                    f"info.json 缺少 {key!r}。路径和帧率必须从 info.json 取 —— "
                    f"硬编码 chunk-000 会在多 chunk 数据集上静默漏读")
        self.data_tpl = self.info["data_path"]
        self.video_tpl = self.info["video_path"]
        self.fps = float(self.info["fps"])
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.total = int(self.info["total_episodes"])

        # 相机 = dtype 为 video 的 feature。不从目录名猜。
        self.cameras = sorted(n for n, f in self.info.get("features", {}).items()
                              if f.get("dtype") == "video")
        if not self.cameras:
            raise SystemExit("info.json 的 features 里没有 dtype=video 的项, 找不到相机")

    def paths(self, idx: int) -> Tuple[str, Dict[str, str]]:
        chunk = idx // self.chunks_size
        table = self.data_tpl.format(episode_chunk=chunk, episode_index=idx)
        vids = {c: self.video_tpl.format(episode_chunk=chunk, video_key=c,
                                         episode_index=idx)
                for c in self.cameras}
        return f"{self.root}/{table}", {c: f"{self.root}/{p}" for c, p in vids.items()}

    def episode_indices(self) -> List[int]:
        """episode 下标。优先用 meta/episodes.jsonl (权威), 退回 total_episodes。"""
        eps = read_jsonl(f"{self.root}/meta/episodes.jsonl")
        if eps:
            return sorted(int(e["episode_index"]) for e in eps)
        return list(range(self.total))


# =====================================================================
# 单条
# =====================================================================
_CTX: Dict[str, Any] = {}


def _init_worker(root: str, profile: dict, params: dict, short_circuit: bool) -> None:
    _CTX.update(layout=Layout(root), profile=profile, params=params,
                short_circuit=short_circuit)


def _run_one(idx: int) -> dict:
    lay: Layout = _CTX["layout"]
    table_path, video_paths = lay.paths(idx)
    episode_id = f"episode_{idx:06d}"

    videos = {c: read_bytes(p) for c, p in video_paths.items()}
    assets = EpisodeAssets(
        data_format="lerobot", table_path=table_path, video_paths=video_paths,
        meta_paths={"episodes": f"{lay.root}/meta/episodes.jsonl",
                    "episodes_stats": f"{lay.root}/meta/episodes_stats.jsonl"})

    result = run_episode(
        videos=videos, parquet_bytes=read_bytes(table_path), episode_id=episode_id,
        profile=_CTX["profile"], assets=assets, params=_CTX["params"],
        short_circuit=_CTX["short_circuit"])
    return result.to_dict()


# =====================================================================
# 批量
# =====================================================================
def iter_results(root: str, profile: dict, params: dict, indices: List[int],
                 workers: int, short_circuit: bool) -> Iterator[dict]:
    if workers <= 1:
        _init_worker(root, profile, params, short_circuit)
        for i in indices:
            yield _run_one(i)
        return
    # 并行的粒度是 episode 级 —— 检测内部是向量运算, 在那一层开进程只会亏
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(root, profile, params, short_circuit)) as ex:
        yield from ex.map(_run_one, indices, chunksize=1)


class _Budget(DropBudget):
    """DropBudget 吃的是 EpisodeResult; 批量跑时手上只有 dict, 这里做个薄适配。"""

    def add_dict(self, d: dict) -> None:
        from .pipeline import CheckOutcome, EpisodeResult

        r = EpisodeResult(episode_id=d["episode_id"], failed=d["failed"],
                          failed_stage=d["failed_stage"], n_frames=d["n_frames"])
        for name, c in d["checks"].items():
            if c["status"] == "ok":
                r.outcomes[name] = CheckOutcome(
                    check=name, stage=c["stage"], status="ok",
                    report=_ReportView(c["report"]))
            else:
                r.outcomes[name] = CheckOutcome(
                    check=name, stage=c["stage"], status=c["status"],
                    error_type=c.get("error_type"), error_message=c.get("error_message"),
                    skip_reason=c.get("skip_reason"))
        fd = d.get("frame_deletion")
        if fd:
            r.manifest = _ManifestView(fd)
        self.add(r)


class _ReportView:
    def __init__(self, d: dict):
        self._d = d
        self.drop_episode = bool(d.get("drop_episode"))
        self.triggered_rules = list(d.get("triggered_rules", []))
        self.findings = []

    def to_dict(self) -> dict:
        return self._d


class _ManifestView:
    def __init__(self, d: dict):
        self.actionable = bool(d["actionable"])
        self.n_frames_deleted = int(d["n_frames_deleted"])
        self.n_segments = int(d["n_segments"])


def main() -> int:
    ap = argparse.ArgumentParser(description="在 LeRobot 数据集上跑七阶段质检 (只记录, 不删)")
    ap.add_argument("--root", required=True, help="数据集根目录 (含 meta/ data/ videos/)")
    ap.add_argument("--profile", required=True, help="dataset_profile.json")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条 (0=全部)")
    ap.add_argument("--workers", type=int, default=1, help="episode 级并行度")
    ap.add_argument("--short-circuit", action="store_true",
                    help="判定整条删除后跳过剩余检测。**调阈值阶段不要开** —— "
                         "会让交叠统计做不了, 见 pipeline.md §3")
    a = ap.parse_args()

    profile = read_json(a.profile) if not a.profile.startswith("{") else json.loads(a.profile)
    params = profile.pop("_params", {})

    lay = Layout(a.root)
    indices = lay.episode_indices()
    if a.limit:
        indices = indices[:a.limit]

    os.makedirs(a.out, exist_ok=True)
    print(f"数据集 {a.root}")
    print(f"  {lay.total} 条 episode, 本次跑 {len(indices)} 条 | fps={lay.fps} "
          f"| chunks_size={lay.chunks_size}")
    print(f"  相机 {lay.cameras}")
    print(f"  短路 {'开 (统计不可用)' if a.short_circuit else '关 (保留交叠统计)'}")

    budget, t0 = _Budget(), time.time()
    for n, d in enumerate(iter_results(a.root, profile, params, indices,
                                       a.workers, a.short_circuit), 1):
        with open(os.path.join(a.out, f"{d['episode_id']}.json"), "w") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        budget.add_dict(d)
        if n % 50 == 0 or n == len(indices):
            print(f"  ... {n}/{len(indices)}  ({time.time() - t0:.0f}s)")

    summary = budget.to_dict()
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    report = budget.format_report()
    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write(report + "\n")

    print("\n" + report)
    print(f"\n结果写入 {a.out}/ (每条一个 json + summary.json + report.txt)")
    print("本流水线不删任何数据; 删帧工单在各 result.json 的 frame_deletion 里")
    return 0


if __name__ == "__main__":
    sys.exit(main())
