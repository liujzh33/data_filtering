"""扫一个数据集, 逐维算出 profile 需要的统计量。

    python -m check_pipeline.tools.scan_percentiles \
        --root /path/to/dataset --columns action observation.state \
        --out  scan.json

产出的是 profile 里**必须由数据算出来、模型给不了**的那几个数:

    q1 / q99     state_extreme 的 band 基准 (全数据集该机型的 1/99 分位)
    range        sudden_change 的量程基准 (p99 - p1)
    min/max/median/std   给人看的交叉验证 —— 手爪是不是窄正区间、维度切分对不对

这一步不能交给模型: 它要读几百上千个文件逐帧池化。模型能做的是**解释**这些数字
(哪一维是手爪、哪一维是关节), 见 prompts/stage0_survey.md。

--------------------------------------------------------------------------
为什么是数据集级而不是 episode 级
--------------------------------------------------------------------------

sudden_change 的阈值是"跳变超过量程的百分之几"。量程若按每条 episode 各算各的,
同一个关节在动作幅度小的 episode 里阈值就小、大的就大 —— 同样大小的异常在不同
episode 里结论相反。state_extreme 同理。所以必须池化全数据集。

**一个数据集有多个机型时要分开扫** (--root 指向该机型的子集), 否则不同机型的量程
会被混在一起, 算出一个谁都不适用的范围。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Dict, List

from ..run_dataset import Layout, read_bytes

# 每个维度最多池化这么多帧。超过就按 episode 内等间隔抽样。
#   q1/q99 是很稳的统计量: 500 万样本下, 1 分位的估计误差远小于维度本身的量程,
#   而全量池化在大数据集上会吃掉几十 GB 内存。
MAX_FRAMES_PER_DIM = 5_000_000


def scan(root: str, columns: List[str], limit: int = 0) -> dict:
    import numpy as np
    import pandas as pd

    lay = Layout(root)
    indices = lay.episode_indices()
    if limit:
        indices = indices[:limit]

    pools: Dict[str, List] = {c: [] for c in columns}
    n_frames = 0
    n_read = 0
    keep = max(1, (len(indices) * 500) // MAX_FRAMES_PER_DIM)   # 抽样步长的粗估

    for k, idx in enumerate(indices, 1):
        table_path, _ = lay.paths(idx)
        b = read_bytes(table_path)
        if b is None:
            print(f"  ! 读不到 {table_path}, 跳过")
            continue
        df = pd.read_parquet(io.BytesIO(b))
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise SystemExit(
                f"{table_path} 缺少列 {missing} (实际: {list(df.columns)})。"
                f"列名必须和 profile 一致, 这里不做猜测")
        for c in columns:
            s = df[c]
            m = np.asarray(np.stack(s.to_numpy()) if s.dtype == object else s.to_numpy(),
                           dtype=np.float64)
            if m.ndim == 1:
                m = m.reshape(-1, 1)
            pools[c].append(m[::keep] if keep > 1 else m)
        n_frames += len(df)
        n_read += 1
        if k % 100 == 0:
            print(f"  ... {k}/{len(indices)}")

    if not n_read:
        raise SystemExit("一条都没读到")

    out = {"root": root, "n_episodes_scanned": n_read, "n_frames_total": n_frames,
           "sampling_step": keep, "columns": {}}
    for c in columns:
        m = np.concatenate(pools[c], axis=0)
        if not np.all(np.isfinite(m)):
            n_bad = int((~np.isfinite(m)).sum())
            raise SystemExit(
                f"列 {c!r} 有 {n_bad} 个 NaN/Inf。分位数会被污染成 NaN 且不报错, "
                f"必须由上游先处理掉")
        dims = []
        for d in range(m.shape[1]):
            x = m[:, d]
            q1, q99 = (float(v) for v in np.percentile(x, [1, 99]))
            dims.append({
                "dim": d,
                "q1": q1, "q99": q99,
                "range": q99 - q1,          # sudden_change 用
                "min": float(x.min()), "max": float(x.max()),
                "median": float(np.median(x)), "std": float(x.std()),
                # 交叉验证用: 恒在 [0,1] 或 [0,0.1] 这种窄正区间的多半是手爪,
                # 弧度量级跨正负的是关节。见 prompts/stage0_survey.md
                "looks_like_gripper": bool(x.min() >= -1e-6 and x.max() <= 1.01),
            })
        out["columns"][c] = {"n_dims": int(m.shape[1]), "n_samples": int(m.shape[0]),
                             "dims": dims}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--columns", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只扫前 N 条 (0=全部)")
    a = ap.parse_args()

    print(f"扫描 {a.root}  列={a.columns}")
    res = scan(a.root, a.columns, a.limit)
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n{res['n_episodes_scanned']} 条 episode, {res['n_frames_total']} 帧")
    for c, info in res["columns"].items():
        print(f"\n=== {c}  ({info['n_dims']} 维) ===")
        print(f"{'dim':>4} {'q1':>10} {'q99':>10} {'range':>9} "
              f"{'min':>10} {'max':>10} {'median':>10}   备注")
        for d in info["dims"]:
            note = "像手爪 (窄正区间)" if d["looks_like_gripper"] else ""
            print(f"{d['dim']:>4} {d['q1']:>10.4f} {d['q99']:>10.4f} {d['range']:>9.4f} "
                  f"{d['min']:>10.4f} {d['max']:>10.4f} {d['median']:>10.4f}   {note}")
    print(f"\n写入 {a.out}")
    print("下一步: 把这些数字连同 prompts/stage0_survey.md 一起发给模型, "
          "让它判断每一维是什么, 产出 dataset_profile.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
