"""把 embodiment_percentiles_ext.json 投影成 state_extreme 的 profile 片段。

用法:
    python3 -m check_pipeline.tools.build_state_extreme_profile \
        --input  embodiment_percentiles_ext.json \
        --output state_extreme_profiles.json

--------------------------------------------------------------------------
为什么是"投影"而不是"重算"
--------------------------------------------------------------------------

原打算重扫数据集、从 action 列重算 q1/q99, 查下来这条路走不通也没必要:

  * **robochallenge1/2 (8 个机型) 没有 action 通道。** states/states.jsonl 只有 state,
    见 state_action_align/robochallenge.py —— 该数据集的 SAA 检测直接 raise。所以这 8 个
    机型的 q1/q99 本来就是从 state 算的 (维度名 joint_positions_* / gripper_width 也是
    state 字段名), 重算无从谈起。state_extreme.md §1.8 说的"用 action 分位数卡 state 值",
    真相是这份文件本身就是混着的, 只是没人写下来。这次按数据集把 source 标清楚。

  * **robottwin2 (5 个机型) 重算是空转。** 维度名 joint_action_* 直接对应 HDF5 的
    joint_action/ 组, 本来就是 action; 且 q1/q99 在原文件里是**原样存着的**, lo/hi 只是
    α=3 的派生值 (13 个机型实测全部精确等于 3.00)。重跑会得到同样的数字。

所以本脚本做的是纯粹的投影 + 修正: 丢掉 lo/hi (烘死 α 的元凶), 按数据集标 source,
剔掉夹爪维度, 修 franka 的维度错位。**不产生任何新的统计量。**

--------------------------------------------------------------------------
franka 的维度错位
--------------------------------------------------------------------------

robottwin2/franka 有 16 维, 但名字套的是 aloha-agilex 的 6+6+2 模板:

     11      joint_action_right_arm_5   [-2.333, 2.478]
     12 GRIP joint_action_left_gripper  [ 1.840, 3.584]   <- 弧度, 这是关节
     13 GRIP joint_action_right_gripper [-2.053, 2.706]   <- 弧度, 这是关节
     14      dim_14                     [ 0.000, 1.000]   <- 这才是夹爪
     15      dim_15                     [ 0.000, 1.000]   <- 这才是夹爪

后果正好反着: 两个真夹爪 (双峰分布, 明确要求排除的) 进了检测, 两个真关节被豁免。
franka 是 7 自由度, 真实布局应是 7+7+2, 按此重映射。

**这个映射未经真实数据验证** (本机没有 robottwin2 副本, moxing 也没装), 所以输出里
带 needs_verification 标记。等能读到真数据时, 拿一条 episode 的 joint_action 维度对一遍。
"""

import argparse
import json
import os
import sys

# 各数据集的信号来源与逻辑列名。
#   source: 这份 q1/q99 是从哪种信号算出来的 —— state_extreme 强制要求声明, 杜绝错配。
#   column: check_state_extreme 收到的 signals 字典的键。**格式适配器必须用这个键**
#           把矩阵喂进来。三个数据集的物理格式各不相同 (robochallenge 是 JSONL,
#           robottwin2 是 HDF5, 都不是 parquet), 所以这里是逻辑名而非物理列名。
DATASET_SOURCES = {
    "robochallenge1": {"source": "state", "column": "state"},
    "robochallenge2": {"source": "state", "column": "state"},
    "robottwin2": {"source": "action", "column": "joint_action"},
}

# franka 的正确布局: 7 自由度双臂 + 2 夹爪。
FRANKA_REMAP = (
    [(i, f"joint_action_left_arm_{i}", False) for i in range(7)]
    + [(7 + i, f"joint_action_right_arm_{i}", False) for i in range(7)]
    + [(14, "joint_action_left_gripper", True), (15, "joint_action_right_gripper", True)]
)

# 某维度的 span 低于本机型关节 span 中位数的这个比例时, 提示复核。
#   不用绝对弧度阈值 —— 不同机型的量程差一个数量级 (UR5 的夹爪行程是 3~228 的整数)。
#   相对量是自校准的。
NARROW_SPAN_FRAC = 0.10


def build(raw: dict, verbose: bool = True) -> tuple:
    """返回 (profiles, warnings)。profiles: {scope: state_extreme profile}"""
    profiles, warnings = {}, []

    for ds in sorted(raw):
        meta = DATASET_SOURCES.get(ds)
        if meta is None:
            raise SystemExit(
                f"数据集 {ds!r} 没有 source/column 声明。必须显式补进 DATASET_SOURCES —— "
                f"猜 source 正是 state_extreme.md §1.8 那个错配的来源")

        for emb in sorted(raw[ds]):
            scope = f"{ds}/{emb}"
            entries = raw[ds][emb]
            notes = []

            if scope == "robottwin2/franka":
                remap = {i: (nm, g) for i, nm, g in FRANKA_REMAP}
                if len(entries) != len(remap):
                    raise SystemExit(
                        f"{scope}: 原文件 {len(entries)} 维, 重映射表 {len(remap)} 维, 对不上")
                notes.append(
                    "维度名与 is_gripper 已按 7+7+2 重映射: 原文件套用了 aloha-agilex 的 "
                    "6+6+2 模板, 把 dim 12/13 (弧度, 真关节) 标成夹爪, 而真夹爪 dim 14/15 "
                    "([0,1]) 没被标出。**此映射未经真实数据验证**")
            else:
                remap = None

            dims, spans, dropped = [], [], []
            for key in sorted(entries, key=int):
                idx = int(key)
                e = entries[key]
                name, is_gripper = (remap[idx] if remap
                                    else (e["dim_name"], bool(e["is_gripper"])))
                q1, q99 = float(e["q1"]), float(e["q99"])

                if is_gripper:
                    # 夹爪是双峰分布 (开/合两个模态), 1-99 分位对它没有"正常范围"的含义。
                    # 这里是**不进列表**, 不是运行时豁免 —— 旧版靠子串匹配, 'effector' 是
                    # 'end_effector' 的子串, 会把末端位姿整体误伤。
                    dropped.append(f"{idx}:{name}")
                    continue
                if not (q99 > q1):
                    raise SystemExit(
                        f"{scope} dim {idx} ({name}): q99={q99} 不大于 q1={q1}, "
                        f"band 宽度非正。该维度在全数据集上没动过, 不该参与检测")

                dims.append({"name": name, "column": meta["column"], "dim": idx,
                             "q1": q1, "q99": q99})
                spans.append(q99 - q1)

            if not dims:
                raise SystemExit(f"{scope}: 剔掉夹爪后没有维度可检")

            med = sorted(spans)[len(spans) // 2]
            for d, sp in zip(dims, spans):
                if sp < NARROW_SPAN_FRAC * med:
                    w = (f"{scope} dim {d['dim']} ({d['name']}): span={sp:.4f}, "
                         f"仅为本机型关节 span 中位数 ({med:.4f}) 的 {sp / med:.1%}。"
                         f"该关节在全数据集上几乎没动, α=1 给它的 band 只有 ±{sp:.4f}, "
                         f"会异常敏感 —— 跑之前确认这个关节是真的被锁住了")
                    warnings.append(w)

            prof = {"enabled": True, "alpha": 1.0, "source": meta["source"],
                    "percentile_scope": scope, "dims": dims}
            if remap:
                prof["needs_verification"] = True
            if notes:
                prof["notes"] = notes
            if dropped:
                prof["excluded_gripper_dims"] = dropped
            profiles[scope] = prof

            if verbose:
                print(f"  {scope:28s} {len(dims):2d} 维参检  剔除夹爪 {len(dropped)}  "
                      f"source={meta['source']}"
                      + ("  [franka 重映射, 待核实]" if remap else ""))
    return profiles, warnings


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=os.path.join(here, "embodiment_percentiles_ext.json"))
    ap.add_argument("--output", default=os.path.join(here, "state_extreme_profiles.json"))
    a = ap.parse_args()

    with open(a.input) as f:
        raw = json.load(f)

    print(f"读入 {a.input}")
    profiles, warnings = build(raw)

    if warnings:
        print(f"\n需要复核的维度 ({len(warnings)} 处):")
        for w in warnings:
            print(f"  ! {w}")

    with open(a.output, "w") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
        f.write("\n")

    n_dims = sum(len(p["dims"]) for p in profiles.values())
    print(f"\n写出 {a.output}")
    print(f"  {len(profiles)} 个机型, 共 {n_dims} 个参检维度")
    print(f"  lo/hi 已丢弃 —— band 由 α 在运行时算, α 才是活参数")
    return 0


if __name__ == "__main__":
    sys.exit(main())
