"""流水线编排 + 删帧清单的回归测试。

覆盖的是**编排层**的职责, 各检测自身的判据在各自的 test_* 里:
  - 七个检测按阶段跑, 每条命中都能追溯到"哪个阶段 / 哪个检测 / 哪条规则"
  - 报错分三档各归各位 (整条失败 / 单检测报错 / 主动跳过), 且原因原文进 JSON
  - 删帧清单列全同一 episode 的所有目标 (多路视频 + 动作表)
  - 清单在 corrupt 未过 / 整条已删时标为不可执行

probe_video 需要 pyav, 这里直接构造 VideoProbe 顶替。需要真 numpy + pandas + pyarrow,
缺任何一个就干净跳过。

用法: python3 -m check_pipeline.tests.test_pipeline
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import numpy as np
    import pandas as pd
except Exception as e:                                    # pragma: no cover
    print(f"跳过: 需要真 numpy + pandas ({e})")
    sys.exit(0)

from check_pipeline import pipeline as pl                  # noqa: E402
from check_pipeline.errors import UnsupportedDataError     # noqa: E402
from check_pipeline.manifest import (                      # noqa: E402
    EpisodeAssets, build_manifest)
from check_pipeline.probe import VideoProbe                # noqa: E402

PASS, FAIL = [], []
CAMS = ("head", "left_wrist")
T, FPS, NDIM = 300, 30.0, 6


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  <- ' + extra) if extra and not cond else ''}")


def raises(name, fn, frag="", exc=UnsupportedDataError):
    try:
        fn()
    except exc as e:
        check(name, frag in str(e), f"got: {e}")
    except Exception as e:
        check(name, False, f"wrong type {type(e).__name__}: {e}")
    else:
        check(name, False, "no exception raised")


# =====================================================================
# 构造素材
# =====================================================================
def traj(spike_at=None):
    """平滑的多维轨迹, 幅度在 [-1, 1] 内。spike_at 处注入一个大毛刺。"""
    t = np.arange(T) / FPS
    x = np.stack([0.8 * np.sin(2 * np.pi * (0.2 + 0.05 * i) * t) for i in range(NDIM)], 1)
    if spike_at is not None:
        x[spike_at, 0] += 8.0
    return x


def parquet_of(x, action=None):
    """默认 state 与 action 相同 -> SAA 的 DA 接近 1, 不会误触发。"""
    a = x if action is None else action
    df = pd.DataFrame({
        "observation.state": [r.tolist() for r in x],
        "action": [r.tolist() for r in a],
        "frame_index": np.arange(T),
        "timestamp": np.arange(T) / FPS,
    })
    b = io.BytesIO()
    df.to_parquet(b, index=False)
    return b.getvalue()


def probes(n=T, gray_mean=120.0, gray_std=40.0, fps=FPS, cams=CAMS):
    return {c: VideoProbe(cam=c, fps=fps, n_declared=n, n_readable=n, n_frames=n,
                          gray_mean=[gray_mean] * n, gray_std=[gray_std] * n,
                          frame_diff=[1e9] * n)
            for c in cams}


def profile(**over):
    p = {
        "saa": {"enabled": True, "action_space": "absolute",
                "state_column": "observation.state", "action_column": "action",
                "da_threshold": 0.7,
                "shared_dims": [{"name": f"j{i}", "state_dim": i, "action_dim": i}
                                for i in range(NDIM)]},
        "sudden_change": {"enabled": True,
                          "dims": [{"name": f"j{i}", "column": "action", "dim": i,
                                    "range": 1.6} for i in range(NDIM)]},
        "state_extreme": {"enabled": True, "source": "action",
                          "percentile_scope": "test/robot",
                          "dims": [{"name": f"j{i}", "column": "action", "dim": i,
                                    "q1": -1.0, "q99": 1.0} for i in range(NDIM)]},
    }
    p.update(over)
    return p


ASSETS = EpisodeAssets(
    data_format="lerobot",
    table_path="data/chunk-000/episode_000123.parquet",
    video_paths={c: f"videos/chunk-000/{c}/episode_000123.mp4" for c in CAMS},
    meta_paths={"episodes": "meta/episodes.jsonl"})


_REAL_PROBE = pl.probe_video


def run(x=None, pr=None, prof=None, assets=ASSETS, action=None, **kw):
    """跑一条 episode, probe 用构造的顶替。"""
    x = traj() if x is None else x
    pr = probes() if pr is None else pr
    pl.probe_video = lambda b, cam, **_: pr[cam]
    try:
        return pl.run_episode(
            videos={c: b"fake" for c in pr}, parquet_bytes=parquet_of(x, action),
            episode_id="ep_000123", profile=profile() if prof is None else prof,
            assets=assets, params={"stillness": {"state_diff_thresh": 1e-3}}, **kw)
    finally:
        pl.probe_video = _REAL_PROBE


# =====================================================================
print("\n[编排] 七个检测都接上了, 且带阶段号")
# =====================================================================
check("ALL_CHECKS 是 7 个", len(pl.ALL_CHECKS) == 7, str(pl.ALL_CHECKS))
check("七个检测齐全", set(pl.ALL_CHECKS) == {
    "corrupt", "black", "occlusion", "stillness",
    "saa", "sudden_change", "state_extreme"}, str(pl.ALL_CHECKS))
check("corrupt 排第一 (下标映射的前提由它确立)", pl.ALL_CHECKS[0] == "corrupt")
check("saa 排在两个帧级检测之前 (episode 级先判)",
      pl.ALL_CHECKS.index("saa") < pl.ALL_CHECKS.index("sudden_change"))
check("阶段划分: 四个看画面的是阶段 1",
      all(pl.CHECK_STAGE[c] == 1
          for c in ("corrupt", "black", "occlusion", "stillness")))
check("阶段划分: 三个看信号的是阶段 2",
      all(pl.CHECK_STAGE[c] == 2
          for c in ("saa", "sudden_change", "state_extreme")))

r = run()
check("干净数据: 七个检测全部跑出结论",
      sorted(c for c, o in r.outcomes.items() if o.status == "ok") ==
      sorted(pl.ALL_CHECKS),
      str({c: o.status for c, o in r.outcomes.items()}))
check("干净数据: 不删 episode", not r.drop_episode, str(r.dropped_by))
check("干净数据: 没有命中任何规则", r.triggered == [], str(r.triggered))
check("干净数据: 没有要删的帧", r.manifest.n_frames_deleted == 0)
check("记录了 fps 与帧数", r.fps == FPS and r.n_frames == T)


# =====================================================================
print("\n[JSON] 每条命中都能追溯到 阶段/检测/规则")
# =====================================================================
rb = run(pr=probes(gray_mean=2.0, gray_std=0.5))     # 全黑
d = rb.to_dict()
check("能 json 序列化", isinstance(json.dumps(d, ensure_ascii=False), str))
check("triggered 是扁平清单", isinstance(d["triggered"], list) and d["triggered"])
t0 = d["triggered"][0]
check("每条命中带 stage", t0["stage"] == 1, str(t0))
check("每条命中带 check", t0["check"] == "black", str(t0))
check("每条命中带 rule", t0["rule"] == "BLACK_RUN_RATIO_EXCEEDED", str(t0))
check("每条命中带 disposition", t0["disposition"] == "drop_episode", str(t0))
check("每条命中带规则中文说明", bool(t0["description"]))
check("每条命中带涉及的视角", set(t0["views"]) == set(CAMS), str(t0["views"]))
check("dropped_by 指出是谁删的", d["dropped_by"] == ["black"], str(d["dropped_by"]))
check("triggered_rules 按检测归组",
      d["triggered_rules"]["black"] == ["BLACK_RUN_RATIO_EXCEEDED"])
check("各检测的完整 report 仍在 checks 里", "measurements" in d["checks"]["black"]["report"])
check("checks 里每项都带 stage", all("stage" in v for v in d["checks"].values()))

# saa 是纯 episode 级的, 它的 Finding 没有 frame_ranges 字段。triggered 必须容得下
# 这种结构差异 —— 这里曾经直接 f.frame_ranges 取值, 在真实数据上崩过。
x_st = traj()
x_ac = x_st.copy()
x_ac[:, :3] = np.roll(x_ac[:, :3], 45, axis=0)   # 只把 action 的前三维错开 -> DA 崩掉
rsaa = run(x_st, action=x_ac)
check("SAA 命中时 triggered 不崩 (它没有 frame_ranges 字段)",
      isinstance(rsaa.to_dict()["triggered"], list))
saa_hits = [t for t in rsaa.to_dict()["triggered"] if t["check"] == "saa"]
check("SAA 确实命中了", bool(saa_hits), str(rsaa.outcomes["saa"].status))
if saa_hits:
    check("SAA 命中项的 frame_ranges 补成空表", saa_hits[0]["frame_ranges"] == [])
    check("SAA 命中项带 dims (哪些维度不对齐)", bool(saa_hits[0]["dims"]))
    check("SAA 命中项的 views 补成空表", saa_hits[0]["views"] == [])


# =====================================================================
print("\n[删帧清单] 列全同一 episode 的所有目标")
# =====================================================================
rs = run(traj(spike_at=150))
m = rs.manifest
check("突变检测产出了删帧区间", m.n_frames_deleted > 0, str(m.ranges))
check("清单可执行", m.actionable, str(m.blocked_reason))
check("区间左闭右开且落在毛刺附近",
      all(140 <= a and b <= 165 for a, b in m.ranges), str(m.ranges))
check("记录了每个区间是哪个检测要求删的", "sudden_change" in m.sources, str(m.sources))

md = m.to_dict()
kinds = [t["kind"] for t in md["targets"]]
check("目标含全部视频路径", kinds.count("video") == len(CAMS), str(kinds))
check("目标含动作数据表", "table" in kinds, str(kinds))
check("目标含需连带更新的元数据", "meta" in kinds, str(kinds))
vids = {t["camera"] for t in md["targets"] if t["kind"] == "video"}
check("视频目标覆盖每一路相机", vids == set(CAMS), str(vids))
check("每个视频目标都带同一组帧区间",
      all(t["frame_ranges"] == md["frame_ranges"]
          for t in md["targets"] if t["kind"] == "video"))
check("表目标的帧区间与视频一致 (同步删除)",
      [t for t in md["targets"] if t["kind"] == "table"][0]["frame_ranges"]
      == md["frame_ranges"])
check("记录了数据格式", md["data_format"] == "lerobot")
check("表目标说明了该格式的连带后果",
      "全局 index" in [t for t in md["targets"] if t["kind"] == "table"][0]["action"])
check("报出删完会留下几处不连续", md["n_segments"] == len(md["frame_ranges"]))
check("报出删帧占比", 0 < md["delete_ratio"] < 1)


# =====================================================================
print("\n[删帧清单] 什么时候不可执行")
# =====================================================================
check("整条已被删 -> 清单不可执行", not rb.manifest.actionable)
check("并说明原因是整条已删", "判整条删除" in (rb.manifest.blocked_reason or ""),
      str(rb.manifest.blocked_reason))

# 各路视频帧数不一致 -> corrupt 命中对齐类规则 -> 下标映射不成立
pr_mis = probes()
pr_mis["left_wrist"] = VideoProbe(cam="left_wrist", fps=FPS, n_declared=T - 5,
                                  n_readable=T - 5, n_frames=T - 5,
                                  gray_mean=[120.0] * (T - 5), gray_std=[40.0] * (T - 5),
                                  frame_diff=[1e9] * (T - 5))
rmis = run(traj(spike_at=150), pr=pr_mis)
check("帧数不一致时 corrupt 命中对齐类规则",
      any(r in pl.ALIGNMENT_RULES for r in rmis.reports["corrupt"].triggered_rules),
      str(rmis.reports["corrupt"].triggered_rules))

# 未提供 assets 时清单列不全目标
rna = run(traj(spike_at=150), assets=None)
check("没给 assets -> 不可执行", not rna.manifest.actionable)
check("并说明是列不全要删的文件", "列不全" in (rna.manifest.blocked_reason or ""),
      str(rna.manifest.blocked_reason))
check("但区间照样记下来供诊断", rna.manifest.n_frames_deleted > 0)


# =====================================================================
print("\n[报错] 三档各归各位, 原因原文进 JSON")
# =====================================================================
p_no_se = profile()
del p_no_se["state_extreme"]
rerr = run(prof=p_no_se)
check("缺 profile 段 -> 该检测记为 error",
      rerr.outcomes["state_extreme"].status == "error")
check("错误类型写明", rerr.errors["state_extreme"]["type"] == "UnsupportedDataError")
check("错误原文写明缺的是哪一段",
      "'state_extreme' 段" in rerr.errors["state_extreme"]["message"],
      rerr.errors["state_extreme"]["message"])
check("错误带阶段号", rerr.errors["state_extreme"]["stage"] == 2)
check("单个检测报错不影响其他六个",
      sum(1 for o in rerr.outcomes.values() if o.status == "ok") == 6,
      str({c: o.status for c, o in rerr.outcomes.items()}))
check("报错的检测不产生结论 (不算通过)", "state_extreme" not in rerr.reports)

p_dis = profile()
p_dis["saa"] = {"enabled": False, "reason": "该数据集无 action 通道"}
rdis = run(prof=p_dis)
check("profile 标 enabled=false -> 记为 skipped (主动关掉, 不是出错)",
      rdis.outcomes["saa"].status == "skipped", rdis.outcomes["saa"].status)
check("并带上 profile 写的理由",
      "无 action 通道" in rdis.skipped["saa"], rdis.skipped["saa"])
check("主动跳过不算通过 (不产生结论)", "saa" not in rdis.reports)
check("主动跳过不进 errors", "saa" not in rdis.errors)

pbad = profile()
pbad["state_extreme"] = dict(pbad["state_extreme"], source="joint")
rbad = run(prof=pbad)
check("source 非法 -> 报错说明是 source 的问题, 不是笼统的'数据错误'",
      "source 必须是" in rbad.errors["state_extreme"]["message"],
      rbad.errors["state_extreme"]["message"])

pr_fps = probes()
pr_fps["left_wrist"] = VideoProbe(cam="left_wrist", fps=15.0, n_declared=T, n_readable=T,
                                  n_frames=T, gray_mean=[120.0] * T, gray_std=[40.0] * T,
                                  frame_diff=[1e9] * T)
rfps = run(pr=pr_fps)
check("各路 fps 不一致 -> 按秒定义的检测报错",
      rfps.outcomes["saa"].status == "error" and
      rfps.outcomes["sudden_change"].status == "error")
check("fps 报错说明了具体是什么问题",
      "fps 不一致" in rfps.errors["saa"]["message"], rfps.errors["saa"]["message"])
check("不按秒定义的 state_extreme 不受 fps 影响",
      rfps.outcomes["state_extreme"].status == "ok")

rno = pl.run_episode(videos={}, parquet_bytes=b"", episode_id="e")
check("没有视频 -> 整条失败 (无结论)", rno.failed and rno.failed_stage == "probe")
check("整条失败带错误类型", rno.failed_error_type == "UnsupportedDataError")


# =====================================================================
print("\n[短路] 默认跑满, 打开后跳过")
# =====================================================================
check("默认不短路: 黑屏已判删, 阶段 2 照样跑完",
      all(rb.outcomes[c].status == "ok" for c in ("saa", "sudden_change", "state_extreme")),
      str({c: rb.outcomes[c].status for c in pl.ALL_CHECKS}))
rsc = run(pr=probes(gray_mean=2.0, gray_std=0.5), short_circuit=True)
check("打开短路: 后续检测记为 skipped",
      rsc.outcomes["saa"].status == "skipped", rsc.outcomes["saa"].status)
check("skipped 与 error 分开记账", "saa" in rsc.skipped and "saa" not in rsc.errors)
check("并写明是被谁短路的",
      "black" in rsc.skipped["saa"], rsc.skipped["saa"])
check("短路不改变 episode 结论", rsc.drop_episode and rsc.dropped_by == ["black"])


# =====================================================================
print("\n[manifest] 单元行为")
# =====================================================================
mm = build_manifest("e", {"a": [(1, 3)], "b": [(2, 5)]}, 100, CAMS, ASSETS)
check("多来源取并集", mm.ranges == [(1, 5)], str(mm.ranges))
check("但保留各自来源", mm.sources == {"a": [(1, 3)], "b": [(2, 5)]}, str(mm.sources))
check("单帧记为 [5,6)",
      build_manifest("e", {"a": [(5, 6)]}, 100, CAMS, ASSETS).ranges == [(5, 6)])
raises("区间越界 -> 报错, 不容忍",
       lambda: build_manifest("e", {"a": [(0, 200)]}, 100, CAMS, ASSETS), "超出 episode 范围")
raises("相机集合对不上 -> 报错 (少删一路就是制造错位)",
       lambda: build_manifest("e", {"a": [(1, 2)]}, 100, ("head", "extra"), ASSETS),
       "缺的那一路不会被删帧")
raises("数据格式未知 -> 报错说明是格式问题",
       lambda: build_manifest("e", {"a": [(1, 2)]}, 100, CAMS,
                              EpisodeAssets(data_format="rosbag", table_path="t",
                                            video_paths={c: c for c in CAMS})),
       "数据格式 'rosbag' 未知")
raises("缺动作表路径 -> 报错",
       lambda: build_manifest("e", {"a": [(1, 2)]}, 100, CAMS,
                              EpisodeAssets(data_format="hdf5", table_path="",
                                            video_paths={c: c for c in CAMS})),
       "会让这条 episode 从该帧起整体错位")
check("hdf5 目标说明了不能原地删行",
      "不能原地删行" in [t for t in build_manifest(
          "e", {"a": [(1, 2)]}, 100, CAMS,
          EpisodeAssets(data_format="hdf5", table_path="t.h5",
                        video_paths={c: c for c in CAMS})).to_dict()["targets"]
          if t["kind"] == "table"][0]["action"])


# =====================================================================
print("\n[预算] 汇总统计")
# =====================================================================
b = pl.DropBudget()
for x in (run(), rb, rs, rerr, rno):
    b.add(x)
bd = b.to_dict()
check("失败的不计入分母", bd["n_failed"] == 1 and bd["n_judged"] == 4)
check("统计了整条删除数", bd["n_dropped"] == 1)
check("统计了帧级删除", bd["frame_deletion"]["n_episodes_edited"] == 1)
check("统计了不连续处数分布", bool(bd["frame_deletion"]["segment_hist"]))
check("统计了各检测报错次数",
      any("state_extreme" in k for k in bd["error_counts"]), str(bd["error_counts"]))
check("统计了规则命中次数",
      "black.BLACK_RUN_RATIO_EXCEEDED" in bd["rule_counts"], str(bd["rule_counts"]))
check("format_report 能出文本", "整条删除" in b.format_report())
check("预算可 json 序列化", isinstance(json.dumps(bd, ensure_ascii=False), str))


print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for n in FAIL:
        print("  FAILED:", n)
    sys.exit(1)
