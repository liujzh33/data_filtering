"""四个检测的规则回归测试。

numpy 有真的就用真的, 没有才退回 _npstub 替身 (见该模块说明)。
probe_video 始终不跑 (需要 pyav), 改为直接构造 VideoProbe 喂给各 check_*。

用法: python3 -m check_pipeline.tests.test_rules
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from check_pipeline.tests import _npstub  # noqa: E402
print(f"numpy: {'替身 (_npstub)' if _npstub.install_if_missing() else '真库'}")

from check_pipeline.errors import UnsupportedDataError  # noqa: E402
from check_pipeline import black, occlusion, stillness  # noqa: E402
from check_pipeline.probe import VideoProbe  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  <- ' + extra) if extra and not cond else ''}")


def raises(name, fn, frag=""):
    try:
        fn()
    except UnsupportedDataError as e:
        check(name, frag in str(e), f"got: {e}")
    except Exception as e:
        check(name, False, f"wrong type {type(e).__name__}: {e}")
    else:
        check(name, False, "no exception raised")


def mkprobe(cam, gm, gs, fd=None, fps=30.0):
    n = len(gm)
    return VideoProbe(cam=cam, fps=fps, n_declared=n, n_readable=n, n_frames=n,
                      gray_mean=gm, gray_std=gs,
                      frame_diff=fd if fd is not None else [1e9] * n)


# =====================================================================
print("\n[black] 判据: mean<18 AND std<8")
# =====================================================================
CASES = [
    # (名字, mean, std, 期望是否黑帧)
    ("真黑屏 (掉线)",          2.0,  0.5, True),
    ("limited range 黑电平",  16.0,  0.4, True),
    ("暗场景有内容",           25.0, 18.0, False),
    ("正常画面",              120.0, 45.0, False),
    ("暗且几乎平, 边界内",      17.9,  7.9, True),
    ("刚好不暗 (归 occlusion)", 18.0,  0.4, False),
    ("刚好不平",               10.0,  8.0, False),
]
for label, m, s, want in CASES:
    v = black.measure_view(mkprobe("c", [m] * 10, [s] * 10), 18.0, 8.0)
    check(f"{label}: mean={m} std={s} -> {'黑' if want else '非黑'}",
          (v.n_black == 10) == want, f"n_black={v.n_black}")

print("\n[black] 唯一规则: 任一视角**最长连续**黑帧段 >= 10% -> drop_episode")
pb = mkprobe("b", [120.0] * 100, [40.0] * 100)


def mkblack(n_black_prefix):
    """前 n 帧连续黑, 其余正常。"""
    n = n_black_prefix
    return mkprobe("a", [2.0] * n + [120.0] * (100 - n),
                   [0.4] * n + [40.0] * (100 - n))


r = black.check_black({"a": mkblack(9), "b": pb})
check("连续 9 帧 (9%) -> 不删", not r.drop_episode,
      f"run_ratio={r.views['a'].longest_black_run_ratio}")

r = black.check_black({"a": mkblack(10), "b": pb})
check("连续 10 帧 (10%) -> 删整条",
      r.drop_episode and r.triggered_rules == ["BLACK_RUN_RATIO_EXCEEDED"],
      f"{r.triggered_rules}")
check("只点名超标的那一路", r.findings[0].views == ["a"], f"{r.findings[0].views}")
check("evidence 写明判据是最长连续段",
      r.findings[0].evidence["criterion"] == "longest_black_run / n_valid")
check("evidence 给出最长段的位置",
      r.findings[0].evidence["per_view_over"]["a"]["longest_black_run_span"] == [0, 10])
check("没超标的那一路也记录了 run_ratio",
      set(r.findings[0].evidence["per_view_longest_run_ratio"]) == {"a", "b"})

print("\n[black] 连续 vs 总量: 散布的黑帧不再触发")
# 20 个孤立黑帧散布全片: 总量 20%, 但最长连续段只有 1 帧
pa = mkprobe("a", ([2.0] + [120.0] * 4) * 20, ([0.4] + [40.0] * 4) * 20)
r = black.check_black({"a": pa, "b": pb})
check("散布 20 个孤立黑帧 (总量 20%) -> 不删", not r.drop_episode)
check("总量确实是 20%", abs(r.views["a"].black_ratio - 0.20) < 1e-9,
      f"{r.views['a'].black_ratio}")
check("最长连续段只有 1 帧", r.views["a"].longest_black_run == 1)
check("总量仍然写进 measurements (只是不参与判定)",
      "black_ratio" in r.views["a"].to_dict())

# 同样 20 帧, 但连成一段 -> 触发
r = black.check_black({"a": mkblack(20), "b": pb})
check("同样 20 帧但连续 -> 删整条", r.drop_episode)

print("\n[black] 整路全黑 (旧版自参照基线的漏检场景)")
pa = mkprobe("a", [12.0] * 100, [0.3] * 100)   # 全程亮度 12, 旧版 abs=10 和相对阈值都抓不到
r = black.check_black({"a": pa, "b": pb})
check("全程亮度 12 -> 最长连续段 100% -> 删整条",
      r.drop_episode and r.views["a"].longest_black_run_ratio == 1.0)

print("\n[black] 坏帧 (NaN) 不计入分母")
nan = float("nan")
pa = mkprobe("a", [nan] * 50 + [2.0] * 6 + [120.0] * 44, [nan] * 50 + [0.4] * 6 + [40.0] * 44)
r = black.check_black({"a": pa, "b": pb})
check("连续 6 帧 / 有效 50 帧 = 12% (分母是有效帧, 不是 100)", r.drop_episode,
      f"run_ratio={r.views['a'].longest_black_run_ratio} n_valid={r.views['a'].n_valid}")
check("n_valid = 50", r.views["a"].n_valid == 50)


# =====================================================================
print("\n[occlusion] 判据: std<8 AND mean>=18, 与 black 互斥")
# =====================================================================
for label, m, s, want in [
    ("手贴住镜头 (不暗但平)",  60.0,  3.0, True),
    ("整片发白",             230.0,  2.0, True),
    ("黑屏 (归 black)",        2.0,  0.5, False),
    ("白墙有微纹理",          200.0, 12.0, False),
    ("正常画面",              120.0, 45.0, False),
]:
    v = occlusion.measure_view(mkprobe("c", [m] * 10, [s] * 10), 8.0, 18.0)
    check(f"{label}: mean={m} std={s} -> {'遮挡' if want else '非遮挡'}",
          (v.n_occluded == 10) == want, f"n_occluded={v.n_occluded}")

print("\n[occlusion] black / occlusion 无缝隙无重叠")
gaps = []
for m in [0, 5, 10, 17, 18, 25, 60, 120, 200, 250]:
    for s in [0, 1, 3, 5, 7.9, 8, 15, 40]:
        pr = mkprobe("c", [float(m)] * 4, [float(s)] * 4)
        nb = black.measure_view(pr, 18.0, 8.0).n_black
        no = occlusion.measure_view(pr, 8.0, 18.0).n_occluded
        flat = s < 8
        if flat and (nb > 0) == (no > 0):
            gaps.append((m, s, nb, no))          # 平: 必须恰好命中一个
        if not flat and (nb or no):
            gaps.append((m, s, nb, no))          # 不平: 两个都不该命中
check("80 组 (mean, std) 组合下每个'平'帧恰好归属一方", not gaps, f"异常组合 {gaps[:5]}")

print("\n[occlusion] 整路遮挡 (旧版 AND+自参照必然漏检的场景)")
pa = mkprobe("a", [60.0] * 100, [2.0] * 100)   # 全程被挡, 旧版相对阈值 = 2*0.15 = 0.3, 抓不到
pb_ok = mkprobe("b", [120.0] * 100, [40.0] * 100)
r = occlusion.check_occlusion({"a": pa, "b": pb_ok})
check("全程 std=2 -> 最长连续段 100% -> 删整条",
      r.drop_episode and r.views["a"].longest_occluded_run_ratio == 1.0)

print("\n[occlusion] 判据同样是最长连续段, 不是总量")
r = occlusion.check_occlusion({"a": mkprobe("a", [60.0] * 9 + [120.0] * 91,
                                            [2.0] * 9 + [40.0] * 91), "b": pb_ok})
check("连续 9 帧 (9%) -> 不删", not r.drop_episode)

r = occlusion.check_occlusion({"a": mkprobe("a", [60.0] * 10 + [120.0] * 90,
                                            [2.0] * 10 + [40.0] * 90), "b": pb_ok})
check("连续 10 帧 (10%) -> 删整条",
      r.drop_episode and r.triggered_rules == ["OCCLUSION_RUN_RATIO_EXCEEDED"],
      f"{r.triggered_rules}")

# 散布 20 个孤立低纹理帧: 总量 20%, 最长连续段 1 帧
pa_scat = mkprobe("a", ([60.0] + [120.0] * 4) * 20, ([2.0] + [40.0] * 4) * 20)
r = occlusion.check_occlusion({"a": pa_scat, "b": pb_ok})
check("散布 20 个孤立低纹理帧 (总量 20%) -> 不删", not r.drop_episode)
check("总量确实是 20%", abs(r.views["a"].occlusion_ratio - 0.20) < 1e-9,
      f"{r.views['a'].occlusion_ratio}")


# =====================================================================
print("\n[stillness] state 阈值必传")
# =====================================================================
state_still = [[0.0, 0.0]] * 100
raises("不传 state_diff_thresh -> 报错",
       lambda: stillness.check_stillness({"a": mkprobe("a", [1.0] * 100, [1.0] * 100)},
                                         state_still),
       "必须显式传入 state_diff_thresh")
raises("state_diff_thresh <= 0 -> 报错",
       lambda: stillness.check_stillness({"a": mkprobe("a", [1.0] * 100, [1.0] * 100)},
                                         state_still, params={"state_diff_thresh": 0}),
       "必须为正数")
raises("state 行数 != 视频帧数 -> 报错",
       lambda: stillness.check_stillness({"a": mkprobe("a", [1.0] * 100, [1.0] * 100)},
                                         [[0.0, 0.0]] * 90,
                                         params={"state_diff_thresh": 1e-3}),
       "不做截断对齐")

print("\n[stillness] state AND image; 阈值 10% (与 black/occlusion 一致)")
P = {"state_diff_thresh": 1e-3}
# 前 9 帧 state 不动 + 画面不动 -> 静止段 [1,9) 共 8 帧; 8/99 < 10% -> 不删
st = [[0.0, 0.0]] * 9 + [[float(i), 0.0] for i in range(91)]
fd = [0.1] * 9 + [9.0] * 91
pa = mkprobe("a", [120.0] * 100, [40.0] * 100, fd=fd)
r = stillness.check_stillness({"a": pa}, st, params=P)
check("连续 8 帧 (8%) -> 不删", not r.drop_episode,
      f"run_ratio={r.views['a'].longest_still_run_ratio}")
check("静止段是连续的 [1,9)", r.views["a"].still_ranges == [(1, 9)],
      f"{r.views['a'].still_ranges}")
check("默认阈值是 0.10", stillness.DEFAULT_PARAMS["max_still_run_ratio"] == 0.10)

# 前 20 帧静止 -> 静止段 [1,20) 共 19 帧, 19/99 = 19% >= 10% -> 删整条
st20 = [[0.0, 0.0]] * 20 + [[float(i), 0.0] for i in range(80)]
r20 = stillness.check_stillness(
    {"a": mkprobe("a", [120.0] * 100, [40.0] * 100, fd=[0.1] * 20 + [9.0] * 80)},
    st20, params=P)
check("连续 19 帧 (19%) -> 删整条", r20.drop_episode,
      f"run_ratio={r20.views['a'].longest_still_run_ratio}")
check("规则码是 STILLNESS_RUN_RATIO_EXCEEDED",
      r20.triggered_rules == ["STILLNESS_RUN_RATIO_EXCEEDED"], f"{r20.triggered_rules}")
check("evidence 给出最长静止段的位置",
      r20.findings[0].evidence["per_view_over"]["a"]["longest_still_run_span"] == [1, 20],
      f"{r20.findings[0].evidence['per_view_over']['a']['longest_still_run_span']}")

print("\n[stillness] 连续 vs 总量: 操作中的瞬时微停顿不该判死")
# state 每两帧才变一次 -> 奇数帧 state 静止; 画面同样只在奇数帧不动。
# 结果是 50 个**孤立**的静止帧: 总量 50%, 但最长连续段只有 1 帧 -> 不删。
# 这正是"减速接近/抓取瞬间/换手过渡"在真实数据里的样子。
r_scatter = stillness.check_stillness(
    {"a": mkprobe("a", [120.0] * 100, [40.0] * 100,
                  fd=[9.0 if i % 2 == 0 else 0.1 for i in range(100)])},
    [[float(i // 2), 0.0] for i in range(100)], params=P)
check("50 个孤立静止帧 (总量 50%) -> 不删", not r_scatter.drop_episode,
      f"total={r_scatter.views['a'].still_ratio}")
check("总量确实过半", r_scatter.views["a"].still_ratio >= 0.4,
      f"{r_scatter.views['a'].still_ratio}")
check("最长连续段只有 1 帧", r_scatter.views["a"].longest_still_run == 1,
      f"{r_scatter.views['a'].longest_still_run}")

# state 静止但画面在动 (场景里有人) -> 不算静止
r = stillness.check_stillness({"a": mkprobe("a", [120.0] * 100, [40.0] * 100,
                                            fd=[9.0] * 100)}, st, params=P)
check("state 静止但画面在动 -> 不算静止", r.views["a"].n_still == 0)

# 画面静止但 state 在动 (机械臂动作幅度小) -> 不算静止
st_move = [[float(i), 0.0] for i in range(100)]
r = stillness.check_stillness({"a": mkprobe("a", [120.0] * 100, [40.0] * 100,
                                            fd=[0.1] * 100)}, st_move, params=P)
check("画面静止但 state 在动 -> 不算静止", r.views["a"].n_still == 0)

print("\n[stillness] 逐路独立 (旧版 all(axis=1) 会让整个检测失效)")
# a 路画面静止, b 路有噪点。旧版要求所有相机都静止 -> 一帧都判不出
pa = mkprobe("a", [120.0] * 100, [40.0] * 100, fd=[0.1] * 20 + [9.0] * 80)
pb = mkprobe("b", [120.0] * 100, [40.0] * 100, fd=[9.0] * 100)
r = stillness.check_stillness({"a": pa, "b": pb}, st20, params=P)
check("一路有噪点不影响另一路判定", r.drop_episode and r.findings[0].views == ["a"],
      f"views={r.findings[0].views if r.findings else None}")
check("b 路占比为 0", r.views["b"].still_ratio == 0.0)

print("\n[stillness] 夹爪帧不被判成静止 (旧版方向做反了)")
# 手臂 (dim0) 全程不动, 夹爪 (dim1) 在第 5-10 帧闭合
st_grip = [[0.0, 0.0]] * 5 + [[0.0, float(i)] for i in range(1, 6)] + [[0.0, 5.0]] * 90
r = stillness.check_stillness({"a": mkprobe("a", [120.0] * 100, [40.0] * 100,
                                            fd=[0.1] * 100)}, st_grip, params=P)
still = set(range(*r.views["a"].still_ranges[0])) if r.views["a"].still_ranges else set()
for rng in r.views["a"].still_ranges[1:]:
    still |= set(range(*rng))
check("夹爪闭合的 5 帧不在静止集合里", not ({5, 6, 7, 8, 9} & still),
      f"still={sorted(still)[:20]}")


# =====================================================================
print("\n[通用] 报错而非静默降级")
# =====================================================================
for mod_name, fn in [
    ("black", lambda ps: black.check_black(ps)),
    ("occlusion", lambda ps: occlusion.check_occlusion(ps)),
]:
    raises(f"{mod_name}: 无 probe -> 报错", lambda f=fn: f({}), "未提供任何相机")
    raises(f"{mod_name}: 全部相机不可用 -> 报错",
           lambda f=fn: f({"a": VideoProbe(cam="a", missing_file=True)}), "没有一路可用")
    raises(f"{mod_name}: 拿不到 fps -> 报错",
           lambda f=fn: f({"a": mkprobe("a", [1.0] * 5, [1.0] * 5, fps=0.0)}), "帧率")
    raises(f"{mod_name}: fps 不一致 -> 报错",
           lambda f=fn: f({"a": mkprobe("a", [1.0] * 5, [1.0] * 5, fps=30.0),
                           "b": mkprobe("b", [1.0] * 5, [1.0] * 5, fps=10.0)}), "帧率不一致")
    raises(f"{mod_name}: 时间轴长度不等 -> 报错",
           lambda f=fn: f({"a": mkprobe("a", [1.0] * 5, [1.0] * 5),
                           "b": mkprobe("b", [1.0] * 7, [1.0] * 7)}), "长度不一致")
    raises(f"{mod_name}: 未知参数 -> 报错",
           lambda f=fn: (black.check_black({"a": mkprobe("a", [1.0] * 5, [1.0] * 5)},
                                           params={"nope": 1})
                         if mod_name == "black" else
                         occlusion.check_occlusion({"a": mkprobe("a", [1.0] * 5, [1.0] * 5)},
                                                   params={"nope": 1})), "未知参数")

pr_nogray = VideoProbe(cam="a", fps=30.0, n_readable=5, n_frames=5)
raises("black: probe 没带灰度 -> 报错",
       lambda: black.check_black({"a": pr_nogray}), "没有带逐帧灰度")
raises("occlusion: probe 没带灰度 -> 报错",
       lambda: occlusion.check_occlusion({"a": pr_nogray}), "没有灰度数据")
raises("stillness: probe 没带 frame_diff -> 报错",
       lambda: stillness.check_stillness(
           {"a": VideoProbe(cam="a", fps=30.0, n_readable=5, n_frames=5,
                            gray_mean=[1.0] * 5, gray_std=[1.0] * 5)},
           [[0.0]] * 5, params=P), "没有 frame_diff")


# =====================================================================
print("\n[通用] to_dict() 可序列化")
# =====================================================================
import json  # noqa: E402
pa = mkprobe("a", [2.0] * 20 + [120.0] * 80, [0.4] * 20 + [40.0] * 80, fd=[0.1] * 100)
reports = {
    "black": black.check_black({"a": pa}),
    "occlusion": occlusion.check_occlusion({"a": pa}),
    "stillness": stillness.check_stillness({"a": pa}, [[0.0]] * 100, params=P),
}
check("三个检测的删除门槛统一为 0.10",
      stillness.DEFAULT_PARAMS["max_still_run_ratio"]
      == black.DEFAULT_PARAMS["max_black_run_ratio"]
      == occlusion.DEFAULT_PARAMS["max_occlusion_run_ratio"] == 0.10)
check("三个检测的判据统一为最长连续段占比",
      all(k.endswith("_run_ratio") for k in
          [next(k for k in black.DEFAULT_PARAMS if k.startswith("max_")),
           next(k for k in occlusion.DEFAULT_PARAMS if k.startswith("max_")),
           next(k for k in stillness.DEFAULT_PARAMS if k.startswith("max_"))]))
for name, rep in reports.items():
    try:
        s = json.dumps(rep.to_dict(), ensure_ascii=False)
        check(f"{name}.to_dict() 能 json 序列化", len(s) > 0)
    except Exception as e:
        check(f"{name}.to_dict() 能 json 序列化", False, str(e))
    check(f"{name} 的 disposition 全是 drop_episode",
          all(f.disposition == "drop_episode" for f in rep.findings))
    check(f"{name} 没有 drop_frames 字段", "drop_frames" not in rep.to_dict())


# =====================================================================
print("\n[pipeline] 编排与删除率预算")
# =====================================================================
from check_pipeline.pipeline import (  # noqa: E402
    ALL_CHECKS, CHECK_STAGE, CheckOutcome, DropBudget, EpisodeResult)

N_CHECKS = len(ALL_CHECKS)


class _FakeReport:
    def __init__(self, drop, rules=None):
        self._d = drop
        self.triggered_rules = rules or []
        self.findings = []

    @property
    def drop_episode(self):
        return self._d

    def to_dict(self):
        return {}


def mkresult(eid, dropped=(), failed=False, stage=None):
    r = EpisodeResult(episode_id=eid, failed=failed, failed_stage=stage)
    if not failed:
        for c in ALL_CHECKS:
            r.outcomes[c] = CheckOutcome(
                check=c, stage=CHECK_STAGE[c], status="ok",
                report=_FakeReport(c in dropped,
                                   [f"{c.upper()}_RULE"] if c in dropped else []))
    return r


check("dropped_by 按固定顺序, 不受命中顺序影响",
      mkresult("x", ("state_extreme", "stillness", "corrupt", "black")).dropped_by
      == ["corrupt", "black", "stillness", "state_extreme"])
check("failed 的 episode 不算 drop", not mkresult("y", failed=True, stage="probe").drop_episode)

# 同样的命中率, 交叠与否让总删除率差 N 倍 —— 这正是 DropBudget 存在的理由
b_overlap = DropBudget()
for i in range(100):
    b_overlap.add(mkresult(f"a{i}", ALL_CHECKS if i < 5 else ()))
b_disjoint = DropBudget()
for i in range(100):
    d = ()
    for j, c in enumerate(ALL_CHECKS):
        if j * 5 <= i < j * 5 + 5:
            d = (c,)
    b_disjoint.add(mkresult(f"b{i}", d))
do, dd = b_overlap.to_dict(), b_disjoint.to_dict()
check(f"{N_CHECKS} 个检测各命中 5% 且完全重合 -> 总删除率 5%", do["drop_rate"] == 0.05)
check(f"{N_CHECKS} 个检测各命中 5% 且互不相交 -> 总删除率 {N_CHECKS * 5}%",
      dd["drop_rate"] == round(N_CHECKS * 0.05, 6), str(dd["drop_rate"]))
check(f"命中率相同, 总删除率差 {N_CHECKS} 倍",
      round(do["drop_rate"] * N_CHECKS, 6) == dd["drop_rate"])
check(f"完全重合时唯一归因全为 0 (有 {N_CHECKS - 1} 个检测冗余)",
      all(v["only"] == 0 for v in do["per_check"].values()))
check(f"互不相交时唯一归因全为 5 ({N_CHECKS} 类独立问题)",
      all(v["only"] == 5 for v in dd["per_check"].values()))

b_fail = DropBudget()
for i in range(100):
    b_fail.add(mkresult(f"c{i}", ("black",) if i < 10 else (),
                        failed=(i >= 90), stage="probe"))
df = b_fail.to_dict()
check("检测失败的不进分母 (分母 90 不是 100)", df["n_judged"] == 90)
check("删除率 = 10/90, 失败不被算成通过", df["drop_rate"] == round(10 / 90, 6))

b_pair = DropBudget()
for i in range(100):
    b_pair.add(mkresult(f"d{i}", ("black", "occlusion") if i < 10
                        else (("black",) if i < 15 else ())))
dp = b_pair.to_dict()
check("两两交叠计数正确", dp["pair_overlap"].get("black&occlusion") == 10)
check("被完全覆盖的检测唯一归因为 0 (关掉它救不回数据)",
      dp["per_check"]["occlusion"]["only"] == 0)


print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
