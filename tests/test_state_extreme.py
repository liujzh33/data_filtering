"""state_extreme (极值过滤) 的规则回归测试。

只依赖 numpy (本检测的判据里没有任何滤波或按时间定义的量), 所以有真 numpy 用真的,
没有就退回 _npstub 替身。

用法: python3 -m check_pipeline.tests.test_state_extreme
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from check_pipeline.tests import _npstub  # noqa: E402
print(f"numpy: {'替身 (_npstub)' if _npstub.install_if_missing() else '真库'}")

import numpy as np  # noqa: E402

from check_pipeline import state_extreme as se  # noqa: E402
from check_pipeline.errors import UnsupportedDataError  # noqa: E402

PASS, FAIL = [], []
Q1, Q99 = -1.0, 1.0
SPAN = Q99 - Q1


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


def mat(*cols):
    """按列拼成 (T, D) 矩阵 (替身也支持的最朴素写法)。"""
    T = len(cols[0])
    return np.asarray([[float(c[t]) for c in cols] for t in range(T)])


def prof(n=1, **kw):
    d = {"enabled": True, "source": "action", "percentile_scope": "ds/ALOHA",
         "dims": [{"name": f"j{i}", "column": "action", "dim": i, "q1": Q1, "q99": Q99}
                  for i in range(n)]}
    d.update(kw)
    return d


def run(m, p=None, params=None):
    n = m.shape[1] if hasattr(m, "shape") else 1
    return se.check_state_extreme({"action": m}, p or prof(n), params=params)


# 一条正常轨迹: 全部落在 [q1, q99] 内
NORMAL = [0.5 * (1 if t % 7 < 4 else -1) * (t % 11) / 11.0 for t in range(200)]


# =====================================================================
print("\n[state_extreme] 正常数据不命中")
# =====================================================================
r0 = run(mat(NORMAL))
check("全部落在 band 内 -> 零命中", r0.n_frames_flagged == 0)
check("零命中时无 finding", not r0.has_extreme)
check("零命中时不删帧", not r0.drop_frames)
check("frames_to_drop 为空", r0.frames_to_drop == [])
check("flagged_ratio 为 0", r0.flagged_ratio == 0.0)


# =====================================================================
print("\n[state_extreme] band 由 alpha 在运行时算出 (旧版 alpha 是死参数)")
# =====================================================================
r_a0 = run(mat(NORMAL), params={"alpha": 0.0})
r_a1 = run(mat(NORMAL), params={"alpha": 1.0})
r_a3 = run(mat(NORMAL), params={"alpha": 3.0})
check("alpha=0 -> band = [q1, q99]",
      r_a0.dims["j0"].lo == Q1 and r_a0.dims["j0"].hi == Q99)
check("alpha=1 -> band 外扩一个 span",
      abs(r_a1.dims["j0"].lo - (Q1 - SPAN)) < 1e-12 and
      abs(r_a1.dims["j0"].hi - (Q99 + SPAN)) < 1e-12)
check("alpha=3 -> band 外扩三个 span",
      abs(r_a3.dims["j0"].lo - (Q1 - 3 * SPAN)) < 1e-12)
check("alpha 单调放宽 band", r_a0.dims["j0"].lo > r_a1.dims["j0"].lo > r_a3.dims["j0"].lo)
check("默认 alpha = 1.0", se.DEFAULT_PARAMS["alpha"] == 1.0)

# 同一条数据, alpha 改变命中结果 —— 旧版做不到 (直接取文件里烘死的 lo/hi)
tail = list(NORMAL); tail[50] = 1.5      # 落在 [q1,q99] 外, 但在 alpha=1 的 band 内
check("alpha=0 时 1.5 命中", run(mat(tail), params={"alpha": 0.0}).n_frames_flagged == 1)
check("alpha=1 时 1.5 放过", run(mat(tail), params={"alpha": 1.0}).n_frames_flagged == 0)
check("alpha 是活参数 (旧版这里没有任何效果)",
      run(mat(tail), params={"alpha": 0.0}).n_frames_flagged !=
      run(mat(tail), params={"alpha": 1.0}).n_frames_flagged)


# =====================================================================
print("\n[state_extreme] 真离群值在大 alpha 下仍被抓住")
# =====================================================================
glitch = list(NORMAL); glitch[80] = 20 * SPAN        # 传感器毛刺 / 单位搞错
for a in (0.0, 0.5, 1.0, 2.0, 3.0):
    r = run(mat(glitch), params={"alpha": a})
    check(f"alpha={a} 仍检出 20 倍量程的毛刺", r.n_frames_flagged >= 1)
check("毛刺区间定位到第 80 帧", (80, 81) in run(mat(glitch)).frames_to_drop,
      str(run(mat(glitch)).frames_to_drop))


# =====================================================================
print("\n[state_extreme] alpha_needed_to_clear: 支撑'跑一批再调'的闭环")
# =====================================================================
x = list(NORMAL); x[10] = Q99 + 0.5 * SPAN           # 越界 0.5 个 span
r = run(mat(x), params={"alpha": 0.0})
check("alpha_needed_to_clear = 越界距离 / span",
      abs(r.dims["j0"].alpha_needed_to_clear - 0.5) < 1e-9,
      str(r.dims["j0"].alpha_needed_to_clear))
check("把 alpha 调到该值以上就不再命中",
      run(mat(x), params={"alpha": 0.51}).n_frames_flagged == 0)
check("调到该值以下仍命中", run(mat(x), params={"alpha": 0.49}).n_frames_flagged == 1)
check("正常数据的 alpha_needed_to_clear 为 0", r0.dims["j0"].alpha_needed_to_clear == 0.0)


# =====================================================================
print("\n[state_extreme] 多维取并集")
# =====================================================================
a = list(NORMAL); a[30] = 10.0
b = list(NORMAL); b[60] = -10.0
c = list(NORMAL)
rm = run(mat(a, b, c), p=prof(3))
check("任一维度越界即该帧命中 (并集)", rm.n_frames_flagged == 2)
check("命中区间是两个散点", len(rm.flagged_ranges) == 2, str(rm.flagged_ranges))
check("命中维度记在 dims", set(rm.findings[0].dims) == {"j0", "j1"},
      str(rm.findings[0].dims))
ev = rm.findings[0].evidence
check("未命中维度也列在 evidence 里", set(ev["per_dim_n_flagged"]) == {"j0", "j1", "j2"})
check("未命中维度计数为 0", ev["per_dim_n_flagged"]["j2"] == 0)
check("evidence 报出段数 (极值多为散点, 段数决定下游怎么删)", ev["n_segments"] == 2)
check("evidence 报出 flagged_ratio (供流水线删除预算用)",
      abs(ev["flagged_ratio"] - 2 / 200) < 1e-9)


# =====================================================================
print("\n[state_extreme] 只做帧级, 没有 episode 级规则")
# =====================================================================
half = [50.0 if t < 100 else NORMAL[t] for t in range(200)]     # 一半帧全是极值
rh = run(mat(half))
check("50% 帧越界仍只删帧", rh.drop_frames and not rh.drop_episode)
check("drop_episode 恒为 False", rh.drop_episode is False)
check("RULES 表里只有 drop_frames", set(v[1] for v in se.RULES.values()) == {"drop_frames"})
check("规则只有一条", list(se.RULES) == ["STATE_EXTREME_FRAMES"])
check("flagged_ratio 如实报出 0.5, 交给流水线预算判断",
      abs(rh.flagged_ratio - 0.5) < 1e-9)
check("区间左闭右开", rh.frames_to_drop == [(0, 100)], str(rh.frames_to_drop))

single = list(NORMAL); single[5] = 99.0
check("单帧记为 [5, 6)", run(mat(single)).frames_to_drop == [(5, 6)],
      str(run(mat(single)).frames_to_drop))
check("删除区间不加 margin", run(mat(single)).frames_to_drop ==
      run(mat(single)).flagged_ranges)


# =====================================================================
print("\n[state_extreme] profile 与数据对不上时自检")
# =====================================================================
# 分位数拿错机型: 本 episode 各维中位数远在 [q1,q99] 之外
wrong = {"enabled": True, "source": "action", "percentile_scope": "ds/WRONG_ROBOT",
         "dims": [{"name": f"j{i}", "column": "action", "dim": i, "q1": 50.0, "q99": 52.0}
                  for i in range(3)]}
raises("过半维度中位数在 [q1,q99] 外 -> 报错",
       lambda: se.check_state_extreme({"action": mat(NORMAL, NORMAL, NORMAL)}, wrong),
       "几乎只能是这份分位数不属于当前机型")
raises("报错里带上 percentile_scope 便于核对",
       lambda: se.check_state_extreme({"action": mat(NORMAL, NORMAL, NORMAL)}, wrong),
       "ds/WRONG_ROBOT")
# 只有一维偏离时不该误报
part = {"enabled": True, "source": "action", "percentile_scope": "ds/ALOHA",
        "dims": [{"name": "j0", "column": "action", "dim": 0, "q1": 50.0, "q99": 52.0},
                 {"name": "j1", "column": "action", "dim": 1, "q1": Q1, "q99": Q99},
                 {"name": "j2", "column": "action", "dim": 2, "q1": Q1, "q99": Q99}]}
rp = se.check_state_extreme({"action": mat(NORMAL, NORMAL, NORMAL)}, part)
check("只有 1/3 维度偏离 -> 不报错, 正常出结果", rp.n_frames_flagged == 200)
# 维度数 < 3 时不做这个检查 (分不清'拿错分位数'和'罕见 episode')
few = {"enabled": True, "source": "action", "percentile_scope": "ds/ALOHA",
       "dims": [{"name": "j0", "column": "action", "dim": 0, "q1": 50.0, "q99": 52.0}]}
r_few = se.check_state_extreme({"action": mat(NORMAL)}, few)
check("单维度不触发自检", r_few.n_frames_flagged == 200)


# =====================================================================
print("\n[state_extreme] profile 校验: 不猜, 直接报错")
# =====================================================================
raises("profile 缺失 -> 报错",
       lambda: se.check_state_extreme({"action": mat(NORMAL)}, {}),
       "必须由 profile 显式给出")
raises("enabled=false -> 报错, 由调用方记为跳过",
       lambda: run(mat(NORMAL), p=prof(1, enabled=False)), "记为'跳过'")
raises("dims 为空 -> 报错",
       lambda: run(mat(NORMAL), p=prof(1, dims=[])), "dims 为空")
raises("缺 source -> 报错 (旧版拿 action 分位数卡 state)",
       lambda: run(mat(NORMAL), p=prof(1, source="")), "必须是 'state' 或 'action'")
raises("source 取值非法 -> 报错",
       lambda: run(mat(NORMAL), p=prof(1, source="joint")), "必须是 'state' 或 'action'")
raises("缺 percentile_scope -> 报错",
       lambda: run(mat(NORMAL), p=prof(1, percentile_scope="")), "必须可追溯")
raises("缺 q1 -> 报错 (存 lo/hi 等于把 alpha 烘死)",
       lambda: run(mat(NORMAL), p={"enabled": True, "source": "action",
                                   "percentile_scope": "s",
                                   "dims": [{"name": "x", "column": "action",
                                             "dim": 0, "q99": 1.0}]}),
       "而不是算好的")
raises("q99 <= q1 -> 报错",
       lambda: run(mat(NORMAL), p={"enabled": True, "source": "action",
                                   "percentile_scope": "s",
                                   "dims": [{"name": "x", "column": "action", "dim": 0,
                                             "q1": 1.0, "q99": 1.0}]}),
       "没动过")
raises("dim 越界 -> 报错",
       lambda: run(mat(NORMAL), p={"enabled": True, "source": "action",
                                   "percentile_scope": "s",
                                   "dims": [{"name": "x", "column": "action", "dim": 9,
                                             "q1": Q1, "q99": Q99}]}),
       "超出列")
raises("列不存在 -> 报错",
       lambda: run(mat(NORMAL), p={"enabled": True, "source": "action",
                                   "percentile_scope": "s",
                                   "dims": [{"name": "x", "column": "nope", "dim": 0,
                                             "q1": Q1, "q99": Q99}]}),
       "不在提供的信号里")
raises("维度名重复 -> 报错",
       lambda: run(mat(NORMAL, NORMAL), p={"enabled": True, "source": "action",
                                           "percentile_scope": "s", "dims": [
           {"name": "x", "column": "action", "dim": 0, "q1": Q1, "q99": Q99},
           {"name": "x", "column": "action", "dim": 1, "q1": Q1, "q99": Q99}]}),
       "维度名重复")
raises("未知参数 -> 报错", lambda: run(mat(NORMAL), params={"lo_q": 5}), "未知参数")
raises("alpha 为负 -> 报错", lambda: run(mat(NORMAL), params={"alpha": -1.0}), "非负")
raises("各列行数不一致 -> 报错, 不截断",
       lambda: se.check_state_extreme(
           {"action": mat(NORMAL), "b": mat(NORMAL[:50])},
           prof(1)),
       "行数不一致")

nanx = list(NORMAL); nanx[7] = float("nan")
raises("NaN -> 报错 (旧版会让整个维度静默失效)", lambda: run(mat(nanx)),
       "该维度一帧都不会被标记")

# profile 覆盖 alpha
check("profile 可覆盖 alpha", run(mat(tail), p=prof(1, alpha=0.0)).n_frames_flagged == 1)


# =====================================================================
print("\n[state_extreme] 输出结构")
# =====================================================================
d = run(mat(glitch)).to_dict()
check("to_dict() 能 json 序列化", isinstance(json.dumps(d), str))
check("顶层有 check 名", d["check"] == "state_extreme")
check("顶层有 triggered_rules", d["triggered_rules"] == ["STATE_EXTREME_FRAMES"])
check("顶层 drop_episode 恒 False", d["drop_episode"] is False)
check("measurements 记录 source", d["measurements"]["source"] == "action")
check("measurements 记录 percentile_scope", d["measurements"]["percentile_scope"] == "ds/ALOHA")
check("measurements 记录 flagged_ratio", "flagged_ratio" in d["measurements"])
check("per_dim 记录 q1/q99 与算出的 band",
      d["measurements"]["per_dim"]["j0"]["q1"] == Q1 and
      d["measurements"]["per_dim"]["j0"]["lo"] == Q1 - SPAN)
check("per_dim 记录 episode 的 min/max/median",
      all(k in d["measurements"]["per_dim"]["j0"]
          for k in ("episode_min", "episode_max", "episode_median")))
check("每条规则码都在 RULES 表里", all(x.rule in se.RULES for x in run(mat(glitch)).findings))
check("DEFAULT_PARAMS 只剩 alpha 一项 (旧版 8 项)", list(se.DEFAULT_PARAMS) == ["alpha"])
check("参数表里没有旧版那些分档阈值",
      not ({"extreme_mild_ratio", "extreme_moderate_ratio", "extreme_severe_ratio",
            "extreme_critical_ratio", "lo_q", "hi_q"} & set(se.DEFAULT_PARAMS)))


print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for n in FAIL:
        print("  FAILED:", n)
    sys.exit(1)
