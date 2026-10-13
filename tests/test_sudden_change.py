"""sudden_change (突变检测) 的规则回归测试。

**这个模块必须用真 numpy + 真 scipy 测。** _scipystub 的 savgol_filter 是恒等映射,
等于关掉平滑, 而本模块的响应倍数标定 (_response_factors) 正是靠"阶跃推过滤波器"量
出来的 —— 恒等映射下残差响应为 0, 整个归一化失去意义。没有真库时直接跳过并明确说明,
不伪装成通过。

用法: python3 -m check_pipeline.tests.test_sudden_change
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import numpy as np
    import scipy.signal  # noqa: F401
except Exception as _e:
    print("=" * 60)
    print("SKIPPED: sudden_change 的测试需要真 numpy + 真 scipy")
    print(f"  当前环境: {_e}")
    print("  替身的 savgol_filter 是恒等映射, 测不了响应倍数标定, 跳过而不是假装通过")
    print("=" * 60)
    sys.exit(0)

from check_pipeline import sudden_change as sc  # noqa: E402
from check_pipeline.errors import (  # noqa: E402
    PipelineEnvironmentError, UnsupportedDataError)
from check_pipeline.tests import _scipystub  # noqa: E402

PASS, FAIL = [], []
FPS = 30.0
RANGE = 3.14


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


# ---------------------------------------------------------------------
def traj(T=300, rng=3.0, quant=1e-4, seed=1):
    """平滑关节轨迹 + 量化噪声, 峰峰值约 rng。"""
    rs = np.random.RandomState(seed)
    t = np.arange(T)
    x = rng / 2 * np.sin(2 * np.pi * t / T) + rng / 4 * np.sin(2 * np.pi * t / (T / 3))
    return np.round(x / quant) * quant + rs.uniform(-quant / 2, quant / 2, T)


def prof(n=1, **kw):
    d = {"enabled": True,
         "dims": [{"name": f"j{i}", "column": "s", "dim": i, "range": RANGE}
                  for i in range(n)]}
    d.update(kw)
    return d


def run(x, p=None, params=None, fps=FPS):
    mat = x.reshape(-1, 1) if x.ndim == 1 else x
    return sc.check_sudden_change({"s": mat}, p or prof(mat.shape[1]), fps, params=params)


BASE = traj()


# =====================================================================
print("\n[sudden_change] 响应倍数标定")
# =====================================================================
f30 = sc._response_factors(3, sc._sg_window(dict(sc.DEFAULT_PARAMS), 30.0), 2)
f60 = sc._response_factors(3, sc._sg_window(dict(sc.DEFAULT_PARAMS), 60.0), 2)
check("加速度响应是解析值 1.0 (只依赖差分阶数)", abs(f30["accel"] - 1.0) < 1e-9)
check("跃度响应是解析值 2.0", abs(f30["jerk"] - 2.0) < 1e-9)
check("残差响应 = 1/3 (w=7,p=2 的 SG 中心系数 7/21)", abs(f30["residual"] - 1 / 3) < 1e-6)
check("残差响应随帧率(窗口)改变 -> 必须运行时标定, 不能写死常数",
      abs(f60["residual"] - f30["residual"]) > 0.05,
      f"30fps={f30['residual']:.4f} 60fps={f60['residual']:.4f}")

check("低帧率窗口被夹到阶数下限, 不会崩", sc._sg_window(dict(sc.DEFAULT_PARAMS), 10.0) == 5)
check("窗口恒为奇数", all(sc._sg_window(dict(sc.DEFAULT_PARAMS), v) % 2 == 1
                          for v in (10, 15, 24, 30, 50, 60)))
r10 = run(traj(T=120), fps=10.0)
check("10fps 能跑通 (旧写法在这里抛 UnsupportedDataError)", r10.n_frames == 120)
check("生效窗口写进 params, 与请求秒数不符时可见",
      r10.params["sg_window_frames"] == 5 and r10.params["sg_window_sec_effective"] == 0.5)


# =====================================================================
print("\n[sudden_change] 干净数据不误报")
# =====================================================================
clean = np.stack([traj(seed=i) for i in range(10)], axis=1)
rc = run(clean, p=prof(10))
check("10 条干净通道零命中", rc.n_frames_anomalous == 0, f"命中 {rc.n_frames_anomalous}")
check("干净数据无 finding", not rc.has_anomaly)
check("干净数据不删 episode", not rc.drop_episode)
check("干净数据不删帧", not rc.drop_frames)
check("干净数据 frames_to_drop 为空", rc.frames_to_drop == [])


# =====================================================================
print("\n[sudden_change] 尖峰与阶跃的灵敏度")
# =====================================================================
def spike(frac, at=150):
    x = BASE.copy()
    x[at] += frac * RANGE
    return x


def step(frac, at=150):
    x = BASE.copy()
    x[at:] += frac * RANGE
    return x


check("尖峰 2% 量程 -> 检出", run(spike(0.02)).has_anomaly)
check("尖峰 0.5% 量程 -> 不检出 (低于阈值)", not run(spike(0.005)).has_anomaly)
check("阶跃 3% 量程 -> 检出", run(step(0.03)).has_anomaly)
check("阶跃 0.5% 量程 -> 不检出", not run(step(0.005)).has_anomaly)
check("阶跃与尖峰的门槛差距 <2 倍 (旧版差 3.3 倍)",
      run(step(0.03)).has_anomaly and run(spike(0.02)).has_anomaly)

loose = run(spike(0.01), params={"jump_frac": 0.004})
tight = run(spike(0.01), params={"jump_frac": 0.10})
check("jump_frac 调小 -> 同一尖峰被检出", loose.has_anomaly)
check("jump_frac 调大 -> 同一尖峰放过", not tight.has_anomaly)
check("jump_frac 是单调的唯一阈值", loose.has_anomaly and not tight.has_anomaly)


# =====================================================================
print("\n[sudden_change] 阈值不随 episode 运动幅度漂移 (量程来自 profile)")
# =====================================================================
# 同一个关节, 两条 episode 运动幅度差 10 倍, 打一样绝对幅度的尖峰。
# 量程来自 profile, 所以两条应该给出一样的判定。旧版用本条 episode 的 p99-p1 当尺度,
# 动得少的那条阈值会紧 10 倍。
big = traj(rng=3.0, seed=5)
small = traj(rng=0.3, seed=5)
d_abs = 0.03 * RANGE
xb = big.copy(); xb[150] += d_abs
xs = small.copy(); xs[150] += d_abs
rb, rs_ = run(xb), run(xs)
check("大幅运动 episode 检出", rb.has_anomaly)
check("小幅运动 episode 同样检出", rs_.has_anomaly)
check("两条 episode 阈值完全相同",
      rb.channels["j0"].threshold == rs_.channels["j0"].threshold)
check("阈值 = jump_frac × profile range",
      abs(rb.channels["j0"].threshold - 0.02 * RANGE) < 1e-12)


# =====================================================================
print("\n[sudden_change] 两条规则与删除区间")
# =====================================================================
r1 = run(spike(0.05))
check("少量异常帧 -> SUDDEN_CHANGE_SHORT", r1.triggered_rules == ["SUDDEN_CHANGE_SHORT"])
check("SHORT 的 disposition 是 drop_frames", r1.findings[0].disposition == "drop_frames")
check("SHORT 不删 episode", not r1.drop_episode)
check("SHORT 要删帧", r1.drop_frames)
check("异常帧数 <= max_frame_drop", r1.n_frames_anomalous <= sc.DEFAULT_PARAMS["max_frame_drop"])

# table.txt(48): 区间左闭右开, 单帧要记成 [i, i+1)
single = sc.check_sudden_change(
    {"s": spike(0.02).reshape(-1, 1)},
    {"enabled": True, "dims": [{"name": "j0", "column": "s", "dim": 0, "range": RANGE}]},
    FPS, params={"jump_frac": 0.019})
check("单帧异常记为 [i, i+1) 而不是 [i, i]",
      all(b - a >= 1 for a, b in single.frames_to_drop) and bool(single.frames_to_drop),
      str(single.frames_to_drop))
check("删除区间不加 margin (等于真实异常区间)",
      r1.frames_to_drop == r1.anomaly_ranges, f"{r1.frames_to_drop} vs {r1.anomaly_ranges}")
check("异常区间落在尖峰附近", any(a <= 151 and b >= 150 for a, b in r1.anomaly_ranges),
      str(r1.anomaly_ranges))

# 大量异常帧 -> 整条丢
many = BASE.copy()
rs2 = np.random.RandomState(0)
many[100:160] += rs2.uniform(-0.5, 0.5, 60)      # 一段全程乱跳
r2 = run(many)
check("大量异常帧 -> SUDDEN_CHANGE_LONG", r2.triggered_rules == ["SUDDEN_CHANGE_LONG"])
check("LONG 的 disposition 是 drop_episode", r2.findings[0].disposition == "drop_episode")
check("LONG 时 frames_to_drop 为空 (整条都丢, 没有'删哪些'这回事)", r2.frames_to_drop == [])
check("LONG 的 frame_ranges 仍作为诊断保留", len(r2.findings[0].frame_ranges) > 0)

r3 = run(many, params={"max_frame_drop": 10 ** 6})
check("max_frame_drop 调大 -> 同样的异常改判 drop_frames",
      r3.triggered_rules == ["SUDDEN_CHANGE_SHORT"] and r3.drop_frames)


# =====================================================================
print("\n[sudden_change] 全程乱跳的通道必须被检出 (旧版在这里静默跳过)")
# =====================================================================
# 旧版 max_noise_ratio=0.15: 实测全程乱跳的关节 noise_ratio=0.169 -> 整条通道跳过,
# 零 Issue。越坏越检不出。这一版没有这道过滤。
rs3 = np.random.RandomState(11)
broken = BASE + rs3.normal(0, 0.5, len(BASE))
rb2 = run(broken)
check("全程乱跳 -> 检出 (不是静默跳过)", rb2.has_anomaly)
check("全程乱跳 -> drop_episode", rb2.drop_episode)
check("全程乱跳命中大量帧", rb2.n_frames_anomalous > 50, f"{rb2.n_frames_anomalous}")


# =====================================================================
print("\n[sudden_change] 慢漂移不算突变 (导数确认)")
# =====================================================================
drift = BASE + np.linspace(0, 0.5 * RANGE, len(BASE))   # 缓慢偏移半个量程
check("慢漂移不触发 (残差大但导数小)", not run(drift).has_anomaly)


# =====================================================================
print("\n[sudden_change] 异常段位置标注 (决定下游能否安全截断)")
# =====================================================================
head = BASE.copy(); head[0:2] += 0.10 * RANGE
tail = BASE.copy(); tail[-2:] += 0.10 * RANGE
mid = run(spike(0.05, at=150))
check("中间段标 middle", mid.findings[0].evidence["segment_positions"] == ["middle"])
check("中间段置 has_middle_segment", mid.findings[0].evidence["has_middle_segment"])
rh = run(head)
check("开头段标 head", "head" in rh.findings[0].evidence["segment_positions"],
      str(rh.findings[0].evidence["segment_positions"]))
rt = run(tail)
check("结尾段标 tail", "tail" in rt.findings[0].evidence["segment_positions"],
      str(rt.findings[0].evidence["segment_positions"]))
check("evidence 带中间删帧的风险说明", "伪装" in mid.findings[0].evidence["note_on_middle"])


# =====================================================================
print("\n[sudden_change] 多通道: 并集 + 全量对照")
# =====================================================================
multi = np.stack([traj(seed=i) for i in range(4)], axis=1)
multi[150, 0] += 0.05 * RANGE
multi[200, 2] += 0.05 * RANGE
rm = run(multi, p=prof(4))
check("任一通道命中即该帧异常 (并集)", len(rm.anomaly_ranges) == 2, str(rm.anomaly_ranges))
check("命中的维度都记在 dims", set(rm.findings[0].dims) == {"j0", "j2"},
      str(rm.findings[0].dims))
ev = rm.findings[0].evidence
check("没命中的通道也列在 evidence 里 (区分'单维在跳'和'整帧出问题')",
      set(ev["per_channel_n_flagged"]) == {"j0", "j1", "j2", "j3"})
check("未命中通道计数为 0", ev["per_channel_n_flagged"]["j1"] == 0)
check("evidence 记录检查通道数与命中通道数",
      ev["n_channels_checked"] == 4 and ev["n_channels_hit"] == 2)


# =====================================================================
print("\n[sudden_change] profile 校验: 不猜, 直接报错")
# =====================================================================
raises("profile 缺失 -> 报错",
       lambda: sc.check_sudden_change({"s": BASE.reshape(-1, 1)}, {}, FPS),
       "必须由 profile 显式声明")
raises("enabled=false -> 报错, 由调用方记为跳过",
       lambda: run(BASE, p=prof(1, enabled=False)), "记为'跳过'")
raises("dims 为空 -> 报错", lambda: run(BASE, p={"enabled": True, "dims": []}),
       "dims 为空")
raises("dim 越界 -> 报错",
       lambda: run(BASE, p={"enabled": True,
                            "dims": [{"name": "x", "column": "s", "dim": 7, "range": 1.0}]}),
       "超出列")
raises("列不存在 -> 报错",
       lambda: run(BASE, p={"enabled": True,
                            "dims": [{"name": "x", "column": "nope", "dim": 0, "range": 1.0}]}),
       "不在提供的信号里")
raises("维度名重复 -> 报错",
       lambda: run(BASE, p={"enabled": True, "dims": [
           {"name": "x", "column": "s", "dim": 0, "range": 1.0},
           {"name": "x", "column": "s", "dim": 0, "range": 1.0}]}),
       "维度名重复")
raises("缺 range -> 报错, 不退化成用本条 episode 估",
       lambda: run(BASE, p={"enabled": True,
                            "dims": [{"name": "x", "column": "s", "dim": 0}]}),
       "量程必须由 profile 提供")
raises("range<=0 -> 报错",
       lambda: run(BASE, p={"enabled": True,
                            "dims": [{"name": "x", "column": "s", "dim": 0, "range": 0.0}]}),
       "没动过")
raises("未知参数 -> 报错", lambda: run(BASE, params={"k_residual": 6.0}), "未知参数")
raises("jump_frac 越界 -> 报错", lambda: run(BASE, params={"jump_frac": 1.5}), "必须落在")
raises("max_frame_drop 为负 -> 报错", lambda: run(BASE, params={"max_frame_drop": -1}),
       "不能为负")
raises("fps<=0 -> 报错", lambda: run(BASE, fps=0.0), "拿不到有效帧率")

nan = BASE.copy(); nan[42] = np.nan
raises("NaN -> 报错, 不做插值填补", lambda: run(nan), "凭空造数据")

raises("各列行数不一致 -> 报错, 不截断",
       lambda: sc.check_sudden_change(
           {"s": BASE.reshape(-1, 1), "a": BASE[:100].reshape(-1, 1)},
           {"enabled": True, "dims": [{"name": "x", "column": "s", "dim": 0, "range": 1.0}]},
           FPS),
       "行数不一致")

# profile 覆盖参数
rp = run(spike(0.01), p=prof(1, jump_frac=0.004))
check("profile 可覆盖 jump_frac", rp.has_anomaly)
rp2 = run(many, p=prof(1, max_frame_drop=10 ** 6))
check("profile 可覆盖 max_frame_drop", rp2.drop_frames)


# =====================================================================
print("\n[sudden_change] 输出结构")
# =====================================================================
d = r1.to_dict()
check("to_dict() 能 json 序列化", isinstance(json.dumps(d), str))
check("顶层有 check 名", d["check"] == "sudden_change")
check("顶层有 triggered_rules", d["triggered_rules"] == ["SUDDEN_CHANGE_SHORT"])
check("顶层有 frames_to_drop", isinstance(d["frames_to_drop"], list))
check("每条规则码都在 RULES 表里", all(x.rule in sc.RULES for x in r1.findings))
check("RULES 表的 disposition 只有两种",
      set(v[1] for v in sc.RULES.values()) == {"drop_frames", "drop_episode"})
check("measurements 记录响应倍数", set(d["measurements"]["response_factors"]) ==
      {"residual", "accel", "jerk"})
check("measurements 记录每通道量程与阈值",
      d["measurements"]["per_channel"]["j0"]["range_used"] == RANGE)
check("per_channel 给出最大等效跳变占量程比例 (便于反调 jump_frac)",
      d["measurements"]["per_channel"]["j0"]["max_equiv_jump_frac_of_range"] is not None)
check("参数表里没有旧版那些无效阈值",
      not ({"k_residual", "k_accel", "k_jerk", "max_noise_ratio", "rel_floor",
            "abs_min_range", "sc_margin_sec"} & set(sc.DEFAULT_PARAMS)))
check("DEFAULT_PARAMS 只剩 5 项 (旧版 23 项)", len(sc.DEFAULT_PARAMS) == 5,
      str(sorted(sc.DEFAULT_PARAMS)))


# =====================================================================
print("\n[sudden_change] scipy 缺失必须报环境错")
# =====================================================================
_scipystub.uninstall()
raises("scipy 不可用 -> PipelineEnvironmentError", lambda: run(BASE),
       "scipy 不可用", exc=PipelineEnvironmentError)
_scipystub.restore()
check("恢复后仍能正常跑", run(BASE).n_frames == 300)


print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for n in FAIL:
        print("  FAILED:", n)
    sys.exit(1)
