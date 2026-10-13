"""saa (state-action 趋势对齐) 的规则回归测试。

numpy / scipy 有真的就用真的, 没有才退回替身 (_npstub / _scipystub)。替身里的
savgol_filter 是恒等映射, 等于关掉了平滑 —— 用例验证的是规则逻辑而不是滤波保真度,
说明见 _scipystub.py。

用法: python3 -m check_pipeline.tests.test_saa
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from check_pipeline.tests import _npstub, _scipystub  # noqa: E402
print(f"numpy: {'替身 (_npstub)' if _npstub.install_if_missing() else '真库'}  "
      f"scipy: {'替身 (恒等 savgol, 平滑未生效)' if _scipystub.install_if_missing() else '真库'}")

from check_pipeline import saa  # noqa: E402
from check_pipeline.errors import PipelineEnvironmentError, UnsupportedDataError  # noqa: E402

PASS, FAIL = [], []


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
# 构造信号
# ---------------------------------------------------------------------
T = 300
FPS = 30.0


def traj(seed=0, noise=0.0):
    """一条平滑的关节角轨迹 (绝对位置)。"""
    rnd = random.Random(seed)
    pos, v = [0.0], 0.0
    for t in range(1, T):
        v = 0.002 * math.sin(t / 40.0)
        pos.append(pos[-1] + v + (rnd.gauss(0, noise) if noise else 0.0))
    return pos


def lead(sig, k):
    """把信号整体前移 k 帧 = action 领先 k 帧。k<0 则是 state 领先 (因果异常)。"""
    if k >= 0:
        return sig[k:] + [sig[-1]] * k
    return [sig[0]] * (-k) + sig[:k]


def to_delta(sig):
    return [sig[i + 1] - sig[i] for i in range(len(sig) - 1)] + [0.0]


def mat(*cols):
    """按列组装成 (T, D)。"""
    return [[c[t] for c in cols] for t in range(len(cols[0]))]


PROF = {"enabled": True, "action_space": "absolute", "da_threshold": 0.70,
        "shared_dims": [{"name": "j0", "state_dim": 0, "action_dim": 0}]}


def prof(**kw):
    p = dict(PROF)
    p.update(kw)
    return p


# =====================================================================
print("[saa] 正常数据: action 领先, DA 高 -> 不删")
# =====================================================================
s = traj(1)
r = saa.check_saa(mat(s), mat(lead(s, 2)), prof(), FPS)
check("绝对 action 领先 2 帧 -> 不删", not r.drop_episode, f"da={r.dims['j0'].da}")
check("DA 接近 1", r.dims["j0"].da > 0.95, f"{r.dims['j0'].da}")
check("估到的 lag 为正 (action 领先)", r.dims["j0"].best_lag_frames > 0,
      f"{r.dims['j0'].best_lag_frames}")
check("lag 同时以秒给出", abs(r.dims["j0"].lag_sec - r.dims["j0"].lag_frames / FPS) < 1e-9)
check("action_space 实测与声明一致", r.action_space_measured == "absolute")
check("零 findings", r.findings == [])

r0 = saa.check_saa(mat(s), mat(s), prof(), FPS)
check("action 与 state 同步 (lag=0) -> 不删", not r0.drop_episode)

# =====================================================================
print("\n[saa] DA 不达标 -> 删整条")
# =====================================================================
rnd = random.Random(9)
noise_sig = [rnd.gauss(0, 1) for _ in range(T)]     # 与 state 完全无关的 action
r = saa.check_saa(mat(s), mat(noise_sig), prof(), FPS)
check("action 与 state 无关 -> 删整条", r.drop_episode, f"da={r.dims['j0'].da}")
check("规则码是 SAA_DA_BELOW_THRESHOLD",
      "SAA_DA_BELOW_THRESHOLD" in r.triggered_rules, f"{r.triggered_rules}")
f = [x for x in r.findings if x.rule == "SAA_DA_BELOW_THRESHOLD"][0]
check("finding 点名是哪个关节维度", f.dims == ["j0"], f"{f.dims}")
check("evidence 给出该维度的 DA", "da" in f.evidence["per_dim_failed"]["j0"])
check("evidence 给出 state/action 的下标",
      f.evidence["per_dim_failed"]["j0"]["state_dim"] == 0
      and f.evidence["per_dim_failed"]["j0"]["action_dim"] == 0)
check("evidence 写明判据", f.evidence["criterion"] == "min(DA over shared dims) >= da_threshold")
check("disposition 是 drop_episode", f.disposition == "drop_episode")

print("\n[saa] action 信号本身是垃圾时: 两条规则都会触发, 但 cross_corr 会说明真相")
# action 与 state 无关 -> lag 是在拟合噪声, best_lag 的具体值没有意义。
# episode 反正要删, 但报告里必须能看出"state 领先 N 帧"这个说法不成立,
# 否则会被误诊成时间戳错位。
check("同时触发因果与 DA 两条规则",
      set(r.triggered_rules) == {"SAA_CAUSALITY_VIOLATED", "SAA_DA_BELOW_THRESHOLD"},
      f"{r.triggered_rules}")
check("cross_corr 接近 0, 说明 lag 不可信", abs(r.dims["j0"].cross_corr) < 0.1,
      f"{r.dims['j0'].cross_corr}")
fc = [x for x in r.findings if x.rule == "SAA_CAUSALITY_VIOLATED"][0]
check("因果 evidence 把 cross_corr 和 best_lag 并排给出",
      "per_dim_cross_corr" in fc.evidence and "per_dim_best_lag" in fc.evidence)

print("\n[saa] 没超标的维度也要列出来 (分辨'单维特别差'和'整体在阈值附近晃')")
prof2 = prof(shared_dims=[{"name": "good", "state_dim": 0, "action_dim": 0},
                          {"name": "bad", "state_dim": 1, "action_dim": 1}])
s2 = traj(3)
r = saa.check_saa(mat(s, s2), mat(lead(s, 2), noise_sig), prof2, FPS)
check("一好一坏 -> 删整条", r.drop_episode)
f = [x for x in r.findings if x.rule == "SAA_DA_BELOW_THRESHOLD"][0]
check("只点名坏的那一维", f.dims == ["bad"], f"{f.dims}")
check("per_dim_da 两维都在", set(f.evidence["per_dim_da"]) == {"good", "bad"})
check("好的那维 DA 高", f.evidence["per_dim_da"]["good"] > 0.95)
check("任一维失败即删 (不看失败比例)", f.evidence["n_dims_failed"] == 1
      and f.evidence["n_dims_checked"] == 2)

# =====================================================================
print("\n[saa] 因果方向: state 领先 action = 时间戳错位 (旧版会把它对齐掉)")
# =====================================================================
r = saa.check_saa(mat(s), mat(lead(s, -3)), prof(), FPS)
check("state 领先 3 帧 -> 删整条", r.drop_episode, f"lag={r.dims['j0'].best_lag_frames}")
check("触发 SAA_CAUSALITY_VIOLATED",
      "SAA_CAUSALITY_VIOLATED" in r.triggered_rules, f"{r.triggered_rules}")
check("记下的 best_lag 是负的", r.dims["j0"].best_lag_frames < 0,
      f"{r.dims['j0'].best_lag_frames}")
check("算 DA 时不使用负 lag 去'修好'因果", r.dims["j0"].lag_frames == 0,
      f"lag_frames={r.dims['j0'].lag_frames}")
f = [x for x in r.findings if x.rule == "SAA_CAUSALITY_VIOLATED"][0]
check("因果 finding 点名维度", f.dims == ["j0"])
check("evidence 列出各维 best_lag", "per_dim_best_lag" in f.evidence)

r = saa.check_saa(mat(s), mat(lead(s, -1)), prof(), FPS)
check("state 只领先 1 帧 -> 在容差内, 不报因果",
      "SAA_CAUSALITY_VIOLATED" not in r.triggered_rules,
      f"lag={r.dims['j0'].best_lag_frames}")

# =====================================================================
print("\n[saa] delta action: 不积分必然失败, 积分后恢复")
# =====================================================================
d = to_delta(lead(s, 2))
raises("delta 数据却声明 absolute -> 报错",
       lambda: saa.check_saa(mat(s), mat(d), prof(action_space="absolute"), FPS),
       "但实测像 delta")
r = saa.check_saa(mat(s), mat(d), prof(action_space="delta"), FPS)
check("声明 delta 并积分后 -> 不删", not r.drop_episode, f"da={r.dims['j0'].da}")
check("delta 积分后 DA 接近 1", r.dims["j0"].da > 0.95, f"{r.dims['j0'].da}")
check("实测判为 delta", r.action_space_measured == "delta")
check("两种假设的一致率都记进 evidence",
      {"da_as_absolute", "da_as_delta", "std_ratio"} <= set(r.action_space_evidence))
check("delta 的 std_ratio 远小于 1", r.action_space_evidence["std_ratio"] < 0.1,
      f"{r.action_space_evidence['std_ratio']}")

raises("absolute 数据却声明 delta -> 报错",
       lambda: saa.check_saa(mat(s), mat(lead(s, 2)), prof(action_space="delta"), FPS),
       "但实测像 absolute")

# =====================================================================
print("\n[saa] 阈值")
# =====================================================================
check("默认 da_threshold = 0.70", saa.DEFAULT_PARAMS["da_threshold"] == 0.70)
r = saa.check_saa(mat(s), mat(noise_sig), prof(da_threshold=0.05), FPS)
check("profile 可以覆盖 da_threshold (DA 规则不再触发)",
      "SAA_DA_BELOW_THRESHOLD" not in r.triggered_rules,
      f"da={r.dims['j0'].da} thr=0.05 rules={r.triggered_rules}")
check("生效的阈值记进 params", r.params["da_threshold"] == 0.05)
raises("da_threshold 越界 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s), prof(da_threshold=1.5), FPS),
       "必须落在 (0, 1]")

print("\n[saa] lag / 平滑窗口按秒给, 跨帧率含义一致")
check("max_lag_sec 是秒", saa.DEFAULT_PARAMS["max_lag_sec"] == 0.20)
check("smooth_window_sec 是秒", saa.DEFAULT_PARAMS["smooth_window_sec"] == 0.17)
check("没有 min_std 参数 (静止维度由 profile 解决)",
      "min_std" not in saa.DEFAULT_PARAMS)

# =====================================================================
print("\n[saa] profile 校验: 不猜, 不降级")
# =====================================================================
raises("没有 profile -> 报错", lambda: saa.check_saa(mat(s), mat(s), {}, FPS),
       "维度配对必须由 profile 显式给出")
raises("profile 标记不做 SAA -> 报错 (由调用方记成跳过)",
       lambda: saa.check_saa(mat(s), mat(s), prof(enabled=False, reason="无共享关节"), FPS),
       "不能当成通过")
raises("shared_dims 为空 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s), prof(shared_dims=[]), FPS),
       "没有可检测的共享关节维度")
raises("action_space 缺失 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s), prof(action_space=""), FPS),
       "必须是 'absolute' 或 'delta'")
raises("state_dim 越界 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s),
                             prof(shared_dims=[{"name": "x", "state_dim": 7, "action_dim": 0}]),
                             FPS),
       "超出实际 state 维度")
raises("action_dim 越界 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s),
                             prof(shared_dims=[{"name": "x", "state_dim": 0, "action_dim": 9}]),
                             FPS),
       "超出实际 action 维度")
raises("shared_dims 缺字段 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s), prof(shared_dims=[{"name": "x"}]), FPS),
       "缺少 state_dim / action_dim")
raises("维度名重复 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s),
                             prof(shared_dims=[{"name": "x", "state_dim": 0, "action_dim": 0},
                                               {"name": "x", "state_dim": 0, "action_dim": 0}]),
                             FPS),
       "维度名重复")
raises("未知参数 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s), prof(), FPS, params={"nope": 1}),
       "未知参数")

print("\n[saa] 数据校验: 不静默截断")
raises("state 与 action 行数不一致 -> 报错",
       lambda: saa.check_saa(mat(s), mat(s[:-10]), prof(), FPS),
       "不做截断对齐")
raises("拿不到帧率 -> 报错", lambda: saa.check_saa(mat(s), mat(s), prof(), 0.0),
       "拿不到有效帧率")
raises("episode 太短 -> 报错",
       lambda: saa.check_saa(mat(s[:8]), mat(s[:8]), prof(), FPS),
       "不足以做平滑与 lag 搜索")

# =====================================================================
print("\n[saa] 死区诊断 (只测量, 不参与判定)")
# =====================================================================
r = saa.check_saa(mat(s), mat(lead(s, 2)), prof(), FPS)
dd = r.dims["j0"].da_at_deadband
check("三个死区档位都输出了", set(dd) == {"0", "p25", "p50"}, f"{sorted(dd)}")
check("判定用的是无死区那一档", abs(dd["0"] - r.dims["j0"].da) < 1e-12)
check("死区档位不改变判定", not r.drop_episode)

# =====================================================================
print("\n[saa] 输出结构")
# =====================================================================
import json  # noqa: E402
r = saa.check_saa(mat(s, s2), mat(lead(s, 2), noise_sig), prof2, FPS)
d = r.to_dict()
try:
    check("to_dict() 能 json 序列化", len(json.dumps(d, ensure_ascii=False)) > 0)
except Exception as e:
    check("to_dict() 能 json 序列化", False, str(e))
check("顶层有 triggered_rules (bad 维是噪声, 因果规则也会命中)",
      set(d["triggered_rules"]) == {"SAA_CAUSALITY_VIOLATED", "SAA_DA_BELOW_THRESHOLD"},
      f"{d['triggered_rules']}")
check("顶层有 drop_episode", d["drop_episode"] is True)
check("measurements 里两维都在", set(d["measurements"]["per_dim"]) == {"good", "bad"})
check("measurements 记录实测 action_space",
      d["measurements"]["action_space_measured"] == "absolute")
check("measurements 记录 min_da", d["measurements"]["min_da"] == round(r.min_da, 6))
check("disposition 全是 drop_episode",
      all(x.disposition == "drop_episode" for x in r.findings))
check("没有 drop_frames 字段", "drop_frames" not in d)
check("每条规则码都在 RULES 表里", all(x.rule in saa.RULES for x in r.findings))

# =====================================================================
print("\n[saa] scipy 缺失必须报环境错, 不能静默用未平滑数据")
# =====================================================================
_scipystub.uninstall()
raises("scipy 不可用 -> PipelineEnvironmentError",
       lambda: saa.check_saa(mat(s), mat(lead(s, 2)), prof(), FPS),
       "不能退化成用未平滑数据", exc=PipelineEnvironmentError)
_scipystub.restore()

print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for n in FAIL:
        print("  FAILED:", n)
    sys.exit(1)
