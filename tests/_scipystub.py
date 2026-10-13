"""测试用的极简 scipy 替身, 只提供 signal.savgol_filter。

**savgol_filter 在这里是恒等映射, 不做任何平滑。**

这是刻意的: saa 的回归测试验证的是**规则逻辑** —— lag 搜索、因果方向判定、DA 统计、
delta 积分、profile 校验 —— 而不是滤波器本身的保真度。测试信号都是构造出来的
(噪声可控或无噪声), 平滑与否不改变这些逻辑的正确性, 恒等映射反而让期望值更好推算。

代价是"平滑到底有没有起作用"这件事测不到, 需要在有真 scipy 的环境里跑一遍。
另外 saa._savgol 在 scipy 缺失时必须抛 PipelineEnvironmentError —— 那条路径靠
**不安装本替身**来验证, 见 test_saa.py。

不是生产代码, 只服务于 tests/。
"""

import sys
import types


def savgol_filter(x, window_length, polyorder, **kw):
    return x


def medfilt(x, kernel_size=3):
    """中值滤波。这个是**真实现**, 不是恒等映射。

    中值就是中值, 纯 Python 算出来和 scipy 完全一致 (零填充边界也照搬), 所以这里
    不存在 savgol_filter 那种"替身把算法关掉了"的问题。
    """
    import statistics

    import numpy as np                            # 此时已是真 numpy 或 _npstub

    w = int(kernel_size)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    vals = [float(v) for v in x]
    xp = [0.0] * pad + vals + [0.0] * pad          # scipy.medfilt 是零填充
    return np.asarray([statistics.median(xp[i:i + w]) for i in range(len(vals))])


def install():
    """把这个替身注册成 scipy.signal, 返回 scipy 模块对象。"""
    sig = types.ModuleType("scipy.signal")
    sig.savgol_filter = savgol_filter
    sig.medfilt = medfilt
    sp = types.ModuleType("scipy")
    sp.signal = sig
    sys.modules["scipy"] = sp
    sys.modules["scipy.signal"] = sig
    return sp


def install_if_missing():
    """有真 scipy 就用真的, 没有才装替身。返回 True 表示装了替身。

    这个替身的 savgol_filter 是恒等映射, 装上它等于**关掉了平滑**。所以真 scipy
    在的时候必须用真的 —— 否则一切与滤波响应有关的行为 (比如 sudden_change 的
    响应倍数标定) 都测不到。
    """
    try:
        import scipy.signal  # noqa: F401
    except Exception:
        install()
        return True
    return False


def uninstall():
    """卸掉替身并**阻断** scipy 导入, 用于验证 scipy 缺失时是否正确报错。

    只从 sys.modules 里删掉是不够的: 环境里装了真 scipy 时会被重新导入成功, 这条
    用例就悄悄变成"什么都没验证"。往 sys.modules 里塞 None 可以让 import 确定失败,
    两种环境下行为一致。
    """
    for name in ("scipy.signal", "scipy"):
        sys.modules[name] = None


def restore():
    """解除阻断: 有真 scipy 就用真的, 没有就装回替身。"""
    for name in ("scipy.signal", "scipy"):
        sys.modules.pop(name, None)
    return install_if_missing()
