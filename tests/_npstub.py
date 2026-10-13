"""测试用的极简 numpy 替身 (只实现本仓库检测模块用到的那部分)。

这个环境里装不了 numpy, 但检测逻辑必须能验证。装到 sys.modules["numpy"] 上即可跑
measure_view / _state_static / check_* 的规则判定。**不是生产代码**, 只服务于
tests/。真正跑数据时用真 numpy。
"""

import builtins
import math
import sys
import types

nan = float("nan")


def _iterable(x):
    return isinstance(x, (list, tuple, A))


class A:
    """一维 (或二维: 元素为 list 的一维) 数组的最小实现。"""

    def __init__(self, data):
        self.data = list(data)

    # ---- 基本协议 ----
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, k):
        if isinstance(k, tuple):    # 二维: a[:, j] 取列 / a[i, j] 取元素
            r, c = k
            if isinstance(r, slice):
                return A([row[c] for row in self.data[r]])
            return self.data[r][c]
        if isinstance(k, slice):
            return A(self.data[k])
        if isinstance(k, A):        # 布尔索引
            return A([v for v, m in zip(self.data, k.data) if m])
        return self.data[k]

    def __setitem__(self, k, v):
        if isinstance(k, slice):
            idx = range(*k.indices(len(self.data)))
            vals = v.data if isinstance(v, A) else v
            if _iterable(vals):
                vals = list(vals)
                if len(vals) != len(idx):
                    raise ValueError(f"shape mismatch {len(vals)} vs {len(idx)}")
                for i, val in zip(idx, vals):
                    self.data[i] = val
            else:
                for i in idx:
                    self.data[i] = vals
        else:
            self.data[k] = v

    def __repr__(self):
        return f"A({self.data})"

    @property
    def shape(self):
        if self.data and _iterable(self.data[0]):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        s = self.shape
        return s[0] * (s[1] if len(s) > 1 else 1)

    # ---- 逐元素运算 ----
    def _cmp(self, other, op):
        if isinstance(other, A):
            return A([op(a, b) for a, b in zip(self.data, other.data)])
        return A([op(a, other) for a in self.data])

    def __lt__(self, o):
        return self._cmp(o, lambda a, b: (not _isnan(a)) and a < b)

    def __le__(self, o):
        return self._cmp(o, lambda a, b: (not _isnan(a)) and a <= b)

    def __gt__(self, o):
        return self._cmp(o, lambda a, b: (not _isnan(a)) and a > b)

    def __ge__(self, o):
        return self._cmp(o, lambda a, b: (not _isnan(a)) and a >= b)

    def __and__(self, o):
        return self._cmp(o, lambda a, b: bool(a) and bool(b))

    def __or__(self, o):
        return self._cmp(o, lambda a, b: bool(a) or bool(b))

    def __invert__(self):
        return A([not bool(a) for a in self.data])

    def __eq__(self, o):
        return self._cmp(o, lambda a, b: a == b)

    def __ne__(self, o):
        return self._cmp(o, lambda a, b: a != b)

    __hash__ = object.__hash__      # 定义 __eq__ 会把 __hash__ 置空, 显式补回

    def __sub__(self, o):
        return self._cmp(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._cmp(o, lambda a, b: b - a)

    def __add__(self, o):
        return self._cmp(o, lambda a, b: a + b)

    __radd__ = __add__

    def __mul__(self, o):
        return self._cmp(o, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._cmp(o, lambda a, b: a / b)

    def sum(self):
        return sum(1 if v is True else (0 if v is False else v) for v in self.data)

    def min(self):
        return min(self.data)

    def max(self):
        return max(self.data)

    def mean(self):
        return sum(self.data) / len(self.data)

    def tolist(self):
        return list(self.data)

    def any(self):
        return any(bool(v) for v in self.data)

    def all(self):
        return all(bool(v) for v in self.data)


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def asarray(x, dtype=None):
    if isinstance(x, A):
        return x
    return A(x)


array = asarray


def zeros(n, dtype=None):
    if isinstance(n, tuple):
        return A([[0] * n[1] for _ in range(n[0])])
    return A([False if dtype is bool else 0.0] * int(n))


def full(n, v, dtype=None):
    return A([v] * int(n))


def isnan(a):
    if isinstance(a, A):
        return A([_isnan(v) for v in a.data])
    return _isnan(a)


def _vals(a):
    return a.data if isinstance(a, A) else list(a)


def sign(a):
    return A([0.0 if _isnan(v) else ((v > 0) - (v < 0)) for v in _vals(a)])


def abs(a):                                  # noqa: A001 - 对齐 numpy 的名字
    return A([math.fabs(v) for v in _vals(a)])


def mean(a):
    d = _vals(a)
    return sum(d) / len(d) if d else nan


def std(a):
    d = [float(v) for v in _vals(a)]
    if not d:
        return nan
    m = sum(d) / len(d)
    return math.sqrt(sum((v - m) ** 2 for v in d) / len(d))


def cumsum(a):
    out, acc = [], 0.0
    for v in _vals(a):
        acc += v
        out.append(acc)
    return A(out)


def empty(n, dtype=None):
    return A([0.0] * int(n))


def corrcoef(x, y):
    """返回 2x2 相关矩阵, 供 corrcoef(x, y)[0, 1] 取用。"""
    xs, ys = [float(v) for v in _vals(x)], [float(v) for v in _vals(y)]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in xs))
    sy = math.sqrt(sum((v - my) ** 2 for v in ys))
    if sx == 0 or sy == 0:
        return A([[1.0, nan], [nan, 1.0]])
    r = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)
    return A([[1.0, r], [r, 1.0]])


def flatnonzero(a):
    return A([i for i, v in enumerate(a.data) if v])


def isfinite(a):
    def _f(v):
        return isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)
    if isinstance(a, A):
        return A([_f(v) for v in a.data])
    return _f(a)


def all(a):                                  # noqa: A001 - 对齐 numpy 的名字
    return builtins.all(bool(v) for v in _vals(a))


def any(a):                                  # noqa: A001 - 对齐 numpy 的名字
    return builtins.any(bool(v) for v in _vals(a))


def max(a):                                  # noqa: A001 - 对齐 numpy 的名字
    return builtins.max(_vals(a))


def min(a):                                  # noqa: A001 - 对齐 numpy 的名字
    return builtins.min(_vals(a))


def median(a):
    d = sorted(float(v) for v in _vals(a))
    if not d:
        return nan
    m = len(d) // 2
    return d[m] if len(d) % 2 else (d[m - 1] + d[m]) / 2.0


def _binary(x, y, op):
    xa, ya = isinstance(x, A), isinstance(y, A)
    if xa and ya:
        return A([op(a, b) for a, b in zip(x.data, y.data)])
    if xa:
        return A([op(a, y) for a in x.data])
    if ya:
        return A([op(x, b) for b in y.data])
    return op(x, y)


def maximum(x, y):
    return _binary(x, y, builtins.max)


def minimum(x, y):
    return _binary(x, y, builtins.min)


def percentile(a, q):
    d = sorted(a.data)
    if not d:
        return nan
    k = (len(d) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return d[lo] if lo == hi else d[lo] * (hi - k) + d[hi] * (k - lo)


def diff(a, axis=0):
    """只支持 axis=0。一维返回一维; 二维 (元素为 list) 返回二维。"""
    d = a.data
    if d and _iterable(d[0]):
        return A([[x - y for x, y in zip(d[i + 1], d[i])] for i in range(len(d) - 1)])
    return A([d[i + 1] - d[i] for i in range(len(d) - 1)])


def concatenate(blocks, axis=1):
    rows = len(blocks[0].data)
    return A([[v for b in blocks for v in b.data[i]] for i in range(rows)])


class _Linalg:
    @staticmethod
    def norm(a, axis=1):
        return A([math.sqrt(sum(v * v for v in row)) for row in a.data])


linalg = _Linalg()
float32 = "float32"
float64 = "float64"


def install():
    """把这个替身注册成 numpy, 返回模块对象。"""
    m = types.ModuleType("numpy")
    for k, v in globals().items():
        if not k.startswith("_"):
            setattr(m, k, v)
    m.ndarray = A
    sys.modules["numpy"] = m
    return m


def install_if_missing():
    """有真 numpy 就用真的, 没有才装替身。返回 True 表示装了替身。

    替身只是"没有 numpy 也能验规则"的兜底, 不是首选。真 numpy 在的时候用真的:
    广播、dtype、边界行为都跟生产环境一致, 能测出替身测不出的问题。
    """
    try:
        import numpy  # noqa: F401
    except Exception:
        install()
        return True
    return False
