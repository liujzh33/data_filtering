"""从 parquet 里按列取出数值信号矩阵。sudden_change / state_extreme 共用。

单独成模块是为了避免旧版那种跨模块耦合: 旧 state_extreme_frame.py 里写着
`from .sudden_change import _is_derivative_channel` —— 一个检测去 import 另一个检测的
**启发式判据**, 改动其中一处会静默影响另一处。这里放的只是纯粹的读取管道, 不含任何
关于"哪些维度该检"的判断; 那件事一律由 profile 决定。
"""

import io
from typing import Any, Dict, Optional, Sequence

from .errors import PipelineEnvironmentError, UnsupportedDataError

__all__ = ["extract_signal_matrices"]


def extract_signal_matrices(parquet_bytes: Optional[bytes],
                            columns: Sequence[str]) -> Dict[str, Any]:
    """从 parquet 里取出指定的那些列, 返回 {列名: (T, D) 矩阵}。

    列名由 profile 显式给出, **不做任何猜测**。

    报错情况 (不降级):
      - pandas / numpy 不可用   -> PipelineEnvironmentError
      - parquet 为 None / 读不出 -> UnsupportedDataError
      - 声明的列不存在           -> UnsupportedDataError
      - 各列行数不一致           -> UnsupportedDataError (**不截断到最短**)
    """
    if parquet_bytes is None:
        raise UnsupportedDataError("未提供 parquet 字节, 取不到 state/action 信号")
    if not columns:
        raise UnsupportedDataError("未指定要读取的列")
    try:
        import numpy as np
        import pandas as pd
    except Exception as e:  # pragma: no cover - 环境问题
        raise PipelineEnvironmentError(f"numpy/pandas 不可用: {e}") from e

    try:
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
    except Exception as e:
        raise UnsupportedDataError(f"parquet 读取失败: {type(e).__name__}: {e}") from e

    have = {str(c) for c in df.columns}
    missing = [c for c in columns if c not in have]
    if missing:
        raise UnsupportedDataError(
            f"profile 指定的列 {missing} 不在 parquet 里 (实际列名: {sorted(have)})。"
            f"profile 与数据不匹配, 不猜其他列")

    out: Dict[str, Any] = {}
    for col in columns:
        s = df[col]
        m = np.asarray(np.stack(s.to_numpy()) if s.dtype == object else s.to_numpy(),
                       dtype=np.float64)
        out[col] = m.reshape(-1, 1) if m.ndim == 1 else m

    lengths = {c: int(m.shape[0]) for c, m in out.items()}
    if len(set(lengths.values())) > 1:
        raise UnsupportedDataError(
            f"各列行数不一致: {lengths}。不做截断对齐 —— 截短之后长度依然自洽, "
            f"但逐帧比对的已经是错位的信号, 事后无法发现")
    return out
