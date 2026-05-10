"""Library CSV + WT one-hot helpers."""
from pathlib import Path
import numpy as np
import pandas as pd


BASES = "ACGT"
_BASE_LUT = np.full(128, -1, dtype=np.int8)
for _i, _b in enumerate(BASES):
    _BASE_LUT[ord(_b)] = _i


def load_library(path: str | Path) -> pd.DataFrame:
    """Load joint_library_combined.csv (or any csv with name/sequence/<CT>_log2FC)."""
    return pd.read_csv(path)


def seq_to_onehot(seq: str, length: int | None = None, offset: int = 0) -> np.ndarray:
    """ACGT one-hot, channels-first (4, L). Non-ACGT -> zero column.

    If `length` is given, the output has `length` columns. `offset` selects
    which sequence position lands at output column 0 (i.e. the output covers
    seq[offset : offset + length]). Out-of-range columns are zero.
    """
    if not isinstance(seq, str):
        seq = ""
    L = len(seq)
    out_L = length if length is not None else max(0, L - offset)
    arr = np.zeros((4, out_L), dtype=np.float32)
    if L == 0 or out_L == 0:
        return arr
    src_lo = max(0, offset)
    src_hi = min(L, offset + out_L)
    if src_hi <= src_lo:
        return arr
    dst_lo = src_lo - offset
    sub = seq[src_lo:src_hi]
    idx = np.frombuffer(sub.encode("ascii"), dtype=np.uint8)
    j = _BASE_LUT[idx]
    valid = j >= 0
    cols = np.nonzero(valid)[0]
    arr[j[cols], cols + dst_lo] = 1.0
    return arr
