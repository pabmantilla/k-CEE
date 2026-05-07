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


def seq_to_onehot(seq: str, length: int | None = None) -> np.ndarray:
    """ACGT one-hot, channels-first (4, L). Non-ACGT -> zero column.

    If `length` is given, pad/truncate to that length (zero-pad on the right).
    """
    if not isinstance(seq, str):
        seq = ""
    L = len(seq)
    out_L = length if length is not None else L
    arr = np.zeros((4, out_L), dtype=np.float32)
    if L == 0:
        return arr
    idx = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    j = _BASE_LUT[idx]
    valid = j >= 0
    n = min(L, out_L)
    rows = np.nonzero(valid[:n])[0]
    arr[j[rows], rows] = 1.0
    return arr
