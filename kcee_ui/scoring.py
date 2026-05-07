"""Per-sequence scalar scores: cossim, eigenMaps, deviation-from-shared."""
import pickle
from pathlib import Path
import numpy as np


def cossim_score(attr_a: np.ndarray, attr_b: np.ndarray) -> np.ndarray:
    """Per-sequence cosine similarity between two attribution stacks of shape (N, 4, L)."""
    a = attr_a.reshape(attr_a.shape[0], -1)
    b = attr_b.reshape(attr_b.shape[0], -1)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def eigenmaps_score(pkl_path: str | Path, key: str = "EI_1 var x r") -> np.ndarray:
    """Load a per-sequence eigenMaps scalar from an eigen_analysis.pkl.

    Canonical formula (per genomic_targets/scripts/2d_targeting/hippo_target_selection.ipynb
    and 2d_diff_call/scripts/notebooks/eixr_distribution.ipynb):
        EI_1 var x r = ei1_var * corrs
    """
    with open(pkl_path, "rb") as f:
        cached = pickle.load(f)
    if isinstance(cached, dict) and key in cached:
        return np.asarray(cached[key], dtype=np.float32)
    if "ei1_var" in cached and "corrs" in cached:
        return np.asarray(cached["ei1_var"], dtype=np.float32) * np.asarray(cached["corrs"], dtype=np.float32)
    raise KeyError(f"Could not find '{key}' or (ei1_var, corrs) in {pkl_path}")


def deviation_from_shared(attr_list: list[np.ndarray]) -> np.ndarray:
    """Per-sequence deviation from the equiangular ray across cell types.

    For each (4, L) element across `n_ct` attribution stacks, decompose the
    `n_ct`-vector v into a parallel component (along (1,1,...,1)/sqrt(n_ct))
    and perpendicular component. Returns the per-sequence fraction of
    squared L2 energy that is perpendicular.

    Result range: [0, 1]. 0 = perfectly shared, 1 = orthogonal to shared.
    Works with any n_ct >= 2.
    """
    if len(attr_list) < 2:
        raise ValueError("Need at least 2 attribution stacks.")
    n = min(a.shape[0] for a in attr_list)
    stacks = np.stack([a[:n].reshape(n, -1) for a in attr_list], axis=1)  # (n, n_ct, F)
    n_ct = stacks.shape[1]
    parallel_sq = (stacks.sum(axis=1) ** 2) / n_ct  # ((sum v_i)^2)/n_ct == |proj|^2 per element
    total_sq = (stacks ** 2).sum(axis=1)
    perp_sq = total_sq - parallel_sq
    num = perp_sq.sum(axis=1)
    den = total_sq.sum(axis=1)
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out.astype(np.float32)
