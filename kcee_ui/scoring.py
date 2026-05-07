"""Per-sequence scalar scores: cossim, eigenMaps, deviation-from-shared."""
import numpy as np


ENHANCER_LEN = 230
ADAPTER_LEN = 15  # constant cloning adapter at each end of the 230-bp insert
INSERT_START = ADAPTER_LEN
INSERT_STOP = ENHANCER_LEN - ADAPTER_LEN  # exclusive; variable region = 200 bp


def attr_to_importance(attr: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """(N, 4, L) attribution * (N, 4, L) onehot -> (N, L) importance.

    Matches eigen_steering.EigenMap.importance: for the WT base at each
    position, picks the attribution channel and sums (other channels are
    zeroed by the one-hot).
    """
    return (attr * onehot).sum(axis=1)


def _z_normalize_per_row(x: np.ndarray) -> np.ndarray:
    """Per-row z-norm: subtract mean, divide by std (replace zero-std with 1)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def cossim_score(imp_a: np.ndarray, imp_b: np.ndarray,
                 start: int = INSERT_START, stop: int = INSERT_STOP) -> np.ndarray:
    """Per-sequence cosine similarity on z-normalized importance over the
    variable insert region (skips the constant cloning adapters at each end)."""
    a = _z_normalize_per_row(imp_a[:, start:stop])
    b = _z_normalize_per_row(imp_b[:, start:stop])
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros_like(num, dtype=np.float32)
    nz = den > 0
    out[nz] = (num[nz] / den[nz]).astype(np.float32)
    return out


def eigenmaps_score(imp_a: np.ndarray, imp_b: np.ndarray,
                    start: int = INSERT_START, stop: int = INSERT_STOP) -> np.ndarray:
    """Per-sequence EigenMaps "EI_1 var x r" on z-normalized importance over
    the first `enhancer_len` positions. Closed-form 2x2 eigendecomposition.

    For each sequence i:
        E = z-normalized (L, 2) matrix [imp_a_i, imp_b_i]
        cov = E.T @ E / L                  # 2x2; on-diag ~1, off-diag = r
        eigenvalues of cov (closed form)
        var_ratio = lam0 / (lam0 + lam1)
        score = var_ratio * r              # r is cov[0,1] which equals
                                            # corrcoef(E[:,0], E[:,1])[0,1]
                                            # because columns are unit-var.
    """
    L = stop - start
    a = _z_normalize_per_row(imp_a[:, start:stop]).astype(np.float64)
    b = _z_normalize_per_row(imp_b[:, start:stop]).astype(np.float64)
    c00 = (a * a).sum(axis=1) / L
    c11 = (b * b).sum(axis=1) / L
    c01 = (a * b).sum(axis=1) / L
    tr = c00 + c11
    det = c00 * c11 - c01 * c01
    disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0.0))
    lam0 = 0.5 * (tr + disc)
    lam1 = 0.5 * (tr - disc)
    total = lam0 + lam1
    var_ratio = np.where(total > 0, lam0 / total, 0.0)
    return (var_ratio * c01).astype(np.float32)


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
    parallel_sq = (stacks.sum(axis=1) ** 2) / n_ct
    total_sq = (stacks ** 2).sum(axis=1)
    perp_sq = total_sq - parallel_sq
    num = perp_sq.sum(axis=1)
    den = total_sq.sum(axis=1)
    out = np.zeros_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out.astype(np.float32)
