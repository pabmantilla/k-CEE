"""Persistent on-disk cache for attribution arrays and derived scores.

Strategy:
- Compressed `.npz` arrays are extracted once to `.npy` files in the cache,
  then opened with `mmap_mode='r'` so subsequent access is OS-page-cached
  and per-row reads are ~free.
- Derived arrays (scores, csv->id maps) are stored as `.npy` keyed by a hash
  of their inputs (path, key, mtime, ...).

Cache location: $KCEE_CACHE_DIR or `~/.cache/kcee-ui/`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import numpy as np


def cache_dir() -> Path:
    p = Path(os.environ.get("KCEE_CACHE_DIR", Path.home() / ".cache" / "kcee-ui"))
    (p / "arrays").mkdir(parents=True, exist_ok=True)
    (p / "scores").mkdir(parents=True, exist_ok=True)
    return p


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _mtime(path: str | Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def mmap_array(npz_path: str | Path, key: str) -> np.ndarray:
    """Return a memory-mapped view of an array from a compressed `.npz`.

    First call: extract the array to a `.npy` in the cache and mmap it.
    Subsequent calls (across sessions): just mmap the existing `.npy`.
    Cache is invalidated by source mtime.
    """
    npz_path = str(Path(npz_path).resolve())
    suffix = Path(npz_path).suffix.lower()
    h = _hash(npz_path, key, str(_mtime(npz_path)))
    npy_path = cache_dir() / "arrays" / f"{h}.npy"
    if not npy_path.exists():
        if suffix == ".npz":
            with np.load(npz_path) as d:
                arr = np.asarray(d[key])
        elif suffix in (".h5", ".hdf5"):
            import h5py
            with h5py.File(npz_path, "r") as f:
                arr = np.asarray(f[key][:])
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        # np.save appends .npy automatically; build the tmp path with that in mind.
        tmp_stem = npy_path.with_name(npy_path.stem + "_tmp")  # no suffix
        np.save(tmp_stem, arr)  # writes <tmp_stem>.npy
        Path(str(tmp_stem) + ".npy").rename(npy_path)
    return np.load(npy_path, mmap_mode="r")


def cached_npy(name: str, deps: tuple, compute_fn) -> np.ndarray:
    """Disk-cached compute. `deps` are stringified into the cache key."""
    h = _hash(name, *(str(d) for d in deps))
    out_path = cache_dir() / "scores" / f"{name}_{h}.npy"
    if out_path.exists():
        return np.load(out_path)
    arr = compute_fn()
    tmp_stem = out_path.with_name(out_path.stem + "_tmp")
    np.save(tmp_stem, np.asarray(arr))
    Path(str(tmp_stem) + ".npy").rename(out_path)
    return np.load(out_path)


def cache_size_mb() -> float:
    p = cache_dir()
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def clear_cache() -> int:
    p = cache_dir()
    n = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n
