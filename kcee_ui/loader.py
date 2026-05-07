from pathlib import Path
import numpy as np
import h5py


def list_attr_keys(path: str | Path) -> list[str]:
    """Return candidate attribution-array keys in an .npz or .h5 file."""
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p) as d:
            return [k for k in d.keys() if d[k].ndim == 3]
    if p.suffix in (".h5", ".hdf5"):
        keys: list[str] = []
        with h5py.File(p, "r") as f:
            f.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) and obj.ndim == 3 else None)
        return keys
    raise ValueError(f"Unsupported file type: {p.suffix}")


def load_attr_file(path: str | Path, key: str) -> np.ndarray:
    """Load attribution array of shape (N, 4, L) from .npz or .h5."""
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p) as d:
            arr = d[key]
    elif p.suffix in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            arr = f[key][:]
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    if arr.ndim != 3 or arr.shape[1] != 4:
        raise ValueError(f"Expected (N, 4, L) array, got {arr.shape}")
    return arr.astype(np.float32, copy=False)
