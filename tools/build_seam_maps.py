"""Pre-bake the SEAM (1059 seq × 3 type × 2 ct) attribution maps into a single
npz so the SEAM viewer cold-load is one `np.load` instead of ~8.4k tiny reads.

Output: SEAM_target_spaces/results/seam_maps_v1.npz with keys
    {tp}__{ct} : (1059, 4, 230) float32   for tp in {wt, foreground, background}, ct in {HepG2, K562}
    onehot     : (1059, 4, 230) float32
    seq_idxs   : (1059,) int64

Run:
    uv run python tools/build_seam_maps.py
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from kcee_ui.data import seq_to_onehot

SEAM_ROOT = Path(
    "/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/SEAM_target_spaces"
)
FG_DIR = SEAM_ROOT / "results" / "foregrounds"
LIB_PKL = SEAM_ROOT / "libraries" / "hippo_target_library.pkl"
OUT = SEAM_ROOT / "results" / "seam_maps_v1.npz"

CELL_TYPES = ("HepG2", "K562")
FILE_FOR = {"wt": "wt_attribution.npy", "foreground": "foreground_scaled.npy"}


def _intra_bg(seq_dir: Path) -> np.ndarray:
    ref_idx = int(np.load(seq_dir / "ref_cluster_idx.npy"))
    return np.load(seq_dir / "cluster_backgrounds.npy")[ref_idx].astype(np.float32)


def main() -> None:
    t0 = time.time()
    with open(LIB_PKL, "rb") as f:
        df = pickle.load(f)["df"].reset_index(drop=True)
    seq_idxs = np.asarray([int(x) for x in df["seq_idx"].values], dtype=np.int64)
    N = len(seq_idxs)
    print(f"library: {N} seqs")

    out: dict[str, np.ndarray] = {}
    for ct in CELL_TYPES:
        for tp in ("wt", "foreground"):
            arr = np.zeros((N, 4, 230), dtype=np.float32)
            for i, sid in enumerate(seq_idxs):
                arr[i] = np.load(FG_DIR / ct / str(sid) / FILE_FOR[tp]).T
            out[f"{tp}__{ct}"] = arr
            print(f"  {tp}__{ct}: {arr.shape} {arr.dtype}")
        arr = np.zeros((N, 4, 230), dtype=np.float32)
        for i, sid in enumerate(seq_idxs):
            arr[i] = _intra_bg(FG_DIR / ct / str(sid)).T
        out[f"background__{ct}"] = arr
        print(f"  background__{ct}: {arr.shape} {arr.dtype}")

    onehot = np.zeros((N, 4, 230), dtype=np.float32)
    for i, seq in enumerate(df["sequence"].astype(str).tolist()):
        onehot[i] = seq_to_onehot(seq, length=230, offset=0)
    out["onehot"] = onehot
    out["seq_idxs"] = seq_idxs

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **out)
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({size_mb:.1f} MB)  in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
