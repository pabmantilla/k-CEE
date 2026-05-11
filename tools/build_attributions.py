"""Build the canonical kcee-ui/data/attributions/ tree.

All output files share the same row set (LegNet's drop policy: 56,975 rows)
and the same shape (56975, 4, 200), so cross-model cossim/EigenMap can index
positionally without a per-source row map.

Layout:
    kcee-ui/data/attributions/
      manifest.csv                   # 56,975 rows of the library, in canonical order
      README.md
      koo_standardtorch/
        {ct}_{method}.h5             # /attr (56975,4,200), /predictions (56975,)
      pablo_ag_ft/
        {ct}_deeplift.h5
      legnet_ensemble/
        {ct}_deeplift.h5             # HepG2, K562 only

Run:
    uv run python tools/build_attributions.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
from kcee_ui.alignment import csv_to_npz_for_slot

DATA_REPO = Path("/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra")
CSV = DATA_REPO / "data/joint_library_combined.csv"
OUT_ROOT = _REPO / "data/attributions"

# (source_dir_name, source_file, attr_key, pred_key, method, cell_type)
SOURCES: list[tuple[str, Path, str, str, str, str]] = [
    # Koo standardtorch — DeepLIFT
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/deeplift_attributions_standardtorch_convfix.npz",
     "attr_HepG2", "predictions_HepG2", "deeplift", "HepG2"),
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/deeplift_attributions_standardtorch_convfix.npz",
     "attr_K562",  "predictions_K562",  "deeplift", "K562"),
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/deeplift_attributions_standardtorch_convfix.npz",
     "attr_WTC11", "predictions_WTC11", "deeplift", "WTC11"),
    # Koo standardtorch — Saliency
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/gradattr_standardtorch_convfix.npz",
     "saliency_HepG2", "predictions_HepG2", "saliency", "HepG2"),
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/gradattr_standardtorch_convfix.npz",
     "saliency_K562",  "predictions_K562",  "saliency", "K562"),
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/gradattr_standardtorch_convfix.npz",
     "saliency_WTC11", "predictions_WTC11", "saliency", "WTC11"),
    # Koo standardtorch — IntGrad
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/gradattr_standardtorch_convfix.npz",
     "intgrad_HepG2", "predictions_HepG2", "intgrad", "HepG2"),
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/gradattr_standardtorch_convfix.npz",
     "intgrad_K562",  "predictions_K562",  "intgrad", "K562"),
    ("koo_standardtorch", DATA_REPO / "genomic_targets/data/gradattr_standardtorch_convfix.npz",
     "intgrad_WTC11", "predictions_WTC11", "intgrad", "WTC11"),
    # Pablo AG-FT — DeepLIFT
    ("pablo_ag_ft", DATA_REPO / "genomic_targets/data/deeplift_attributions_uniform_convfix.npz",
     "attr_HepG2", "predictions_HepG2", "deeplift", "HepG2"),
    ("pablo_ag_ft", DATA_REPO / "genomic_targets/data/deeplift_attributions_uniform_convfix.npz",
     "attr_K562",  "predictions_K562",  "deeplift", "K562"),
    ("pablo_ag_ft", DATA_REPO / "genomic_targets/data/deeplift_attributions_uniform_convfix.npz",
     "attr_WTC11", "predictions_WTC11", "deeplift", "WTC11"),
    # LegNet ensemble — DeepLIFT (HepG2 + K562 only)
    ("legnet_ensemble", DATA_REPO / "legnet_rep/results/attrs_HepG2.h5",
     "attributions", "predictions", "deeplift", "HepG2"),
    ("legnet_ensemble", DATA_REPO / "legnet_rep/results/attrs_K562.h5",
     "attributions", "predictions", "deeplift", "K562"),
]


def _load_src(path: Path, key: str, pred_key: str) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path, mmap_mode="r") as d:
            return d[key][:], d[pred_key][:] if pred_key in d.files else np.array([])
    if path.suffix in (".h5", ".hdf5"):
        with h5py.File(path, "r") as f:
            attr = f[key][:]
            pred = f[pred_key][:] if pred_key in f else np.array([])
        return attr, pred
    raise ValueError(f"unsupported: {path}")


def _write_h5(out_path: Path, attr: np.ndarray, pred: np.ndarray, *,
              source_file: str, method: str, cell_type: str, n_csv_rows: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("attr", data=attr, dtype=attr.dtype,
                          chunks=(1, *attr.shape[1:]),
                          compression="gzip", compression_opts=4)
        if pred.size:
            h5.create_dataset("predictions", data=pred, dtype=pred.dtype)
        h5.attrs["method"] = method
        h5.attrs["cell_type"] = cell_type
        h5.attrs["provenance_path"] = source_file
        h5.attrs["n_csv_rows"] = n_csv_rows
        h5.attrs["shape"] = list(attr.shape)


def main() -> int:
    df = pd.read_csv(CSV)
    common = df.dropna(subset=["sequence", "HepG2_log2FC", "K562_log2FC"]).index.to_numpy()
    print(f"common CSV rows: N={len(common)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # manifest = the 56,975-row library subset, with original csv_row preserved
    manifest = df.loc[common].copy()
    manifest.insert(0, "csv_row", common)
    manifest.to_csv(OUT_ROOT / "manifest.csv", index=False)
    print(f"wrote manifest.csv ({len(manifest)} rows)")

    overall_t0 = time.time()
    for src_dir, src_path, attr_key, pred_key, method, ct in SOURCES:
        t0 = time.time()
        if not src_path.exists():
            print(f"  SKIP missing source: {src_path}")
            continue
        attr_src, pred_src = _load_src(src_path, attr_key, pred_key)
        n_attr = attr_src.shape[0]
        slot = {"name": f"{src_dir}/{ct}_{method}", "key": attr_key, "n_attr": n_attr}
        m = csv_to_npz_for_slot(slot, df)
        file_rows = m[common]
        assert (file_rows >= 0).all(), f"{slot['name']}: common rows not covered"
        attr_out = attr_src[file_rows].astype(np.float32, copy=False)
        pred_out = pred_src[file_rows].astype(np.float32, copy=False) if pred_src.size else np.array([])
        assert not np.isnan(attr_out).any(), f"{slot['name']}: NaN after reindex"
        out_path = OUT_ROOT / src_dir / f"{ct}_{method}.h5"
        _write_h5(out_path, attr_out, pred_out,
                  source_file=str(src_path), method=method, cell_type=ct, n_csv_rows=len(common))
        print(f"  wrote {out_path.relative_to(OUT_ROOT)}: {attr_out.shape} "
              f"policy={slot['_align_policy']} ({time.time()-t0:.1f}s)")

    print(f"done in {time.time()-overall_t0:.1f}s -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
