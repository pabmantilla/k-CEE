"""Convert a .npz attribution file to an .h5 file with the same keys.

The kcee-ui viewer benefits from row-chunked HDF5 because it enables true
lazy single-row reads (see cache.load_attr_row), whereas .npz forces a full
decompression of the entire array on first touch.

Usage:
    uv run python tools/npz_to_h5.py \
        --input  /path/to/deeplift_attributions.npz \
        --output /path/to/deeplift_attributions.h5

Optional:
    --keys attr_HepG2,attr_K562,predictions_HepG2  (default: all keys)

Each dataset is written with:
    - same shape and dtype as the source ndarray
    - chunks=(1, *rest) so a single row read is one chunk
    - gzip compression at level 4 (good size/speed trade-off)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def _row_chunks(shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """chunks=(1, *rest) for >=1-D arrays; None for 0-D scalars."""
    if len(shape) == 0:
        return None
    return (1, *shape[1:])


def convert(input_path: Path, output_path: Path, keys: list[str] | None = None) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"input npz not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[npz_to_h5] loading {input_path}")
    npz = np.load(input_path, allow_pickle=False, mmap_mode="r")
    available = list(npz.files)
    print(f"[npz_to_h5] keys in npz ({len(available)}): {available}")

    selected = keys if keys else available
    missing = [k for k in selected if k not in available]
    if missing:
        raise KeyError(f"requested keys not in npz: {missing}")

    overall_t0 = time.time()
    with h5py.File(output_path, "w") as h5:
        for k in selected:
            t0 = time.time()
            try:
                arr = npz[k]
            except Exception as e:
                print(f"[npz_to_h5] SKIP {k!r}: failed to load ({e!r})")
                continue
            if not isinstance(arr, np.ndarray):
                print(f"[npz_to_h5] SKIP {k!r}: not an ndarray (got {type(arr).__name__})")
                continue
            if arr.dtype == object:
                print(f"[npz_to_h5] SKIP {k!r}: object dtype is not h5-friendly")
                continue

            chunks = _row_chunks(arr.shape)
            ds_kwargs: dict = {"data": arr, "dtype": arr.dtype}
            if chunks is not None:
                ds_kwargs["chunks"] = chunks
                ds_kwargs["compression"] = "gzip"
                ds_kwargs["compression_opts"] = 4
            try:
                h5.create_dataset(k, **ds_kwargs)
            except TypeError as e:
                print(f"[npz_to_h5] SKIP {k!r}: dtype not h5-friendly ({e})")
                continue
            dt = time.time() - t0
            print(
                f"[npz_to_h5] wrote {k!r}: shape={arr.shape} dtype={arr.dtype} "
                f"chunks={chunks} in {dt:.1f}s"
            )

    overall_dt = time.time() - overall_t0
    print(f"[npz_to_h5] done in {overall_dt:.1f}s -> {output_path}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert an .npz to a row-chunked .h5")
    p.add_argument("--input", required=True, type=Path, help="path to .npz input")
    p.add_argument("--output", required=True, type=Path, help="path to .h5 output")
    p.add_argument(
        "--keys",
        default=None,
        help="comma-separated subset of keys to convert (default: all)",
    )
    args = p.parse_args(argv)
    args.keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    convert(args.input, args.output, args.keys)
    return 0


if __name__ == "__main__":
    sys.exit(main())
