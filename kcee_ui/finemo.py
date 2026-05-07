"""Load finemo hits.tsv and group by sequence index (peak_id).

Canonical convention from `genomic_targets/scripts/2d_diff_call/scripts/notebooks/ctcf_focus.ipynb`:
    h['x0'] = h['start'] - peak_start    # peak_start = df['start_hg38'].iloc[seq_i]
    h['x1'] = h['end']   - peak_start
i.e. use the trimmed (start, end), genome coords, simple subtraction. No strand mirror.
"""
import pickle
from pathlib import Path
import pandas as pd

from kcee_ui.cache import cache_dir, _hash, _mtime


def load_finemo_hits(tsv_path: str | Path) -> dict[int, list[dict]]:
    """Return {peak_id: [{start, end, motif, strand}, ...]}, in genomic coords."""
    tsv_path = Path(tsv_path)
    src = str(tsv_path.resolve())
    h = _hash(src, str(_mtime(src)))
    cache_path = cache_dir() / "finemo" / f"{h}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError):
            pass
    df = pd.read_csv(tsv_path, sep="\t",
                     usecols=["motif_name", "peak_id", "start", "end", "strand"],
                     dtype={"motif_name": str, "strand": str},
                     low_memory=False)
    out: dict[int, list[dict]] = {}
    for pid, sub in df.groupby("peak_id"):
        out[int(pid)] = [
            {"start": int(r.start), "end": int(r.end),
             "motif": str(r.motif_name), "strand": str(r.strand)}
            for r in sub.itertuples(index=False)
        ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(cache_path)
    return out
