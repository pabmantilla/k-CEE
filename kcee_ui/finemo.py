"""Load finemo hits.tsv and group by sequence index (peak_id)."""
from pathlib import Path
import pandas as pd


def load_finemo_hits(tsv_path: str | Path) -> dict[int, list[dict]]:
    """Return {peak_id: [{start, end, motif, strand}, ...]}.

    Uses start_untrimmed / end_untrimmed (full motif span, before similarity trim).
    """
    df = pd.read_csv(tsv_path, sep="\t")
    out: dict[int, list[dict]] = {}
    cols_needed = {"start_untrimmed", "end_untrimmed", "motif_name", "strand", "peak_id"}
    if not cols_needed.issubset(df.columns):
        return out
    for pid, sub in df.groupby("peak_id"):
        out[int(pid)] = [
            {
                "start": int(r.start_untrimmed),
                "end": int(r.end_untrimmed),
                "motif": str(r.motif_name),
                "strand": str(r.strand),
            }
            for r in sub.itertuples(index=False)
        ]
    return out
