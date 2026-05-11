"""Per-slot csv_row <-> attr_row alignment with explicit drop policies + guard.

Different attribution files index their rows differently:
  - Koo standardtorch (n=56980): identity over CSV, rows 18321/18322 are NaN.
  - Pablo K/H        (n=56978): dropna(sequence)  -> drops [18321, 18322].
  - Pablo WTC11      (n=56980): identity, BUT rows 18321/18322 are garbage
                                (model evaluated on invalid input). Treat as
                                identity here; caller must mask those rows.
  - LegNet           (n=56975): dropna(sequence, HepG2_log2FC, K562_log2FC) ->
                                drops [18321, 18322, 41187, 54802, 55855].

The previous fallback `out[:m] = np.arange(m)` silently misaligned LegNet from
row 18321 onwards. `csv_to_npz_for_slot` raises instead; `assert_pair_aligned`
catches downstream consumers that try to compare two maps that disagree about N.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _candidate_policies(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Return (name, kept_csv_rows) for every known drop policy.

    A slot's drop policy is inferred by matching n_attr to one of these.
    """
    n = len(df)
    out: list[tuple[str, np.ndarray]] = [("identity", np.arange(n, dtype=np.int64))]
    if "sequence" in df.columns:
        out.append(("dropna_seq",
                    df.index[df["sequence"].notna()].to_numpy(dtype=np.int64)))
    log2fc_cols = [c for c in ("HepG2_log2FC", "K562_log2FC") if c in df.columns]
    if "sequence" in df.columns and log2fc_cols:
        keep = df.dropna(subset=["sequence", *log2fc_cols]).index.to_numpy(dtype=np.int64)
        out.append(("dropna_seq_log2fc_" + "_".join(log2fc_cols), keep))
    return out


def csv_to_npz_for_slot(slot: dict, df: pd.DataFrame, *, strict: bool = True) -> np.ndarray:
    """Build csv_row -> attr_row mapping (length len(df), -1 where slot doesn't cover).

    Picks the drop policy whose kept-row count matches `slot["n_attr"]`. Raises
    AlignmentError when no policy matches (this is the silent-misalignment trap;
    we never silently use a "first m rows" fallback).

    Set `strict=False` to fall back to a *length-checked* identity (only valid
    when n_attr == n_csv); still raises for any other mismatch.
    """
    n_attr = int(slot.get("n_attr") or 0)
    n_csv = len(df)
    out = np.full(n_csv, -1, dtype=np.int64)
    if n_attr == 0:
        return out

    bad_seq = (df["sequence"].isna().to_numpy()
               if "sequence" in df.columns else np.zeros(n_csv, dtype=bool))

    for name, kept in _candidate_policies(df):
        if len(kept) == n_attr:
            out[kept] = np.arange(n_attr, dtype=np.int64)
            n_masked = int(bad_seq[out >= 0].sum())
            out[bad_seq] = -1
            slot["_align_policy"] = name
            slot["_n_seq_masked"] = n_masked
            return out

    if not strict and n_attr == n_csv:
        out[:] = np.arange(n_csv, dtype=np.int64)
        n_masked = int(bad_seq.sum())
        out[bad_seq] = -1
        slot["_align_policy"] = "identity"
        slot["_n_seq_masked"] = n_masked
        return out

    raise AlignmentError(
        f"slot {slot.get('name') or slot.get('key')!r}: n_attr={n_attr} matches no known "
        f"drop policy against CSV n={n_csv}. Tried "
        f"{[(name, len(k)) for name, k in _candidate_policies(df)]}. "
        f"Add the policy in alignment.py or fix the upstream file."
    )


def assert_slot_aligned(slot: dict, df: pd.DataFrame, csv_to_npz: np.ndarray) -> None:
    """Sanity-check that `csv_to_npz` was built from a known policy and matches n_attr."""
    n_attr = int(slot.get("n_attr") or 0)
    if csv_to_npz.shape != (len(df),):
        raise AlignmentError(
            f"slot {slot.get('name')!r}: csv_to_npz shape {csv_to_npz.shape} != ({len(df)},)"
        )
    n_present = int((csv_to_npz >= 0).sum())
    n_masked = int(slot.get("_n_seq_masked", 0))
    if n_present + n_masked != n_attr:
        raise AlignmentError(
            f"slot {slot.get('name')!r}: csv_to_npz covers {n_present} rows "
            f"(+ {n_masked} NaN-seq masked) but n_attr={n_attr}"
        )
    if "_align_policy" not in slot:
        # map was built outside csv_to_npz_for_slot; verify by rebuilding
        rebuilt = csv_to_npz_for_slot(dict(slot), df, strict=True)
        if not np.array_equal(rebuilt, csv_to_npz):
            raise AlignmentError(
                f"slot {slot.get('name')!r}: csv_to_npz disagrees with canonical policy"
            )


def assert_pair_aligned(*maps_with_names: tuple[str, np.ndarray], n_csv: int | None = None) -> None:
    """For cossim / EigenMap / dev_from_shared: every input map must share the
    same `n_csv` shape so that `common = (a>=0) & (b>=0) & ...` actually
    refers to the same CSV rows in every map. Catches the case where one map
    was built against a different library.
    """
    if not maps_with_names:
        return
    ref_name, ref = maps_with_names[0]
    if n_csv is not None and ref.shape[0] != n_csv:
        raise AlignmentError(f"map {ref_name!r} length {ref.shape[0]} != n_csv {n_csv}")
    for name, m in maps_with_names[1:]:
        if m.shape != ref.shape:
            raise AlignmentError(
                f"maps {ref_name!r} and {name!r} have different shapes "
                f"({ref.shape} vs {m.shape}); built against different libraries?"
            )


class AlignmentError(AssertionError):
    """Raised when csv_row<->attr_row mapping is suspect.

    Subclasses AssertionError so existing `assert` paths still catch it.
    """
