# alignment guard: csv_row <-> attr_row

## Why this exists

Cossim, EigenMaps, and dev_from_shared all compare attributions for the SAME
sequence across multiple sources. If `_a_map[i] = 12345` in source A but
`_b_map[i] = 12345` actually points to a different sequence in source B, the
scores look plausible but mean nothing.

The previous `_build_attr_csv_to_npz` in `app.py` had a silent fallback:
```python
m = min(n_attr, n_csv)
out[:m] = np.arange(m)
```
This fired for LegNet (`n_attr=56975`, `n_csv=56980`) because neither
`n_attr == n_csv` nor `seq_valid.sum() == n_attr` matched. Result: LegNet rows
0..56974 were claimed to be CSV rows 0..56974 in order, but LegNet actually
drops `[18321, 18322, 41187, 54802, 55855]` (NaN seq or NaN log2FC). So LegNet
row 18321 is CSV row 18323, not 18321. Every cross-model score on rows
18321..56974 was using mismatched sequences.

## What's in `kcee_ui/alignment.py`

- `csv_to_npz_for_slot(slot, df)` — picks the drop policy whose kept-row count
  equals `slot["n_attr"]`. Three known policies:
  - `identity` (n=56980): Koo standardtorch, Pablo WTC11 (caller must mask
    rows 18321/18322 for Pablo WTC11 since those contain garbage from running
    the model on invalid input).
  - `dropna_seq` (n=56978): Pablo K/H — drops rows where `sequence` is NaN.
  - `dropna_seq_log2fc_HepG2_log2FC_K562_log2FC` (n=56975): LegNet.
  Raises `AlignmentError` if `n_attr` matches none of them.

- `assert_slot_aligned(slot, df, csv_to_npz)` — runs once at slot load. Verifies
  shape, covered-row count, and that the map agrees with the canonical policy.

- `assert_pair_aligned(*(name, map))` — runs at every cossim/EigenMap/dev cache
  miss. Catches maps with different lengths (different libraries).

## What's wired in `app.py`

- `_build_attr_csv_to_npz` delegates to `csv_to_npz_for_slot(..., strict=True)`.
- After building each slot's map, `assert_slot_aligned` is called; on failure
  the slot's map is reset to all-`-1` and a sidebar error is shown.
- `_cossim_full`, `_eigenmaps_full`, `_dev_full` all run `assert_pair_aligned`
  at the top of their `_go` closures.

## To add a new drop policy

Add a `(name, kept_csv_rows)` entry to `_candidate_policies` in
`kcee_ui/alignment.py`. The matcher is `len(kept) == n_attr`, so policies must
have unique kept-row counts. If two policies could share `n_attr`, add an
explicit `slot["drop_policy"]` override and read it first.

## Things this guard does NOT check

- That attribution values themselves came from running the model on the same
  sequence as the CSV row (i.e. provenance). For that, rerun model on a sample
  row and check `predictions[i]` against a fresh inference. This guard only
  ensures row-INDEX agreement, not value provenance.
- That `insert_offset` per slot is correct (handled separately — see
  `.ui-guy/wt_alignment.md`).
