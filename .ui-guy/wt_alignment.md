---
name: WT one-hot alignment per data source
description: Each slot's `insert_offset` is what aligns the 230bp library insert with the saved attribution; misalign and Koo/LegNet logos look wrong by 15bp
type: project
---

The library `sequence` column is a 230bp insert: `LA(15) + var(200) + RA(15)`. The variable region used by all the models is `insert[15:215]`. Saved attribution arrays differ per source in what slice they cover:

| Source       | attr shape   | what attr position 0 represents | `insert_offset` |
|--------------|--------------|---------------------------------|-----------------|
| Koo lab      | (N, 4, 200)  | insert[15] (= var[0])           | 15              |
| MPRA-LegNet  | (N, 4, 200)  | insert[15] (= var[0])           | 15              |
| Pablo models | (N, 4, 281)  | construct[0] = insert[0]        | 0               |

**Why:** Before 2026-05-10 the UI built the library one-hot with `seq_to_onehot(seq_230bp, length=attr_L)` → for Koo/LegNet (attr_L=200) this took `insert[0:200]` (LA + var[0:185]) instead of `insert[15:215]` (var). Result: every importance / WT-projected logo / cossim / EigenMaps for Koo lab and LegNet was off by 15bp, so the maps looked noisy/uninterpretable. Pablo models was unaffected because for attr_L=281 the var slice happens to live at `[15:215]` of both the attr and the (insert + zero pad) one-hot.

**How to apply:**
- Any time you build a per-row WT one-hot, use `seq_to_onehot(seq, length=attr_L, offset=insert_offset)` — never just `length=attr_L`.
- For per-row plot, pass `crop=(var_lo, var_hi)` from `_var_window(insert_offset, attr_L)` to `plot_attribution`. Don't reintroduce the old symmetric `adapter_len` trim — for var-only sources it over-trims to 185bp.
- For cossim / EigenMaps / dev_from_shared, slice importance to the per-slot var window (`_var_window`) before scoring. The hard-coded `INSERT_START=15, INSERT_STOP=215` in `kcee_ui/scoring.py` only makes sense in *insert* coords, NOT for a 200bp var-only importance vector.
- Hits are in genomic coords; `_hits_to_local` shifts them by `(15 - insert_offset)` so they land in attribution coords.
- New attribution sources: add `insert_offset` to the slot dict in `defaults.py`. Default 0 means "attr starts at insert position 0".
