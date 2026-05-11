---
name: NaN rows in standardtorch attribution file
description: Two known bad rows in deeplift_attributions_standardtorch.{npz,h5} — guard plots and zero-fill for scoring
type: project
---

`…/genomic_targets/data/deeplift_attributions_standardtorch.npz` (the file `"Koo lab models"` reads from in the UI) has **rows 18321 and 18322** entirely NaN for all three cell types in both `attr_*` and `predictions_*`. All other 56978 rows are finite.

Likely a model forward-pass failure for those two sequences during the SLURM array job (probably one bad shard task that wasn't re-run). **Per `feedback_dont_rename_existing_files.md` we don't rename/regenerate the file in place** — the UI handles this defensively instead.

**Why:** A single NaN poisoning hits hard:
- `attr_to_importance` propagates → entire importance row NaN → cossim / EigenMaps / dev_from_shared scores become NaN for those rows (visible as gaps in scatter).
- `plot_attribution` on a NaN row blows up `fast_logo` (eps calc, ylim).

**How to apply:**
- `_cached_importance` (app.py) replaces NaN with 0 in attribution before computing importance — the importance row stays meaningful (just zeros where the model failed) and downstream scores aren't all-NaN.
- `_dev_full` does the same zero-fill on the per-slot subarrays before `deviation_from_shared`.
- Per-row plot loop in app.py guards `if not np.isfinite(attr_row).all()` and shows a warning instead of plotting.
- If a future regen produces a new file with NaN rows, **expect this to happen again** — check with:
  ```python
  import numpy as np
  with np.load(p) as d:
      for k in [k for k in d.files if d[k].ndim == 3]:
          bad = np.where(np.isnan(d[k]).any(axis=(1, 2)))[0]
          print(k, bad)
  ```
  Update the warning message in app.py per-row plot loop with the new row indices.
