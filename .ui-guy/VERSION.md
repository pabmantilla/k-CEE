# kcee-ui — version log

## v0.9.4 — 2026-05-11
- **SEAM: background is the intra-cluster bg, not the scaled avg.** Per `seam_foreground_viewer.ipynb`, the "background" should be `cluster_backgrounds[ref_cluster_idx]` (the entropic-position bg from MetaExplainer at the seq's reference cluster), NOT `average_background_scaled.npy` which was the global cell-type average. New helper `_load_intra_bg(seq_dir)` resolves the ref cluster and slices. Affects both the scatter (cossim-bg) and the rendered maps.
- Stale cossim-bg numbers from earlier versions are invalidated automatically (Streamlit caches re-hash on function-code change).
- Files: `kcee_ui/seam.py`.

## v0.9.3 — 2026-05-11
- **SEAM mode: WT-projected logo toggle.** Sidebar checkbox `WT-projected logo (attr × onehot)` (parity with kcee mode). When on, each rendered map (WT / foreground / background, per cell type) is multiplied by the WT one-hot from `df["sequence"]` so only the WT base at each position survives; titles get a `· WT-projected` tag.
- Files: `kcee_ui/seam.py`.

## v0.9.2 — 2026-05-11
- **SEAM scatter gets the same plot-controls panel as kcee mode.** Left-column `Plot controls` box with: `color by` / `colorscale` / `color clipping` (auto 2–98% / manual / full range) / `figure width/height` / `marginal x/y` (none/counts/density/probability) / `auto axis limits` + manual `xmin/xmax/ymin/ymax` / `highlight seq_idx`.
- **`color by` modes for SEAM:** `by type (WT/fg/bg)` (default — gray/red/blue traces); or per-seq value: `EI_1 var x r`, `log2FC HepG2 − K562`, `log2FC HepG2`, `log2FC K562`, `predicted HepG2`, `predicted K562`, `predicted HepG2 − K562`, `condition`. When colored by value, each visible type still gets its own trace (so type is preserved as marker shape: WT = circle, foreground = △, background = ▽) and the colorscale + colorbar are shared.
- **Marginals** (counts/density/probability) work over the union of visible points and are color-weighted when a value-based `color by` is active.
- Reference lines at x=0 and y=0; `highlight seq_idx` draws a red ring around every visible-type point for that sequence.
- Files: `kcee_ui/seam.py`.

## v0.9.1 — 2026-05-11
- **SEAM viewer is now a mech/func scatter** mirroring the kcee browser layout. Each of the 1059 sequences contributes up to 3 points sharing the same x (log2FC HepG2 − K562, measured): **gray WT**, **red foreground (scaled)**, **blue background (avg scaled)**. Y axis = cossim(HepG2 attr, K562 attr) on z-normalised importance over the 200-bp var region.
- **Sidebar view picker:** `all three` / `foregrounds only` / `backgrounds only` / `WT only`. Sidebar also has a condition filter (`all / diff-diff / same-diff / same-same`).
- **Click a point → maps below.** In `all three` view: 3 rows (WT/foreground/background) × 2 cols (HepG2/K562); the clicked type's title is bolded + colored. In one-type view: a single row (the selected type) × 2 cols (HepG2/K562). Title strip shows seq_idx, condition, EI_1, predictions, log2FC.
- Maps + onehot + cossim pre-computed once and cached (`_stack_maps` / `_compute_scores`). Cold load ≈90 s (≈6 k npy reads); warm refresh is instant.
- Files: `kcee_ui/seam.py`.

## v0.9.0 — 2026-05-11
- **SEAM mode toggle.** Top-right radio (`kcee` / `SEAM`) switches between the existing per-row attribution browser and a new SEAM foreground/background viewer.
- **SEAM viewer (stub):** picker for condition (`diff-diff` / `same-diff` / `same-same` / `all`) + sequence (sorted by EI_1 ascending; label shows name · seq_idx · EI_1 · condition). Renders a 3×2 logo grid per pick: WT attribution / SEAM foreground (scaled) / SEAM background (avg scaled, ref cluster) for HepG2 + K562. Predictions and EI_1 shown in the caption.
- Data source: `Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/SEAM_target_spaces/`. 1059 sequences (353 per condition), Pablo's AG models only, HepG2+K562 only — WTC11 and other families are not represented in SEAM space.
- Files: `app.py`, `kcee_ui/seam.py` (new).

## v0.8.1 — 2026-05-11
- **Methods mode: flat x-axis at 0.** When `k-condition == methods` the A/B/C slots share the same (family, cell line) so predictions are identical — the functional pred-diff axis is meaningless. Default x is now `zeros(N)` with label `(no functional axis — predictions identical across methods)`, leaving the scatter purely attribution-driven (y = score).
- **Mode tag under the title.** A small `st.caption` below "k-CEE attribution browser" reflects the current sidebar pickers, e.g. `Koo lab models · DeepLIFT · comparing cell lines`, `HepG2 · DeepLIFT · comparing models`, or `Koo lab models · HepG2 · comparing methods`. Filled via `st.empty()` slot so the tag updates when pickers change.
- Files: `app.py`.

## v0.8.0 — 2026-05-11
- **Attribution method is now a comparison axis.** Sidebar `k-condition` gains `methods` (fixes family + cell line, varies attribution method). `cell lines` and `models` modes now expose explicit Family + Method (or Cell type + Method) pickers. Method dropdowns are restricted to methods whose H5 exists on disk, so new files (e.g. Pablo+IntGrad / LegNet+IntGrad once SLURM 2166841 / 2166850 land) auto-appear with no code changes.
- **DATA_SOURCES collapsed from 5 entries to 3 families** (`Koo lab models` / `Pablo models` / `MPRA-LegNet`). Per-method Koo entries are gone — method is no longer a top-level data source.
- `defaults.py` refactored around `FAMILIES = {family: [slots]}` with cascading-picker helpers (`family_names`, `methods_for_family`, `cts_for_family_method`, `methods_for_family_ct`, `slots_for_family_method`, `slots_for_ct_method`, `slots_for_family_ct`, `methods_at_ct_across_families`, `cts_eligible_for_models_mode`, `families_with_multiple_methods_anywhere`, `cts_for_family_with_multiple_methods`). Legacy `DEFAULT_SLOTS / KOO_SALIENCY_SLOTS / KOO_INTGRAD_SLOTS / PABLO_SLOTS / LEGNET_SLOTS / MODEL_CT_OPTIONS / slots_for_cell_type` re-exported for one-off scripts.
- A/B/C selectboxes show the varying axis (cell type / family / method) instead of the full slot name. Session-state keys re-tagged with mode prefixes (`cl__` / `mdl__` / `mth__`) so switching k-condition or any picker doesn't leak stale A/B indices (same lesson as v0.5.3, v0.6.0).
- `MODE_MODELS` redefined as `k_condition != "cell lines"` (single-CT modes); `_ax_kind` resolves to `"method"` in methods mode so color-by labels read correctly.
- Files: `kcee_ui/defaults.py`, `app.py`.

## v0.7.1 — 2026-05-11
- **Centered circular load-up spinner.** CSS pins Streamlit's `stStatusWidget` to screen center and replaces its tiny built-in icon/text with a 56px circular CSS spinner (rotating red top arc on a grey ring) + "Loading…" label, so the load-up indicator is obvious instead of a thin top-right bar.
- **Per-attribution-map spinner.** Each attribution-logo column now shows `st.spinner("loading attribution map · {slot}…")` around `load_attr_row` + `cached_attribution_png`, so slow first-render rows have a visible buffer/loading sign.
- Files: `app.py`.

## data v3 — 2026-05-11 (in progress)
- **IG sweeps submitted for Pablo's AG models and LegNet ensemble**, to mirror the existing Koo grad source (IntGrad axis only — saliency dropped per Pablo's request). Both use captum `IntegratedGradients`, var-region-only dinuc baselines [15:215], hypothetical mean-centering, output slice `(N, 4, 200)`.
  - Pablo: dedicated `compute_intgrad_pablo_ag.py` + `submit_intgrad_pablo_ag.sh` (cleaner mirror of `compute_grad_standardtorch.py`, no saliency). Verified locally with the `test` mode (4 K562 seqs, intgrad (4,4,200), range ~[-0.009, 0.014]) before resubmitting. Array `2166841` (3 cts × 10 shards, n_shuffles=20, n_steps=50, bs=16). Output: `genomic_targets/data/intgrad_shards_pablo_ag/` → merge to `intgrad_pablo_ag.npz`.
  - LegNet: `legnet_rep/scripts/kcee/{compute_grad_legnet.py,submit_grad_legnet.sh}` mirroring `compute_grad_standardtorch.py` but ensemble-averaging IG across the 10 fold models (like `compute_legnet_attrs.py` does for DeepLIFT-SHAP). Array `2166850` (2 cell lines × 10 shards, n_shuffles=20, n_steps=50, bs=16). Output: `legnet_rep/results/grad_shards/` → merge to `gradattrs_legnet.npz`.
- Shard files now hold only `intgrad` + `predictions` keys (no `saliency`).
- After merge, wire into kcee-ui: extend `tools/build_attributions.py` SOURCES with `pablo_ag_ft × intgrad × 3` and `legnet_ensemble × intgrad × 2`; add `PABLO_INTGRAD_SLOTS / LEGNET_INTGRAD_SLOTS` in `kcee_ui/defaults.py`; expose as `Pablo models (IntGrad)` and `MPRA-LegNet (IntGrad)` in `DATA_SOURCES`. Bump UI version when this lands.
- Prior submissions cancelled mid-flight: arrays `2162457`/`2162458` (saliency+IG, before Pablo dropped saliency); arrays `2164474`/`2164475` (IG-only via the old eigensteering script, cancelled at 14:26 in favor of the dedicated pablo_ag script + verification step).

## v0.7.0 — 2026-05-11
- **Library annotation column for filter + color.** New sidebar "Library annotation" picker exposes any low-cardinality library CSV column (≤200 unique values, excluding `sequence` / `csv_row` / `name` / `*_log2FC` / `*_hg38`) — e.g. `category` (values like `promoter`, `putative enhancer, HepG2`). When set:
  - Multiselect "filter {col}" restricts `common_csv` to rows whose value is in the picked set; applied **before** scoring so cached `scores_full` is indexed by the filtered rows (no cache invalidation, no score re-compute).
  - `_COLOR_MODES` gets a `library: {col}` option; resolution maps each row to the integer code of its value, NaN where the column is null. Reuses the existing scatter color pipeline (cmap + clipping) — no separate categorical legend yet.
- Selectbox key is `_slot_key_tag`-scoped so changing the data source / cell-type resets the annotation picker cleanly; multiselect key is `annot_vals__{col}` so switching columns doesn't bleed selections across columns.
- Files: `app.py`.

## v0.6.0 — 2026-05-11
- **Removed editable per-slot expanders from the sidebar.** Slots auto-populate from `defaults.py` (these are just data files being loaded, not models being configured). Replaced the Slot 1/2/3 expander column with a single "Loaded: …" caption.
- **Fix: A/B (and C in 3D) selectboxes didn't refresh when changing data source or `models_ct`.** Root cause: the selectbox keys (`abc_a`, `abc_b`, `abc_a3`, …) were shared across all sources/cell types, so session_state held the old integer index and Streamlit kept the stale value until the user clicked the selectbox. Now the keys include `_slot_key_tag` (e.g. `abc_a__koo_lab_models_deeplift`, `abc_a__models_HepG2`), so each source/cell-type combo gets its own session_state slot and the defaults (`index=0`, `index=1`) apply cleanly.
- Files: `app.py`.

## v0.5.3 — 2026-05-11
- **Fix: switching data source between the three Koo sources (DeepLIFT / Saliency / IntGrad) kept the old DeepLIFT paths in every slot**, so saliency/intgrad slots silently loaded the wrong file. Root cause: `_slot_key_tag = data_source.split()[0].lower()` collapsed all three Koo sources to `"koo"`, so the slot text-input session-state keys (`path_koo_0`, …) collided. Switching the dropdown left the old value in session_state and Streamlit ignored the new `value=` default. Now the tag is the full slugified source name (`koo_lab_models_saliency` etc.), so each source has its own keys.
- Files: `app.py`.

## v0.5.2 — 2026-05-11
- **Alignment guard: replace silent `out[:m] = arange(m)` fallback** in `_build_attr_csv_to_npz`. Old code aliased LegNet (n_attr=56975) onto CSV rows [0..56974], silently misaligning row 18321 onwards (the 5 LegNet drops are [18321, 18322, 41187, 54802, 55855], not a contiguous tail). New module `kcee_ui/alignment.py` enumerates drop policies (`identity` / `dropna_seq` / `dropna_seq_log2fc_*`), picks the one matching `n_attr` exactly, and raises `AlignmentError` otherwise — no more silent first-m fallback.
- **Runtime guards**: `assert_slot_aligned` runs at slot load (surfaces as sidebar error if it fires); `assert_pair_aligned` runs at the top of every cross-model `_cossim_full` / `_eigenmaps_full` / `_dev_full` cache miss to catch maps built against different libraries.
- **NaN-seq mask**: `csv_to_npz_for_slot` always sets `m[i] = -1` for CSV rows where `sequence` is NaN (rows 18321, 18322). Catches Pablo-WTC11 "garbage rows" (model evaluated on missing-sequence input — values not NaN so the per-plot `isfinite` guard didn't catch them). Slot records `_n_seq_masked`; assertion accounts for it.
- See `.ui-guy/alignment_guard.md`.
- Files: `kcee_ui/alignment.py` (new), `app.py`.

## v0.5.1 — 2026-05-10
- **dev_from_shared now operates on WT×attr importance (N, L)** for parity with cossim/EigenMaps. Was previously consuming raw 4-channel hypothetical (N, 4, L), which gave a different geometric decomposition. `deviation_from_shared` rejects (N, 4, L) inputs explicitly. `_dev_full` now uses `_cached_importance` like the 2D scorers; cache key bumped to `v3`.
- Files: `app.py`, `kcee_ui/scoring.py`.

## v0.5.0 — 2026-05-10
- **Bugfix: WT one-hot was off by 15bp for Koo lab + LegNet**, making attribution maps and importance/cossim/EigenMaps for those sources look noisy/wrong. Root cause: `_cached_csv_onehot` built the library one-hot at the attr's length without knowing where attr position 0 sits in the 230bp insert. Now each slot has an `insert_offset` (15 for var-only files, 0 for full-construct), threaded through `seq_to_onehot(..., offset=…)`, the importance projection, the per-row WT logo, hits, and all scoring windows. See `.ui-guy/wt_alignment.md`.
- **Plot now always crops to the 200bp var region** for every source (uniform display length). Replaced the buggy symmetric `adapter_len=15` trim in `plot_attribution` with an explicit `crop=(start, stop)` arg; caller computes `crop` from `_var_window(insert_offset, attr_L)`.
- **Auto-detect insert_offset from `attr_L`** (200→15, else→0) so that when the in-progress 200bp Pablo regen lands, the UI flips its layout without any code change. The static `insert_offset` in `defaults.py` is now a hint; the file's actual shape wins.
- **Defensive NaN handling** for the two known bad rows (18321, 18322) in `deeplift_attributions_standardtorch.{npz,h5}`: per-row plot shows a warning and skips; importance/dev_from_shared zero-fill so cossim/EigenMaps don't go all-NaN. See `.ui-guy/nan_rows.md`.
- Score caches bumped to `v2` (incl. `insert_offset` in deps) so old caches are bypassed.
- Files: `app.py`, `kcee_ui/defaults.py`, `kcee_ui/data.py`, `kcee_ui/plotting.py`.

## data v2 — 2026-05-10
- "Pablo models" data source being regenerated to be **fully uniform** with the other two sources: hypothetical=True + mean-center, var-region-only dinuc shuffle [15:215] (flanks held WT), saved shape (N, 4, 200). New file written as `deeplift_attributions_uniform.{npz,h5}` — existing `deeplift_attributions.{npz,h5}` (projected, full 281bp) is left untouched per Pablo's "don't rename existing files" rule.
- SLURM array job 2140910 (30 tasks, ~70 min). After completion:
  ```
  bash submit_attributions.sh merge
  uv run python tools/npz_to_h5.py \
    --input  .../genomic_targets/data/deeplift_attributions_uniform.npz \
    --output .../genomic_targets/data/deeplift_attributions_uniform.h5
  ```
- To switch the viewer over: edit `kcee_ui/defaults.py` `PABLO_ATTR_FILE` to point at `deeplift_attributions_uniform.{h5,npz}`.
- Code change in eigen-interactions: `compute_shard` and `_compute_deeplift` now use hypothetical=True + mean-center + var-region-only references + 200bp slice. Canonical commits 441e3f1 (hypothetical fix) → followed by an unpushed local edit on the koo submodule for the var-region/200bp slice; Pablo to commit/push when convenient.
