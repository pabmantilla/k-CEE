# kcee-ui — version log

## v0.19.3 — 2026-05-15
- **Fixed `sphere render failed: Invalid element(s) … 'color' … [None]`.** Continuous color substituted NaN→`None`, which `Scatter3d.marker.color` rejects. Now NaN-color rows are dropped from the sphere via a plot mask `_pm` applied consistently to `x/y/z`, `customdata`, and color (matches the 2D scatter, which also hides NaN-color points). Discrete + var-ratio fallback unaffected.
- Files: `app.py`, `.ui-guy/VERSION.md`.

## v0.19.2 — 2026-05-15
- **Reliable 3D click.** plotly's `plotly_click` rarely fires on gl3d (the camera treats almost any click as a drag → "just click and drag, no selection"). Synthesized a real click from events that DO fire reliably in 3D: `plotly_hover` tracks the point under the cursor; a native `mousedown`→`mouseup` with <6px movement and <600ms selects the hovered point. `plotly_click` kept as a bonus primary path. Drag-to-rotate unaffected (movement >6px = no select).
- Files: `kcee_ui/components/sphere_click/index.html`, `.ui-guy/VERSION.md`.

## v0.19.1 — 2026-05-15
- **Wireframe sphere is now thin LINES, not a `go.Surface`.** The surface spanned the whole unit sphere and intercepted 3D click-rays before they reached the points behind it (plotly returns the topmost trace under the cursor — the surface — which has no `customdata`, so clicks were silently dropped). Lines (~1px, `hoverinfo='skip'`, no customdata) don't occlude, so clicks land on the points. Most likely cause of "renders but can't click".
- Files: `app.py`, `.ui-guy/VERSION.md`.

## v0.19.0 — 2026-05-15
- **Sphere click rebuilt to actually work.** Root causes of prior failures: (1) CDN Plotly could be blocked → blank; (2) plotly.py 6 `to_json()` binary-encodes numpy arrays (base64 typed-arrays) which the bare component + JS click couldn't decode; (3) re-clicking the same dot returned an unchanged value so Streamlit didn't rerun.
- Fixes: **Plotly.js 2.35.2 bundled locally** (`kcee_ui/components/sphere_click/plotly.min.js`, same-origin, no CDN). Figure data arrays (`x/y/z`, `customdata`, `marker.color`, surface mesh) emitted as **plain JSON lists** (`.tolist()`), so the browser gets standard arrays. Component returns `{row, n}` with a **click nonce** so every click (even the same dot) reruns. On-iframe **status line** surfaces load/render/click state for diagnosis.
- `show csv row` number_input still kept as fallback. Figure math/colors/baseline unchanged from v0.17–0.18.
- Files: `app.py`, `kcee_ui/components/sphere_click/index.html`, `kcee_ui/components/sphere_click/plotly.min.js` (new), `.ui-guy/VERSION.md`.

## v0.18.0 — 2026-05-15
- **True 3D click-select** via a no-build static `components.v1.declare_component` (Plotly 2.35.2 CDN + bare `postMessage`, no npm/pip). Renders the SAME plotly figure and posts the clicked point's `customdata[0]` (csv row) back to Python. Replaces native `st.plotly_chart` in the sphere branch — Streamlit native selection does not fire on `Scatter3d`. `show csv row` number_input kept as a manual fallback.
- Files: `app.py`, `kcee_ui/components/sphere_click/index.html`, `.ui-guy/VERSION.md`.

## v0.17.1 — 2026-05-15
- **Reverted `streamlit-plotly-events`** — its 0.0.6 frontend does not render on Streamlit 1.57 (sphere went blank). Dependency removed. Sphere renders via native `st.plotly_chart` again (visible + rotatable). Native `on_select` selection is still attempted (harmless if 3D click no-ops); `show csv row` number_input remains the reliable selection path.
- **Black backdrop**: explicit `paper_bgcolor`/`scene.bgcolor` black with white fonts and dark grid, matching the look Pablo preferred (no longer theme-dependent).
- Files: `app.py`, `pyproject.toml`, `.ui-guy/VERSION.md`.

## v0.17.0 — 2026-05-15
- **Sphere reuses the panel color exactly**: removed the bespoke `st.selectbox("sphere color")` + tri-cmap/log2FC branch. The sphere `Scatter3d` now mirrors the 2D scatter — discrete → per-row hex; continuous → `plot_colorscale` with `_vmin/_vmax` and `color_label` colorbar; falls back to the viridis var-ratio tri-cmap (`cmin=1/3`) only when no panel color metric is active (all-NaN).
- **True click-select** via `streamlit-plotly-events` (`plotly_events`, click only). Streamlit native `on_select` does not fire on `Scatter3d`. Maps `curveNumber`/`pointNumber` → `_csv_rows` → `sphere_pick_csv` → existing `sel_csv` pipeline. `show csv row` number_input kept as a secondary fallback.
- **"fully shared (mech+func)" baseline**: dashed black `Scatter3d` line from origin along the equiangular diagonal `(1,1,1)/√3` (top eigenvector identical across all 3 cell types, var-ratio→1).
- Files: `app.py`, `pyproject.toml`, `.ui-guy/VERSION.md`.

## v0.16.1 — 2026-05-15
- **3D sphere selection workaround**: Streamlit `on_select` does not fire on plotly `Scatter3d` clicks. Added a `show csv row` number_input under the sphere — hover a dot to read its `csv_row`, type it in → drives the existing `sel_csv` attribution panel (fallback after the event path). Hover template already exposes `csv_row`.
- Files: `app.py`, `.ui-guy/VERSION.md`.

## v0.16.0 — 2026-05-15
- **Interactive plotly `Scatter3d` eigenvector sphere** replaces the static matplotlib sphere — rotate/zoom + click a point → attribution-logo panel, reusing the exact same `event.selection → sel_csv` pipeline as the 2D plotly scatter (point `customdata` = csv row id).
- Faint `go.Surface` unit sphere as a non-clickable reference (`hoverinfo='skip'`).
- Color selector (`st.selectbox`): default `EI_1 var ratio (z-norm)` (viridis tri-cmap, cmin=1/3) + one option per cell type `"{CT} log2FC"` (RdBu reversed, symmetric ±98th-pctile). Collapses the 3-panel-per-CT log2FC idea into one rotatable+selectable scene.
- Removed matplotlib usage from this branch.
- Files: `app.py`, `.ui-guy/VERSION.md`, `.ui-guy/design.md`.

## v0.15.0 — 2026-05-15
- **New default 3D main-panel view: matplotlib "eigenvector sphere"** — per-sequence dominant eigenvector of the z-normed cell-type importance covariance plotted on the unit sphere, radius = var-explained, colored by var-ratio (viridis tri-cmap). Math reuses the exact `dev_from_shared_eig` pipeline.
- Added `view3d` sidebar radio (`eigenvector sphere` [default] / `scatter`); plotly scatter unchanged and reachable via toggle. 2D behavior untouched.
- New `top_eigvec_from_shared` (kcee_ui/scoring.py) and disk-cached `_top_eigvec_full` (app.py, packed (N,4): cols 0:3 = eigvec, col 3 = var-ratio).
- Files: `app.py`, `kcee_ui/scoring.py`.

## v0.14.0 — 2026-05-14
- **Removed clustering UI (kmeans, gmm, t-SNE diagnostic panel) — too slow.** Stripped the `cluster by kmeans/gmm` checkbox block from the `color` expander, the centroid overlay on the main scatter, the `cluster_label` download column, the t-SNE panel below the scatter, and the t-SNE → row-picker hop.
- `sklearn` (`KMeans`, `TSNE`, `GaussianMixture`) imports dropped; `scikit-learn` removed from `pyproject.toml`.
- Files: `app.py`, `pyproject.toml`.

## v0.13.3 — 2026-05-12
- **`method` radio (`kmeans` / `gmm`) at top of the cluster block.** Checkbox label flips to `cluster by kmeans/gmm`; session key `pc_kmeans_enable` and downstream label/centroid/t-SNE plumbing unchanged.
- **`covariance_type` selectbox for GMM** (`full` / `tied` / `diag` / `spherical`, default `full`); rendered only when method is `gmm`. Cache key extended with `(method, gmm_cov)`.
- **Download column rename `cluster_kmeans` → `cluster_label`** so it isn't method-specific. t-SNE panel title now reads `t-SNE of cluster feature space (…)`.
- Files: `app.py`.

## v0.13.2 — 2026-05-12
- **SEAM cold-load: 141 s → 0.22 s.** Replaced the per-seq loop in `_stack_maps` (~8.4k tiny `np.load` calls — 1059 seqs × 2 cell types × {wt, foreground_scaled, full cluster_backgrounds.npy}) with a single `np.load` of a pre-baked `seam_maps_v1.npz` (27.3 MB, all 6 arrays + onehot + seq_idxs).
- New pre-bake script `tools/build_seam_maps.py` writes the npz to `SEAM_target_spaces/results/seam_maps_v1.npz`. Re-run whenever upstream foregrounds change.
- Files: `kcee_ui/seam.py`, `tools/build_seam_maps.py` (new).

## v0.13.1 — 2026-05-12
- **t-SNE diagnostic panel** rendered below the main scatter, gated behind a new `show t-SNE embedding` checkbox inside the k-means block (only shown when k-means is committed).
- **Inline controls**: `perplexity` (5–50, default 30), `random_state`, and a `run t-SNE` button that gates the compute. Caption reflects state (`no embedding yet` / `embedding ready · perplexity=p · seed=rs`).
- **Reuses the standardized k-means feature matrix** (`_X_full[_fit_mask]`, optionally standardized) as the t-SNE input. Cache keyed by `(km_hash, perplexity, random_state)`; embedding stored as `(N_full, 2) float32` with NaN for excluded rows. `D < 2` fallback drops the t-SNE call and places the 1D values on x with y=0.
- **Colored by k-means cluster id** (one Scattergl trace per cluster, ordered by count DESC, same `_extended_palette(k)` as the main scatter). Black-ringed centroid markers over the mean (x, y) per cluster. Clicking a point feeds the same `sel_csv` row-picker as the main scatter.
- Files: `app.py`.

## v0.13.0 — 2026-05-12
- **`cluster by k-means` toggle in the `color` expander** — fully independent of the `discretize into bins` toggle (both can be on at once; when both are committed, bins drive color and k-means contributes the centroid overlay + download column only).
- **Features dropdown** (no default — first option is the `— pick features —` placeholder): `(x, y)` / `mech only (y-axis value)` / `func only (x-axis value)` / `(x, y) + active color metric` (only when the current color is a continuous metric) / `single mech: pairwise cossim` / `single mech: EI_1 / eigenmaps`. Single-mech entries reuse the cached `_cossim_full` / `_eigenmaps_full` / `_dev_eig_full(weighted=True)` calls (no duplicated plumbing).
- **Full-dataset fit + stable label caching across filters.** Cache key hashes features+k+std+random_state+x/y/color tags+the valid-row csv index set, so toggling other filters does not retrigger the fit. Labels stored under `pc_kmeans_labels__<hash>`, aligned to `common_csv` with `-1` for rows excluded from the fit.
- **Compute-cost warning** before `set clusters` when `N_valid * k > 200_000` OR `N_valid * D * k * 300 * 10 > 5e9`. `set clusters` is the gate; no separate "compute anyway" button.
- **Centroid overlay** drawn on top of the scatter (black-ringed dots, `size = max(12, dot_size*2)`, hover `cluster i · n=count`). Centroids projected back to (x, y); 1D features use the cluster-member mean for the missing axis; the color-metric dim is dropped for `(x, y) + active color metric`.
- **`cluster_kmeans` opt-in download column** in the `download lib` popover (default-off, value `cluster_{i}` per row, empty for excluded).
- Files: `app.py`, `pyproject.toml`, `uv.lock` (added `scikit-learn`).

## v0.12.4 — 2026-05-11
- **Per-bin marginal normalization.** New `marginals → per-bin normalize (discrete only)` checkbox: when on, each category's marginal histogram is divided by ITS OWN finite count instead of the global count, so a small bin's distribution is comparable to a dominant bin's. Switches `barmode` from `stack → overlay` (with 0.55 opacity per bar) so per-bin distributions are visible without geometric stacking. Threaded as `per_bin_norm: bool` through `_scatter_fig` → `_add_discrete_marg`.
- **Bins no longer have to span the full data range.** Each bin's range slider now exposes BOTH thumbs as fully draggable across `[dmin, dmax]`. Source of truth shifted from inner-edge list (length N−1) to `_full_edges` (length N+1), where `full_edges[0]` is bin 0's left thumb and `full_edges[N]` is bin N−1's right thumb. Session_state key renamed to `pc_bin_full_edges__<tag>` to force a clean migration.
- **Unbinned rows are auto-excluded.** `np.digitize(metric, full_edges) - 1` returns -1 below / N above the bin coverage; both are masked into `_excluded_rows_mask` and dropped from the scatter + marginals. Caption surfaces the unbinned count alongside the explicitly-excluded count: `bins applied · N bins · K excluded · M unbinned`.
- Files: `app.py`.

## v0.12.3 — 2026-05-11
- **One range slider per bin (replaces edge sliders).** Each bin's row now is `[include checkbox][color swatch][range slider with start + end thumbs]` — the slider's two thumb values ARE the bin's start/end ticks Pablo asked for. Bin 0's left thumb is locked at `dmin`, bin N−1's right thumb at `dmax`; middle bins span `(edges[k-1], edges[k])`.
- **Edges stored once in `st.session_state[pc_bin_edges__<tag>]`** as the source of truth. Each slider has an `on_change` callback that writes the moved thumb back into the edges list, then propagates the new shared boundary into the adjacent bin's slider state (keeps neighbours in sync without thumbs drifting). Edges sorted on every update so out-of-order drags self-correct.
- **Old per-bin text legend (start—end labels) removed** since the slider now visualises the same start/end values directly.
- Files: `app.py`.

## v0.12.2 — 2026-05-11
- **Per-bin row UI inside the `color` expander.** When bins are applied each bin gets its own row: an `include` checkbox + color swatch + start tick (`<code>{lo:.3g}</code>`) + end tick (`<code>{hi:.3g}</code>`). The two ticks make the bin boundaries explicit instead of relying on a `[lo, hi]` legend. Replaces the old `<br>`-joined HTML legend for custom bins (library-categorical legends still use the old compact form).
- **Unchecking a bin removes its rows from the scatter** (and stacked marginals). Builds an `_excluded_rows_mask` per-bin via `np.isin(_idx, excluded_set)`, then `valid &= ~_excluded_rows_mask` before the scatter is drawn. NaN-metric rows are never auto-excluded by bin index. Caption surfaces the excluded count.
- Files: `app.py`.

## v0.12.1 — 2026-05-11
- **Plot controls grouped into collapsible expanders, all closed by default.** Five sections inside the bordered `Plot controls` panel: `library category` (annotation column + filter, lifted out of its old standalone block), `axes` (x/y axis choosers), `color` (color-by selectbox + `discretize into bins` toggle + binning UI + per-bin legend), `layout` (dot size, figure width/height, auto axis limits + manual mins/maxes), `marginals` (marginal x/y), `overlays` (FiNeMo hits checkbox, highlight csv row). The data-pool building (axis_pool, color_pool) stays outside expanders so binning controls can read the current color metric on first paint. Each expander reveals its widgets only when clicked, so the panel header is just six section labels by default.
- Files: `app.py`.

## v0.12.0 — 2026-05-11
- **Binning is now an inline option on any continuous color metric** instead of a special `custom bins` entry. The `custom bins` and `metric to bin` selectboxes are gone — pick any continuous color via `color by`, then flip the `discretize into bins` toggle that appears beneath it. Toggle off → normal continuous coloring (state A). Toggle on, bins not yet `set` → continuous coloring + vertical gradient strip on the right with bar-position preview (state B). Click `set bins` → discrete coloring with X distinct categorical colors and the gradient strip disappears (state C). Click `unset bins` to go back to preview.
- **X distinct categorical colors for X bins.** Bin colors come from the existing `_PALETTE` (Wong + extension, 12 entries, recycles past 12) instead of `sample_colorscale` along the active continuous colormap — adjacent bins are now visually distinct rather than near-identical gradient samples.
- **`# bars` cap raised to 11** (was 7) since the categorical palette can support up to 12 distinct colors before recycling.
- All session-state keys for the binning UI are scoped by `color_mode`, so switching the active color metric resets the toggle/edges cleanly.
- Files: `app.py`.

## v0.11.5 — 2026-05-11
- **Discrete coloring now splits the scatter into one trace per category** ordered by point count DESC (largest added first → smallest rendered ON TOP), so small bins are no longer buried under the dominant cloud. Applies to all discrete coloring (custom bins + library categorical).
- **Per-category stacked marginals.** When discrete and a marginal is enabled, each category gets its own colored bar at every shared bin edge; bars stack (`barmode="stack"`) in the same largest-first order so smaller categories sit at the top of the column. Density/probability normalize against the global finite count so stacked totals stay comparable. Replaces the old "fall back to plain gray counts" path. Helper `_shared_edges` builds the common 30-bin grid once per axis; helper `_add_discrete_marg` does the per-category bincount + Bar emit (used for both x and y orientations).
- Files: `app.py`.

## v0.11.4 — 2026-05-11
- **`set bins` button gates the discrete coloring.** In custom-bins mode the scatter is now colored continuously by the chosen metric until the user clicks `set bins` — the bars on the gradient act as a live preview of where the cuts would land, so you can position them against the continuous colormap. Clicking commits the current edges, switches the scatter to discrete coloring, and unlocks `bin_<metric>` in the download popover. Button label flips to `back to continuous` once active. State stored in `st.session_state["pc_color_bins_active"]`. Caption beneath the button surfaces the current state (`preview …` / `bins applied · N bins`).
- **Scatter's native colorbar is suppressed in custom-bins mode** (added `show_colorbar` arg to `_scatter_fig`) so the right-side gradient strip is the only colorbar — no double colorbars in continuous-preview mode.
- Files: `app.py`.

## v0.11.3 — 2026-05-11
- **Custom-bins control relabeled `# bars` (default 2), with `# bins = # bars + 1`.** Bars on the gradient *are* the bin boundaries — the new label matches that mental model. Default 2 bars → 3 bins.
- **Bin assignment is now an opt-in download column.** When custom-bins coloring is active, `download lib` exposes `bin_<metric>` (e.g. `bin_log2FC HepG2 − K562`) in the columns multiselect; selecting it writes the per-row bin label (`[lo, hi]`, or `NA` if the metric was NaN) into the CSV. Default-off so existing exports are unchanged.
- Files: `app.py`.

## v0.11.2 — 2026-05-11
- **Custom-bins gradient now sits to the right of the scatter** instead of inside the left controls panel. Switched to a vertical `go.Heatmap` (height = `plot_fig_h`, width 120 px) with `add_hline` edge bars and `yaxis.side="right"` — visually it occupies the same slot as the scatter's native colorbar. Slider controls for each edge stay in the left controls panel. Scatter render is wrapped in `st.columns([6, 1])` only when `_color_grad_fig` is set; non-bins modes render the scatter at full width as before.
- Files: `app.py`.

## v0.11.1 — 2026-05-11
- **Custom-bins widget redesigned to mirror the scatter colorbar.** Histogram + separate color strip dropped — replaced with a single horizontal `go.Heatmap` rendered with the active `plot_colorscale` (so the strip looks identical to the scatter's continuous colorbar) and black `add_vline` bars at each edge. Edge controls are now `st.slider`s (one per inner edge) instead of number_inputs, so editing a bin boundary feels like dragging a thumb against the gradient.
- Files: `app.py`.

## v0.11.0 — 2026-05-11
- **`color by → custom bins`.** New entry in the color-by selectbox that discretizes any continuous color metric into user-defined bins. When picked: sub-selectbox `metric to bin` (drawn from the continuous entries of `_color_pool`), `# bins` input (2–8, default 4), and N−1 `edge i` controls (default at evenly-spaced quantiles, clamped to `[dmin, dmax]`). Bin colors are sampled from the active `plot_colorscale` (`sample_colorscale` at midpoints) so the discrete swatches match the scatter's continuous look. Per-row hex array routes through the existing `_color_is_discrete` path; legend below the selectbox shows `[lo, hi]` ranges with swatches. NaN metric rows get `_NA_HEX`. Edge keys include `bin_metric` + `n_bins` so switching either resets stale values.
- **Continuous color gradient now spans full data range** instead of 2–98% percentile clip — extreme highs and lows now map to distinct colorbar endpoints (no more saturating before the actual min/max).
- Files: `app.py`.

## v0.10.9 — 2026-05-11
- **Download lib: `attr_<slot>` is now default-on, `imp_<slot>` removed.** Importance is `sequence · attr`, so downstream code can derive it from the two columns already present; one less heavy option in the multiselect. `attr_<slot>` (full hypothetical (4×L) = 800 floats per row, space-separated, row-major A→T) is selected by default for every ABC slot — the input-like default schema is now `seq_idx, name, chr, start, stop, sequence, pred_<slot>…, attr_<slot>…`. Resolved lazily per-box via `_cached_load`.
- Files: `app.py`.

## v0.10.8 — 2026-05-11
- (Superseded by v0.10.9 — imp removed, attr default-on.) Added `attr_<slot>` and `imp_<slot>` as download options.
- Files: `app.py`.

## v0.10.7 — 2026-05-11
- **Unified `color by` pool with discrete categorical support.** The `color by` selectbox now draws from the same value pool as the x/y axis dropdowns (every continuous axis option) plus per-slot `residual (slot − ct)` entries and the legacy `average magnitude` synthetic, AND any low-cardinality (≤20 unique) library column as a true discrete entry (`library: {col}`). Discrete entries use a 12-color colorblind-friendly palette (Wong + extension) with `#BDBDBD` reserved for missing; per-row hex strings feed `marker.color` directly so Plotly draws true categorical colors instead of mapping codes through a continuous colormap. Old `pred_cts`/`meas_cts`/`resid_cts` cascade and `_CM_PRED/_CM_MEAS/_CM_RESID/_CM_XAXIS/_CM_ANNOT` constants removed; the intermediate `color_cell_line` selectbox is gone — each combo is an explicit pool entry. `_scatter_fig` gains a `discrete: bool` cache-key arg: when true, marker drops `colorscale`/`colorbar`/`cmin`/`cmax`, hover loses `color=` line, and marginal histograms fall back to plain counts (no color-weighted means). Percentile clip and `_color_arr` finite-check are skipped for discrete. A compact swatch+label legend renders directly below the selectbox when discrete (≤12 categories, names truncated at 28 chars). The point-click → csv_row flow, box overlays, and highlight ring are unaffected.
- Files: `app.py`.

## v0.10.6 — 2026-05-11
- **Download lib: input-like compact default schema.** Default columns are now `seq_idx, name, chr, start, stop, sequence` + one `pred_<slot>` column per ABC slot with a `pred_key`. Library `*_hg38` columns are renamed to `chr/start/stop` on output. Optional pool also includes the three `*_log2FC` columns + `category` + `str_hg38` (unchecked by default). `box_color` removed from the CSV (`box_id` alone suffices). New `float decimals` input (default 4) feeds `to_csv(float_format=…)` so predictions/log2FC stay compact.
- Files: `app.py`.

## v0.10.5 — 2026-05-11
- **3D mech axis is now "deviation from shared" via eigendecomposition** (matches `genomic_targets/scripts/3d_example/eigen_interactions_filtering.ipynb`). New `dev_from_shared_eig(imp_list, weighted)` in `scoring.py`: per-sequence covariance of z-normalised importance across the N cell types → eigendecomposition → return `1 - |EI_1 · shared_dir|` (range [0, 1]; 0 = perfectly shared, 1 = orthogonal). Sign-invariant.
- **Sidebar `score (mech axis)` in 3D** now has only `cossim` (default — unweighted) and `EigenMaps` (= `var_ratio_1 × (1 - |EI_1·shared|)`, analog of 2D `var_ratio*r`). `dev_from_shared` is no longer a dropdown option — it IS the y-axis, and these two options just choose how to compute it. New cached `_dev_eig_full` mirrors `_dev_full`'s plumbing.
- **Color-clipping UI removed.** The `auto (2–98%) / manual / full range` radio + manual `vmin/vmax` inputs are gone — colorbar always uses auto 2–98% now.
- Files: `app.py`, `kcee_ui/scoring.py`.

## v0.10.4 — 2026-05-11
- (Superseded by v0.10.5 — 3D score model rewritten.) 3D mode `cossim`/`EigenMaps`/`dev_from_shared` chooser via pairwise mean; color-clipping UI removed.
- Files: `app.py`.

## v0.10.3 — 2026-05-11
- **Plot controls panel consolidation.** Colorscale selectbox dropped — `plot_colorscale` is now pinned to the per-score-type default `score_cmap` (`RdBu_r` for cossim, `Inferno` for EigenMaps, `Magma` for dev_from_shared). New `dot size` slider (1–12, default 4) threaded through `_scatter_fig` and applied to both the base scatter marker and box-overlay markers (highlight ring stays at 16). Sidebar `Library annotation` section moved into the plot controls panel as `library category`; widgets render in the same bordered container, rendered in two passes (annotation before scoring, rest after). New `x-axis` / `y-axis` selectboxes — pool includes `auto`, score (mech axis), per-CT measured log2FC, per-slot predicted activity, `log2FC HepG2 − K562`, and `pred(A) − pred(B)` (when both have pred_keys); overrides apply before the `valid` mask is built and update axis labels.
- Files: `app.py`.

## v0.10.2 — 2026-05-11
- **Colorbar title runs vertically alongside the gradient** (`colorbar.title.side="right"`) — long labels like `measured log2FC (HepG2)` no longer eat horizontal space above the scale. Applied in both kcee mode (`_scatter_fig` marker) and SEAM mode (per-type marker).
- **Colorscale options curated to colorblind-friendly only.** New list: `Viridis` (default), `Cividis`, `Plasma`, `Magma`, `Inferno` (sequential) + `RdBu_r`, `PuOr_r` (diverging). Dropped `Turbo` (rainbow-like, not CB-safe). Applied to both modes; `dev_from_shared` 3D-mode default switched from `Turbo` to `Magma`.
- Files: `app.py`, `kcee_ui/seam.py`.

## v0.10.1 — 2026-05-11
- **Box numbering is now position-based, assigned at render/download time.** Boxes carry only a hidden `_uid` (stable widget key) and a color; the displayed `box N` label uses the current array position (1-based). Deleting box 2 of {1,2,3} leaves the remaining two relabeled as box 1 and box 2 — no gaps. CSV `box_id` column likewise uses the position at download time.
- Files: `app.py`.

## v0.10.0 — 2026-05-11
- **Isolate / box-select on the scatter.** New toolbar strip above the kcee scatter: `isolate` toggle (left), inline box chips with colored swatch + point count + `✕` delete (middle), and `download lib` popover + `clear boxes` button (right). When isolate is on, the scatter switches to `dragmode="select"` and `selection_mode=("points","box")`; dragging a box appends to `st.session_state["isolate_boxes"]` (geometry-hash dedup via `isolate_last_box_hash` so reruns don't re-append). Each box is drawn as a rect shape + a color-matched overlay Scattergl on top of the base trace (base trace untouched so the point-click → csv_row flow is unchanged for non-isolate clicks). Point-click is gated off while isolate mode is active. Download popover builds a combined `library.iloc[rows] × box_id/box_color` CSV (per-box rows, no dedup) with user-selected boxes/columns. `_scatter_fig` gained `dragmode`/`boxes_hash` cache-key args plus a leading-underscore `_boxes` pass-through so cache invalidates on dragmode/box changes but doesn't try to hash the live box list.
- Files: `app.py`.

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
