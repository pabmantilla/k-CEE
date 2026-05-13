# t-SNE diagnostic panel (v0.13.1)

Diagnostic for the k-means fit: does the embedding visually separate the clusters that k-means found?

## Location

- Controls are inline inside the k-means block in the `color` expander, rendered after the `kmeans applied · …` caption — only when k-means is committed (`pc_kmeans_active` is True AND `_kmeans_labels is not None`).
- The figure is rendered **below the main scatter** inside the `_center` column, OUTSIDE the toolbar columns, before the row-picker / maps.

## Input features

Uses the **same standardized matrix that k-means was fit on**: `_X_full[_fit_mask]` (rows where every feature column is finite), optionally column-standardized when the k-means `standardize` toggle was on. No re-derivation of the feature choice — the t-SNE inherits whatever k-means chose.

## Controls

- `show t-SNE embedding` — checkbox (key `pc_tsne_show`).
- `perplexity` — slider 5–50, default 30 (key `pc_tsne_perplexity`). Warns if `perplexity > N_valid/3` ("sklearn will clip") but still runs.
- `random_state (t-SNE)` — number_input, default 0 (key `pc_tsne_rs`).
- `run t-SNE` — button (key `pc_tsne_run`). **This** is the compute gate, not the checkbox. Compute-cost warning shows above when `est_s = (N/5000)*(perp/30)*60 > 30` OR `N_valid > 20000`.

## Cache key

`pc_tsne_emb__<tsne_hash>` where `tsne_hash = sha1((km_hash, perplexity, random_state))[:16]`. Embedding stored as a `(N_full, 2)` float32 array with NaN for rows excluded from the k-means fit (`_kmeans_labels < 0`). The currently-active embedding's hash is held in `st.session_state["pc_tsne_active_hash"]`, cleared when k-means is uncommitted.

## D < 2 fallback

If the k-means feature dim is 1 (e.g. `mech only` / `func only` / single-mech cossim / EI_1), sklearn's t-SNE requires ≥2D. The fallback skips the TSNE call, places the 1D values on the x-axis, sets y=0, and renders an `st.info` note inside the panel.

## Rendering

- Plain `go.Figure` (no subplots / marginals), one Scattergl trace per cluster, ordered by count DESC (largest first → smallest on top), `_extended_palette(k)` colors. Marker size `max(3, plot_dot_size - 1)`. Hover `seq_idx={idx} · cluster={i}`.
- Black-ringed centroid trace on top: fill = cluster palette color, size `max(12, plot_dot_size * 2)`, mean (x, y) per cluster over rows with finite t-SNE coords AND `_kmeans_labels == i`.
- Equal aspect ratio (`yaxis.scaleanchor="x"`), tick labels hidden, axis titles `tsne-1` / `tsne-2`. Same `plot_fig_w` / `plot_fig_h` as the main scatter.

## Click → row picker

`st.plotly_chart(..., on_select="rerun", key="tsne_chart__<active_hash>")`. Selected `customdata` (per-row `common_csv` index) feeds the same `sel_csv` used by the main scatter — t-SNE clicks drive the same per-row attribution-logo display below the scatter.
