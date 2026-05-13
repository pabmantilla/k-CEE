# K-means clustering (v0.13.0)

`cluster by k-means` checkbox lives in the `color` expander, after the `discretize into bins` block. Independent of bins — both can be active simultaneously.

## Features dropdown

First option is the placeholder `— pick features —` (no fit until a real option is picked). Real options:
- `(x, y)` — stack of active x_arr, y_arr.
- `mech only (y-axis value)` — 1D over y.
- `func only (x-axis value)` — 1D over x.
- `(x, y) + active color metric` — 3D; only offered when the current `color by` resolves to a continuous `_color_pool` entry.
- `single mech: pairwise cossim` — reuses `_cossim_full(...)` (A vs B), indexed by `common_csv`.
- `single mech: EI_1 / eigenmaps` — 2D mode → `_eigenmaps_full(...)`; 3D mode → `_dev_eig_full(weighted=True, ...)`.

## Fit scope

`X` is built over `common_csv` (the active filtered pool); rows where any selected feature column is non-finite get `-1` labels and are excluded from the fit. Optional column-wise standardisation (default on) with a `std == 0` guard. `KMeans(n_clusters=k, n_init=10, random_state=…).fit_predict(X)`.

## Cache key composition

Stored under `st.session_state["pc_kmeans_labels__<hash>"]` + `pc_kmeans_centroids__<hash>`. Hash inputs: features choice, k, standardize, random_state, x_axis tag, y_axis tag, color_mode tag (only when used), and a sha1 of the valid-row csv index set. Stable across UI filter changes that don't alter the input feature space.

## Interaction with bins

- bins committed AND k-means committed → bins drive color; k-means contributes centroid overlay + `cluster_kmeans` download column.
- bins NOT committed AND k-means committed → k-means drives discrete color via `_PALETTE` (extended via the existing HSL helper past 12) routed through the `_color_is_discrete` path. Per-cluster stacked marginals come for free.
- rows excluded from fit get `_NA_HEX`.

## Compute-cost warning

Before the `set clusters` button: if `N_valid * k > 200_000` OR `N_valid * D * k * 300 * 10 > 5e9`, render an `st.warning` advising the user; otherwise silent. `set clusters` is the gate — no separate "compute anyway" button.

## Centroid overlay

Drawn after the base scatter as a single Scattergl trace. Marker: black ring (`line.color=black, line.width=1.5`), size `max(12, dot_size*2)`, fill = cluster palette colour. Hover: `cluster {i} · n={count}`. Centroids live in feature space; projected to (x, y) for display — for 1D / single-mech features the missing axis is filled with the cluster-member mean from the visible rows.
