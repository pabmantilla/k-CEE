# GMM clustering (v0.13.3)

Adds Gaussian Mixture Model as a second method alongside k-means in the `color → cluster by kmeans/gmm` expander block.

## Controls

- **`method` radio (top of cluster block):** `kmeans` (default) / `gmm`. Key `pc_cluster_method`. Renders right after the `cluster by kmeans/gmm` checkbox opens.
- **`covariance_type` selectbox (gmm only):** `full` (default) / `tied` / `diag` / `spherical`. Key `pc_gmm_cov`. Rendered immediately after `random_state`. Pass-through empty string when method is `kmeans`.

## Fit call

`GaussianMixture(n_components=k, covariance_type=…, random_state=…, n_init=1).fit_predict(_Xfit)`. `n_init=1` matches sklearn's default — GMM `n_init>1` is much slower than KMeans's. Centroids come from `_km_model.means_` (same shape as `KMeans.cluster_centers_`, so the destandardisation step and downstream centroid plumbing are unchanged).

## What's shared with kmeans

- Session-state keys (`pc_kmeans_enable`, `pc_kmeans_k`, `pc_kmeans_standardize`, `pc_kmeans_rs`, `pc_kmeans_active`, `pc_kmeans_labels__<hash>`, `pc_kmeans_centroids__<hash>`).
- Cache key prefix `pc_kmeans_labels__` / `pc_kmeans_centroids__`; hash now includes `(method, gmm_cov)` so kmeans-only entries don't get invalidated when method is unchanged.
- Discrete-color override, centroid overlay, `cluster_label` download column, t-SNE diagnostic panel — all read `_kmeans_labels` / `_kmeans_centroids_feat` / `_kmeans_colors` / `_kmeans_k` which are populated regardless of method.
- Compute-cost warning threshold (GMM with `covariance_type=full` is roughly the same order of magnitude as KMeans for `D ≤ 3`).

## What's different

Only the fit call. No `predict_proba` / soft-assignments, no covariance ellipses — minimal version.

See [[kmeans_clustering]] and [[tsne_panel]] for shared plumbing details.
