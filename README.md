# k-CEE UI

Streamlit browser for pre-computed attribution maps. Click any point on the
scatter to view per-sequence sequence logos with optional FiNeMo motif
underlines.

## Install

```bash
cd ~/projects/kcee-ui
uv sync
```

## Run

```bash
uv run streamlit run app.py
```

## Sidebar

- **k-condition** — what the comparison is over.
  - `cell lines` (default): one model, A/B (and C in 3D) are different cell
    types. x-axis is `HepG2_log2FC − K562_log2FC`.
  - `models`: one cell type, A/B are different attribution sources (e.g.
    AlphaGenome vs MPRA-LegNet). x-axis is `pred(A) − pred(B)`.
- **Data source** (cell-lines mode): `AlphaGenome (standardtorch)` or
  `MPRA-LegNet`. Switches all slot defaults; per-source widget state is
  isolated.
- **Comparison** — `2D` (A/B) or `3D` (A/B/C, deviation-from-shared score).
- **Score (mech axis)** — `cossim` or `EigenMaps`. Both are computed live
  from importance maps (z-normalized over the 200-bp variable insert,
  adapters at positions 0–14 / 215–229 dropped). Matches
  `eigen-interactions/scripts/compare_eigenmaps.ipynb`.

## Plot controls (right of the scatter)

- colormap / vmin–vmax (auto / manual / full range)
- figure width and height
- marginal x/y histograms (counts / density / probability)
- color by: average magnitude · predicted activity · measured log2FC ·
  predicted − measured residual · score (y) · x-axis
- auto axis limits or manual xmin/xmax/ymin/ymax
- **highlight csv row** — type a CSV row index to draw a red ring over that
  point. The ring carries `customdata`, so clicking it loads the attribution
  logos like clicking the underlying point.
- show FiNeMo hits

## Input format

Each slot accepts an `.npz` or `.h5` with `(N, 4, L)` attribution arrays.

- AlphaGenome (standardtorch): keys `attr_{HepG2,K562,WTC11}` and
  `predictions_{HepG2,K562,WTC11}`, `L=230`.
- MPRA-LegNet (per-cell-type files): keys `attributions` and `predictions`,
  `L=230`.

## Logos

- Adapter regions (15 bp at each end) are trimmed before display and
  scoring; the variable region is 200 bp (positions 15–214).
- Logos render with `fast_logo` (cached A/C/G/T glyph paths via Matplotlib
  TextPath); first click warms the cache, repeated clicks use a
  `session_state` PNG cache.

## Tools

- `tools/npz_to_h5.py` — convert an `.npz` attribution file to a row-chunked
  `.h5` for true single-row reads (much faster click → logo).

  ```bash
  uv run python tools/npz_to_h5.py \
      --input  /path/to/attr.npz \
      --output /path/to/attr.h5
  ```
