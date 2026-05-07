# k-CEE UI

Streamlit browser for pre-computed attribution maps from up to 3 models, with
a toggle between two scalar score modes:

- **cossim** — per-sequence cosine similarity between two model attribution stacks
- **eigenMaps** — per-sequence scalar from an `eigen_analysis.pkl` (default `EI_1 var x r = ei1_var * ratio`)

## Install

```bash
cd ~/projects/kcee-ui
uv sync
```

## Run

```bash
uv run streamlit run app.py
```

## Input format

Each model slot accepts an `.npz` or `.h5` file containing one or more 3D
arrays of shape `(N, 4, L)` (channels-first, AGCT). For each slot you pick:

- the file path
- the dataset key (e.g. `attr_K562`, `attr_HepG2` for the AlphaGenome npz, or
  `attributions` for the LegNet h5)
- a display name

Example sources used in the parent project:

- `genomic_targets/data/deeplift_attributions.npz` — keys `attr_K562`, `attr_HepG2`, `attr_WTC11`, shape `(N, 4, 281)`
- `legnet_rep/results/attrs_<ct>.h5` — key `attributions`, shape `(N, 4, 230)`

## eigenMaps mode

Point the sidebar at an `eigen_analysis.pkl` produced by the EigenMap pipeline
(see `eigen-interactions`). The default scoring key is `EI_1 var x r`; if not
present, the loader falls back to `ei1_var * ratio`.
