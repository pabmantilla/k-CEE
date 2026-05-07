"""k-CEE attribution browser.

Streamlit UI to browse pre-computed AlphaGenome attribution maps.
2D or 3D comparison; click a point on the scatter to see logos with
finemo underlines.

Run:
    uv run streamlit run app.py
"""
import hashlib
from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from kcee_ui.loader import load_attr_file, list_attr_keys
from kcee_ui.scoring import cossim_score, eigenmaps_score, deviation_from_shared, attr_to_importance
from kcee_ui.plotting import plot_attribution, cached_attribution_png
from kcee_ui.data import load_library, seq_to_onehot
from kcee_ui.finemo import load_finemo_hits
from kcee_ui.defaults import (
    DEFAULT_SLOTS, DEFAULT_LIBRARY_CSV, DATA_SOURCES,
    MODEL_CT_OPTIONS, slots_for_cell_type,
)
from kcee_ui.cache import mmap_array, cached_npy, cache_dir, cache_size_mb, clear_cache, _mtime, load_attr_row


st.set_page_config(page_title="k-CEE attribution browser", layout="wide")
st.title("k-CEE attribution browser")

N_SLOTS = 3
ENHANCER_LEN = 230
NAME_MAX = 24


def short_name(name: str) -> str:
    name = str(name or "")
    return name if len(name) <= NAME_MAX else name[:NAME_MAX - 1] + "…"


# --- caches ---
@st.cache_data(show_spinner="extracting attributions to local cache (one-time)…")
def _cached_load(path: str, key: str) -> np.ndarray:
    """Returns a mmap'd .npy view. First call extracts; subsequent are instant."""
    return mmap_array(path, key)


@st.cache_data(show_spinner=False)
def _cached_keys(path: str) -> list[str]:
    return list_attr_keys(path)


@st.cache_data(show_spinner=False)
def _cached_attr_shape(path: str, key: str) -> tuple[int, ...]:
    """Cheap shape peek without materializing the array (used for n_attr)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            return tuple(f[key].shape)
    # npz: opening the archive is cheap; .shape on the array view is too,
    # but np.load loads the array. Use the mmap'd cache view instead so we
    # only pay the one-time extraction.
    return tuple(mmap_array(path, key).shape)


@st.cache_data(show_spinner=False)
def _cached_library(path: str) -> pd.DataFrame:
    return load_library(path)


@st.cache_data(show_spinner="building library one-hot…")
def _cached_csv_onehot(csv_path: str, attr_L: int) -> np.ndarray:
    """(N_csv, 4, attr_L) one-hot built from library['sequence']. NaN
    sequences become zero rows. Cached per (csv_path, attr_L)."""
    lib = _cached_library(csv_path)
    seqs = lib["sequence"].values
    n = len(seqs)
    out = np.zeros((n, 4, attr_L), dtype=np.float32)
    for i, s in enumerate(seqs):
        if isinstance(s, str) and s:
            out[i] = seq_to_onehot(s, length=attr_L)
    return out


@st.cache_data(show_spinner="projecting attributions to importance…")
def _cached_importance(path: str, key: str, csv_path: str,
                       map_hash: str, _csv_to_npz: np.ndarray) -> np.ndarray:
    """(N_npz, L) importance for a slot. Built from raw attribution and the
    library one-hot reordered into npz row order via attr_csv_to_npz."""
    deps = (path, key, _mtime(path), csv_path, map_hash)

    def _go():
        attr = np.asarray(_cached_load(path, key))  # (N_npz, 4, L)
        attr_L = int(attr.shape[2])
        oh_csv = _cached_csv_onehot(csv_path, attr_L)  # (N_csv, 4, L)
        n_npz = int(attr.shape[0])
        oh_npz = np.zeros((n_npz, 4, attr_L), dtype=np.float32)
        ok = _csv_to_npz >= 0
        csv_idx = np.nonzero(ok)[0]
        npz_idx = _csv_to_npz[ok]
        oh_npz[npz_idx] = oh_csv[csv_idx]
        return attr_to_importance(attr, oh_npz).astype(np.float32)

    return cached_npy("importance", deps, _go)


@st.cache_data(show_spinner=False)
def _load_pred(path: str, key: str) -> np.ndarray:
    if not key or not Path(path).exists():
        return np.array([])
    try:
        return np.asarray(mmap_array(path, key), dtype=np.float32)
    except (KeyError, ValueError):
        return np.array([])


@st.cache_data(show_spinner="loading finemo hits…")
def _cached_finemo(tsv_path: str) -> dict[int, list[dict]]:
    if not tsv_path or not Path(tsv_path).exists():
        return {}
    return load_finemo_hits(tsv_path)


# --- sidebar: k-condition ---
k_condition = st.sidebar.selectbox(
    "k-condition", ["cell lines", "models"], index=0, key="k_condition",
    help="'cell lines' compares attribution maps across cell lines for one model. "
         "'models' compares different models on the same cell line.",
)
MODE_MODELS = (k_condition == "models")


# --- sidebar: library ---
# Path is captured here at sidebar render; the actual CSV (18MB) is loaded
# lazily at first real use (mapping construction below) so empty-slot startups
# are cheap.
st.sidebar.header("Library")
csv_path = st.sidebar.text_input("library CSV", value=DEFAULT_LIBRARY_CSV)
library: pd.DataFrame | None = None
library_load_error: str | None = None


# --- sidebar: data source + model slots ---
models_ct: str | None = None
if MODE_MODELS:
    st.sidebar.header("Cell type")
    models_ct = st.sidebar.selectbox(
        "cell type", MODEL_CT_OPTIONS, index=0, key="models_ct",
        help="Eligible cell types have >=2 attribution sources.",
    )
    _source_slots = slots_for_cell_type(models_ct)
    _slot_key_tag = f"models_{models_ct}"
    st.sidebar.header("Available models")
    if len(_source_slots) < 2:
        st.sidebar.warning(f"Only {len(_source_slots)} source(s) for {models_ct}.")
else:
    st.sidebar.header("Data source")
    _source_names = list(DATA_SOURCES.keys())
    data_source = st.sidebar.selectbox("attribution dataset", _source_names, index=0, key="data_source")
    _source_slots = DATA_SOURCES[data_source]
    _slot_key_tag = data_source.split()[0].lower()  # "alphagenome" / "mpra-legnet"
    st.sidebar.header("Available models")

slots: list[dict] = []
n_show = len(_source_slots) if MODE_MODELS else max(N_SLOTS, len(_source_slots))
for i in range(n_show):
    d = _source_slots[i] if i < len(_source_slots) else {
        "cell_type": f"slot{i+1}", "model": "", "key": "", "pred_key": "",
        "log2fc_col": "", "path": "", "finemo_tsv": "",
    }
    _label = d.get("model") or d["cell_type"] if MODE_MODELS else d["cell_type"]
    with st.sidebar.expander(f"Slot {i + 1} — {_label}", expanded=(i < 2)):
        path = st.text_input("attr file (.npz / .h5)", value=d["path"], key=f"path_{_slot_key_tag}_{i}")
        finemo_tsv = st.text_input("finemo hits.tsv", value=d.get("finemo_tsv", ""), key=f"fm_{_slot_key_tag}_{i}")
        if not path or not Path(path).exists():
            slots.append({**d, "path": "", "predictions": None, "finemo": {}, "n_attr": 0})
            if path:
                st.warning("Path not found")
            continue
        try:
            keys = _cached_keys(path)
        except Exception as e:
            st.error(f"keys: {e}")
            slots.append({**d, "path": "", "predictions": None, "finemo": {}, "n_attr": 0})
            continue
        if not keys:
            st.warning("No 3D arrays found")
            slots.append({**d, "path": "", "predictions": None, "finemo": {}, "n_attr": 0})
            continue
        default_key_idx = keys.index(d["key"]) if d["key"] in keys else 0
        key = st.selectbox("attribution key", keys, index=default_key_idx, key=f"key_{_slot_key_tag}_{i}")
        name = st.text_input("display name", value=d["model"] or key, key=f"name_{_slot_key_tag}_{i}")
        # Defer heavy reads (predictions NPZ, finemo TSV) to first real use.
        # n_attr is needed to build the csv->npz row map; derive it cheaply
        # from the 3D attribution shape rather than loading predictions here.
        try:
            n_attr = int(_cached_attr_shape(path, key)[0])
        except Exception:
            n_attr = 0
        _ext = Path(path).suffix.lower()
        st.caption(f"attr N={n_attr if n_attr else '—'}  ·  format: {_ext or '—'}")
        slots.append({**d, "path": path, "key": key, "name": name,
                      "pred_key": d.get("pred_key", ""),
                      "finemo_tsv": finemo_tsv,
                      "n_attr": n_attr})

loaded = [s for s in slots if s.get("path")]


# --- per-slot mappings ---
# attr_csv_to_npz: csv_row -> row in deeplift_attributions.npz (seq_valid filter for
#   K562/HepG2 (56978), identity for WTC11 (56980)).
# finemo_csv_to_pid: csv_row -> peak_id in finemo hits.tsv. Built from the canonical
#   regions.npz (sibling of hits.tsv) which holds peak_name in finemo order; this
#   ordering differs from the deeplift order because regions.npz has additional
#   per-CT filtering (e.g. K562_log2FC.notna()).

def _build_attr_csv_to_npz(slot, library_df: pd.DataFrame) -> np.ndarray:
    n_attr = slot["n_attr"]
    n_csv = len(library_df)
    out = np.full(n_csv, -1, dtype=np.int64)
    if n_attr == 0:
        return out
    if n_attr == n_csv:
        out[:] = np.arange(n_csv)
        return out
    seq_valid = library_df["sequence"].notna().values
    if int(seq_valid.sum()) == n_attr:
        out[np.nonzero(seq_valid)[0]] = np.arange(n_attr)
        return out
    m = min(n_attr, n_csv)
    out[:m] = np.arange(m)
    return out


@st.cache_data(show_spinner=False)
def _cached_finemo_csv_to_pid(finemo_tsv_path: str, name_to_csv_keys: tuple[str, ...],
                               name_to_csv_vals: tuple[int, ...]) -> np.ndarray | None:
    """Build csv_row -> finemo peak_id using the regions.npz sibling of hits.tsv."""
    if not finemo_tsv_path:
        return None
    regions_path = Path(finemo_tsv_path).parent / "regions.npz"
    if not regions_path.exists():
        return None
    r = np.load(regions_path, allow_pickle=True)
    if "peak_name" not in r.files:
        return None
    peak_names = r["peak_name"]
    n_csv = len(name_to_csv_keys)
    n2c = dict(zip(name_to_csv_keys, name_to_csv_vals))
    pid_for_csv = np.full(n_csv, -1, dtype=np.int64)
    for pid, pn in enumerate(peak_names):
        c = n2c.get(str(pn))
        if c is not None:
            pid_for_csv[c] = pid
    return pid_for_csv


if loaded and csv_path and Path(csv_path).exists():
    try:
        library = _cached_library(csv_path)
        st.sidebar.caption(f"{len(library)} rows")
    except Exception as e:
        library_load_error = f"CSV load: {e}"
        st.sidebar.error(library_load_error)

if loaded and library is not None:
    for s in loaded:
        s["attr_csv_to_npz"] = _build_attr_csv_to_npz(s, library)
        s["covered_csv"] = np.nonzero(s["attr_csv_to_npz"] >= 0)[0]


# --- sidebar: comparison mode ---
st.sidebar.header("Comparison")
dim = st.sidebar.radio("dimension", ["2D", "3D"], horizontal=True)

ABC: list[int] = []  # indices into `loaded`
display_names = [short_name(s["name"]) for s in loaded]
if loaded:
    if dim == "2D":
        a_idx = st.sidebar.selectbox("A", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=0, key="abc_a")
        b_idx = st.sidebar.selectbox("B", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=min(1, len(loaded) - 1), key="abc_b")
        ABC = [a_idx, b_idx]
    else:
        if len(loaded) < 3:
            st.sidebar.warning("3D needs 3 loaded slots.")
        a_idx = st.sidebar.selectbox("A", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=0, key="abc_a3")
        b_idx = st.sidebar.selectbox("B", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=min(1, len(loaded) - 1), key="abc_b3")
        c_idx = st.sidebar.selectbox("C", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=min(2, len(loaded) - 1), key="abc_c3")
        ABC = [a_idx, b_idx, c_idx]


# --- score over common CSV rows ---
def _common_csv(slots_subset: list[dict]) -> np.ndarray:
    if not slots_subset:
        return np.array([], dtype=np.int64)
    return reduce(np.intersect1d, [s["covered_csv"] for s in slots_subset])


def _map_hash(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.int64).tobytes()).hexdigest()[:16]


@st.cache_data(show_spinner="computing cossim…")
def _cossim_full(path_a: str, key_a: str, path_b: str, key_b: str,
                 csv_path: str, a_map_hash: str, b_map_hash: str,
                 _a_map: np.ndarray, _b_map: np.ndarray) -> np.ndarray:
    """Per-CSV-row cossim on z-normalized importance over the enhancer.
    NaN where either slot doesn't cover the row."""
    deps = ("cossim_imp", path_a, key_a, _mtime(path_a),
            path_b, key_b, _mtime(path_b),
            csv_path, a_map_hash, b_map_hash)

    def _go():
        imp_a = _cached_importance(path_a, key_a, csv_path, a_map_hash, _a_map)
        imp_b = _cached_importance(path_b, key_b, csv_path, b_map_hash, _b_map)
        common = (_a_map >= 0) & (_b_map >= 0)
        out = np.full(_a_map.shape[0], np.nan, dtype=np.float32)
        if int(common.sum()) > 0:
            out[common] = cossim_score(imp_a[_a_map[common]], imp_b[_b_map[common]])
        return out

    return cached_npy("cossim_full", deps, _go)


@st.cache_data(show_spinner="computing eigenMaps…")
def _eigenmaps_full(path_a: str, key_a: str, path_b: str, key_b: str,
                    csv_path: str, a_map_hash: str, b_map_hash: str,
                    _a_map: np.ndarray, _b_map: np.ndarray) -> np.ndarray:
    """Per-CSV-row EigenMaps[var_ratio*r] on z-normalized importance over
    the enhancer. NaN where either slot doesn't cover the row."""
    deps = ("eig_imp", path_a, key_a, _mtime(path_a),
            path_b, key_b, _mtime(path_b),
            csv_path, a_map_hash, b_map_hash)

    def _go():
        imp_a = _cached_importance(path_a, key_a, csv_path, a_map_hash, _a_map)
        imp_b = _cached_importance(path_b, key_b, csv_path, b_map_hash, _b_map)
        common = (_a_map >= 0) & (_b_map >= 0)
        out = np.full(_a_map.shape[0], np.nan, dtype=np.float32)
        if int(common.sum()) > 0:
            out[common] = eigenmaps_score(imp_a[_a_map[common]], imp_b[_b_map[common]])
        return out

    return cached_npy("eigenmaps_full", deps, _go)


@st.cache_data(show_spinner="computing dev_from_shared…")
def _dev_full(paths: tuple[tuple[str, str], ...], map_hashes: tuple[str, ...],
              _maps: tuple[np.ndarray, ...]) -> np.ndarray:
    """Per-CSV-row deviation, NaN where any slot doesn't cover the row."""
    deps = tuple((p, k, _mtime(p), mh) for (p, k), mh in zip(paths, map_hashes)) + (len(_maps),)

    def _go():
        arrs_full = [np.asarray(_cached_load(p, k)) for (p, k) in paths]
        common = np.ones(_maps[0].shape[0], dtype=bool)
        for m in _maps:
            common &= (m >= 0)
        out = np.full(_maps[0].shape[0], np.nan, dtype=np.float32)
        if int(common.sum()) > 0:
            sub = [a[m[common]] for a, m in zip(arrs_full, _maps)]
            out[common] = deviation_from_shared(sub)
        return out

    return cached_npy("dev_full", deps, _go)


scores: np.ndarray | None = None  # aligned to `common_csv` (below)
score_label = "—"
score_cmap = "Viridis"
score_zmid: float | None = None
common_csv: np.ndarray = np.array([], dtype=np.int64)

if loaded and ABC and library is not None:
    sl_subset = [loaded[i] for i in ABC]
    common_csv = _common_csv(sl_subset)
    if dim == "2D":
        score_mode = st.sidebar.selectbox("score (mech axis)", ["cossim", "EigenMaps"], index=0)
        a, b = sl_subset[0], sl_subset[1]
        if ABC[0] == ABC[1]:
            st.sidebar.warning("A and B are the same slot.")
        elif score_mode == "cossim":
            try:
                a_map = a["attr_csv_to_npz"]
                b_map = b["attr_csv_to_npz"]
                scores_full = _cossim_full(
                    a["path"], a["key"], b["path"], b["key"],
                    csv_path,
                    _map_hash(a_map), _map_hash(b_map),
                    a_map, b_map,
                )
                scores = scores_full[common_csv]
                score_label = f"cossim({short_name(a['name'])}, {short_name(b['name'])})"
                score_cmap = "RdBu_r"
                score_zmid = 0.0
            except Exception as e:
                st.sidebar.error(str(e))
        else:  # eigenMaps
            try:
                a_map = a["attr_csv_to_npz"]
                b_map = b["attr_csv_to_npz"]
                scores_full = _eigenmaps_full(
                    a["path"], a["key"], b["path"], b["key"],
                    csv_path,
                    _map_hash(a_map), _map_hash(b_map),
                    a_map, b_map,
                )
                scores = scores_full[common_csv]
                score_label = "EigenMaps[var_ratio*r]"
                score_cmap = "Inferno"
            except Exception as e:
                st.sidebar.error(str(e))
    else:  # 3D
        if len(set(ABC)) >= 2:
            try:
                paths = tuple((s["path"], s["key"]) for s in sl_subset)
                maps = tuple(s["attr_csv_to_npz"] for s in sl_subset)
                map_hashes = tuple(_map_hash(m) for m in maps)
                scores_full = _dev_full(paths, map_hashes, maps)
                scores = scores_full[common_csv]
                score_label = f"dev_from_shared({', '.join(short_name(s['name']) for s in sl_subset)})"
                score_cmap = "Turbo"
            except Exception as e:
                st.sidebar.error(str(e))
        else:
            st.sidebar.warning("Pick at least 2 distinct slots.")


# --- sidebar: display options ---
st.sidebar.header("Display")
show_wt_logo = st.sidebar.checkbox("WT-projected logo (attr × onehot)", value=False)

# --- sidebar: cache ---
with st.sidebar.expander("Cache", expanded=False):
    st.caption(str(cache_dir()))
    st.caption(f"size: {cache_size_mb():.1f} MB")
    if st.button("Clear cache"):
        n = clear_cache()
        st.success(f"Cleared {n} files. Restart to rebuild.")


def _hits_to_local(hits, start_hg38, attr_L):
    """Per ctcf_focus.ipynb: x = hit.start - start_hg38. No pad, no strand mirror."""
    out = []
    if hits is None or not np.isfinite(start_hg38):
        return out
    peak_start = int(start_hg38)
    for h in hits:
        s = int(h["start"]) - peak_start
        e = int(h["end"]) - peak_start
        if 0 <= s < attr_L and e > s:
            out.append({**h, "start": s, "end": min(e, attr_L)})
    return out


# --- main: gating message ---
if not loaded or library is None:
    msg = "Load a library CSV and at least one model in the sidebar."
    if not loaded and library is not None:
        msg = "Load at least one model slot in the sidebar."
    elif loaded and library is None:
        msg = "Set the library CSV path in the sidebar."
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:center;
                    height:60vh;text-align:center;">
            <div style="font-size:1.2rem;color:#888;">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if scores is None or len(common_csv) == 0:
    st.info("Configure A/B (and C for 3D) plus a score in the sidebar.")
    st.stop()


# --- 2D scatter (CSV-row indexed) ---
if MODE_MODELS and len(ABC) >= 2:
    a = loaded[ABC[0]]
    b = loaded[ABC[1]]
    a_pred = _load_pred(a["path"], a["pred_key"]) if a.get("pred_key") else np.array([])
    b_pred = _load_pred(b["path"], b["pred_key"]) if b.get("pred_key") else np.array([])
    a_npz = a["attr_csv_to_npz"][common_csv]
    b_npz = b["attr_csv_to_npz"][common_csv]
    xa = np.full(common_csv.shape[0], np.nan, dtype=np.float32)
    xb = np.full(common_csv.shape[0], np.nan, dtype=np.float32)
    if a_pred.size:
        ok_a = (a_npz >= 0) & (a_npz < len(a_pred))
        if int(ok_a.sum()) > 0:
            xa[ok_a] = np.asarray(a_pred, dtype=np.float32)[a_npz[ok_a]]
    if b_pred.size:
        ok_b = (b_npz >= 0) & (b_npz < len(b_pred))
        if int(ok_b.sum()) > 0:
            xb[ok_b] = np.asarray(b_pred, dtype=np.float32)[b_npz[ok_b]]
    x = (xa - xb).astype(np.float32)
    _xaxis_label = f"pred({short_name(a['name'])}) − pred({short_name(b['name'])})   [func]"
else:
    hk_cols = library[["HepG2_log2FC", "K562_log2FC"]].iloc[common_csv].values
    x = hk_cols[:, 0] - hk_cols[:, 1]
    _xaxis_label = "log2FC (HepG2 / K562)   [func]"
y = scores


_HISTNORM = {"counts": "", "density": "density", "probability": "probability"}


def _binned_mean(values: np.ndarray, color: np.ndarray, bins: int = 30):
    finite = np.isfinite(values) & np.isfinite(color)
    v = values[finite]
    c = color[finite]
    if v.size == 0:
        return None
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).astype(np.float64)
    sums = np.bincount(idx, weights=c, minlength=bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    return centers, widths, counts, means


def _binned_counts(values: np.ndarray, bins: int = 30):
    finite = np.isfinite(values)
    v = values[finite]
    if v.size == 0:
        return None
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    return centers, widths, counts


@st.cache_resource(show_spinner=False)
def _scatter_fig(score_label: str, color_label: str, xaxis_label: str,
                 x_hash: str, y_hash: str, custom_hash: str, color_hash: str,
                 marg_color_hash: str,
                 colorscale: str, vmin: float | None, vmax: float | None,
                 fig_w: int, fig_h: int, marg_x: str, marg_y: str,
                 xmin: float | None, xmax: float | None,
                 ymin: float | None, ymax: float | None,
                 highlight_csv: int,
                 _x: np.ndarray, _y: np.ndarray, _custom: np.ndarray,
                 _color: np.ndarray, _marg_color: np.ndarray) -> go.Figure:
    has_marg = (marg_x != "none") or (marg_y != "none")
    if has_marg:
        fig = make_subplots(
            rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
            row_heights=[0.2, 0.8], column_widths=[0.8, 0.2],
            horizontal_spacing=0.02, vertical_spacing=0.02,
        )
        scatter_row, scatter_col = 2, 1
    else:
        fig = go.Figure()
        scatter_row = scatter_col = None

    marker = dict(
        size=4,
        color=_color,
        colorscale=colorscale,
        opacity=0.7,
        colorbar=dict(title=color_label),
    )
    if vmin is not None:
        marker["cmin"] = float(vmin)
    if vmax is not None:
        marker["cmax"] = float(vmax)

    scatter = go.Scattergl(
        x=_x,
        y=_y,
        mode="markers",
        marker=marker,
        customdata=_custom,
        hovertemplate=(
            "csv_row=%{customdata}<br>x=%{x:.3f}"
            "<br>mech=%{y:.3f}<br>color=%{marker.color:.3f}<extra></extra>"
        ),
        selected=dict(marker=dict(opacity=0.7)),
        unselected=dict(marker=dict(opacity=0.7)),
        name="",
    )

    if has_marg:
        fig.add_trace(scatter, row=scatter_row, col=scatter_col)
        if marg_x != "none":
            out = _binned_mean(_x, _marg_color, bins=30)
            if out is not None:
                centers, widths, counts, means = out
                hn = marg_x
                total = counts.sum() or 1.0
                if hn == "density":
                    yvals = counts / (total * widths)
                elif hn == "probability":
                    yvals = counts / total
                else:
                    yvals = counts
                fig.add_trace(
                    go.Bar(
                        x=centers, y=yvals, width=widths,
                        marker=dict(color=means, colorscale=colorscale,
                                    cmin=vmin, cmax=vmax, showscale=False),
                        showlegend=False, name="",
                    ),
                    row=1, col=1,
                )
            else:
                out2 = _binned_counts(_x, bins=30)
                if out2 is not None:
                    centers, widths, counts = out2
                    hn = marg_x
                    total = counts.sum() or 1.0
                    if hn == "density":
                        yvals = counts / (total * widths)
                    elif hn == "probability":
                        yvals = counts / total
                    else:
                        yvals = counts
                    fig.add_trace(
                        go.Bar(
                            x=centers, y=yvals, width=widths,
                            marker_color="#888",
                            showlegend=False, name="",
                        ),
                        row=1, col=1,
                    )
        if marg_y != "none":
            out = _binned_mean(_y, _marg_color, bins=30)
            if out is not None:
                centers, widths, counts, means = out
                hn = marg_y
                total = counts.sum() or 1.0
                if hn == "density":
                    xvals = counts / (total * widths)
                elif hn == "probability":
                    xvals = counts / total
                else:
                    xvals = counts
                fig.add_trace(
                    go.Bar(
                        x=xvals, y=centers, width=widths, orientation='h',
                        marker=dict(color=means, colorscale=colorscale,
                                    cmin=vmin, cmax=vmax, showscale=False),
                        showlegend=False, name="",
                    ),
                    row=2, col=2,
                )
            else:
                out2 = _binned_counts(_y, bins=30)
                if out2 is not None:
                    centers, widths, counts = out2
                    hn = marg_y
                    total = counts.sum() or 1.0
                    if hn == "density":
                        xvals = counts / (total * widths)
                    elif hn == "probability":
                        xvals = counts / total
                    else:
                        xvals = counts
                    fig.add_trace(
                        go.Bar(
                            x=xvals, y=centers, width=widths, orientation='h',
                            marker_color="#888",
                            showlegend=False, name="",
                        ),
                        row=2, col=2,
                    )
        fig.update_xaxes(title_text=xaxis_label, row=2, col=1)
        fig.update_yaxes(title_text=f"{score_label}   [mech]", row=2, col=1)
    else:
        fig.add_trace(scatter)
        fig.update_layout(
            xaxis_title=xaxis_label,
            yaxis_title=f"{score_label}   [mech]",
        )

    fig.update_layout(
        width=int(fig_w),
        height=int(fig_h),
        margin=dict(l=40, r=20, t=30, b=40),
        template="plotly_white",
        dragmode="zoom",
    )
    if xmin is not None and xmax is not None:
        if has_marg:
            fig.update_xaxes(range=[xmin, xmax], row=2, col=1)
        else:
            fig.update_xaxes(range=[xmin, xmax])
    if ymin is not None and ymax is not None:
        if has_marg:
            fig.update_yaxes(range=[ymin, ymax], row=2, col=1)
        else:
            fig.update_yaxes(range=[ymin, ymax])
    if highlight_csv is not None and highlight_csv >= 0:
        idx = np.where(_custom == int(highlight_csv))[0]
        if idx.size:
            j = int(idx[0])
            hl = go.Scatter(
                x=[float(_x[j])], y=[float(_y[j])],
                mode="markers",
                marker=dict(size=16, color="rgba(0,0,0,0)",
                            line=dict(color="#e63946", width=3)),
                customdata=[int(highlight_csv)],
                hovertemplate=f"csv_row={int(highlight_csv)}<extra></extra>",
                showlegend=False, name="highlight",
            )
            if has_marg:
                fig.add_trace(hl, row=2, col=1)
            else:
                fig.add_trace(hl)
    return fig


_ctrl, _center = st.columns([1, 4])

with _ctrl:
    with st.container(border=True):
        st.markdown("**Plot controls**")
        plot_colorscale = st.selectbox(
            "colorscale",
            ["Viridis", "Plasma", "Magma", "Turbo", "Cividis", "RdBu_r", "Inferno"],
            index=0,
            key="pc_cmap",
        )
        plot_clip_mode = st.radio(
            "color clipping",
            ["auto (2–98%)", "manual", "full range"],
            index=0,
            key="pc_clip",
        )
        plot_vmin: float | None = None
        plot_vmax: float | None = None
        if plot_clip_mode == "manual":
            plot_vmin = float(st.number_input("vmin", value=0.0, key="pc_vmin"))
            plot_vmax = float(st.number_input("vmax", value=1.0, key="pc_vmax"))
        pred_cts = [s["name"] for s in (loaded[i] for i in ABC) if s.get("pred_key")]
        if MODE_MODELS:
            meas_cts = [models_ct] if (models_ct and f"{models_ct}_log2FC" in library.columns) else []
            resid_cts = [s["name"] for s in (loaded[i] for i in ABC)
                         if s.get("pred_key") and models_ct
                         and f"{models_ct}_log2FC" in library.columns]
        else:
            meas_cts = [s["cell_type"] for s in (loaded[i] for i in ABC)
                        if f"{s['cell_type']}_log2FC" in library.columns]
            resid_cts = [s["cell_type"] for s in (loaded[i] for i in ABC)
                         if s.get("pred_key") and f"{s['cell_type']}_log2FC" in library.columns]
        _ax_kind = "model" if MODE_MODELS else "cell line"
        _xaxis_color_label = (
            "model prediction difference (x-axis)"
            if MODE_MODELS else "log2FC HepG2 vs K562 (x-axis)"
        )
        _COLOR_MODES = [
            "average magnitude",
            f"predicted activity ({_ax_kind})",
            f"measured log2FC ({'cell line' if not MODE_MODELS else 'cell line'})",
            f"predicted − measured residual ({_ax_kind})",
            "score (y-axis)",
            _xaxis_color_label,
        ]
        color_mode = st.selectbox(
            "color by",
            _COLOR_MODES,
            index=0,
            key="pc_color",
        )
        # Normalize color_mode to canonical keys used in the resolution branches.
        _CM_PRED = f"predicted activity ({_ax_kind})"
        _CM_MEAS = f"measured log2FC ({'cell line' if not MODE_MODELS else 'cell line'})"
        _CM_RESID = f"predicted − measured residual ({_ax_kind})"
        _CM_XAXIS = _xaxis_color_label
        if color_mode == _CM_PRED and not pred_cts:
            color_mode = "average magnitude"
        elif color_mode == _CM_MEAS and not meas_cts:
            color_mode = "average magnitude"
        elif color_mode == _CM_RESID and not resid_cts:
            color_mode = "average magnitude"
        color_cell_line = None
        if color_mode == _CM_PRED and pred_cts:
            color_cell_line = st.selectbox(_ax_kind, pred_cts, index=0, key="pc_color_ct_pred")
        elif color_mode == _CM_MEAS and meas_cts:
            color_cell_line = st.selectbox("cell line", meas_cts, index=0, key="pc_color_ct_meas")
        elif color_mode == _CM_RESID and resid_cts:
            color_cell_line = st.selectbox(_ax_kind, resid_cts, index=0, key="pc_color_ct_resid")
        plot_fig_w = int(st.slider("figure width (px)", 400, 2400, 900, 50, key="pc_w"))
        plot_fig_h = int(st.slider("figure height (px)", 300, 1200, 600, 50, key="pc_h"))
        _MARG = ["none", "counts", "density", "probability"]
        plot_marg_x = st.selectbox("marginal x", _MARG, index=0, key="pc_mx")
        plot_marg_y = st.selectbox("marginal y", _MARG, index=0, key="pc_my")
        plot_auto_lims = st.checkbox("auto axis limits", value=True, key="pc_autolims")
        plot_xmin: float | None = None
        plot_xmax: float | None = None
        plot_ymin: float | None = None
        plot_ymax: float | None = None
        if not plot_auto_lims:
            _xc1, _xc2 = st.columns(2)
            plot_xmin = float(_xc1.number_input("xmin", value=-3.0, step=0.1, key="pc_xmin"))
            plot_xmax = float(_xc2.number_input("xmax", value= 3.0, step=0.1, key="pc_xmax"))
            _yc1, _yc2 = st.columns(2)
            plot_ymin = float(_yc1.number_input("ymin", value=-1.0, step=0.1, key="pc_ymin"))
            plot_ymax = float(_yc2.number_input("ymax", value= 1.0, step=0.1, key="pc_ymax"))
        show_finemo_hits = st.checkbox("show FiNeMo hits", value=False, key="pc_finemo")
        highlight_csv = int(st.number_input("highlight csv row", value=-1, step=1, key="pc_highlight",
                                            help="-1 to disable; otherwise show a marker at this row"))

# Build the shared color array (aligned to common_csv) driven by the
# unified "color by" control. Both the scatter and the marginals use
# this same array by design.
_color_arr = None
color_label = "—"

def _slot_by_name(name):
    return next((s for s in (loaded[i] for i in ABC) if s.get("name") == name), None)

def _slot_by_ct(ct):
    return next((s for s in (loaded[i] for i in ABC) if s.get("cell_type") == ct), None)

def _predicted_for(slot):
    if slot is None or not slot.get("pred_key"):
        return None
    preds = _load_pred(slot["path"], slot["pred_key"])
    if preds is None or not getattr(preds, "size", 0):
        return None
    npz_idx = slot["attr_csv_to_npz"][common_csv]
    out = np.full(common_csv.shape[0], np.nan, dtype=np.float32)
    ok = (npz_idx >= 0) & (npz_idx < len(preds))
    if int(ok.sum()) > 0:
        out[ok] = np.asarray(preds, dtype=np.float32)[npz_idx[ok]]
    return out

def _measured_for(ct):
    col = f"{ct}_log2FC"
    if col not in library.columns:
        return None
    return library[col].iloc[common_csv].to_numpy(dtype=np.float32, copy=False)

if color_mode == _CM_PRED and color_cell_line:
    arr = _predicted_for(_slot_by_name(color_cell_line))
    if arr is not None:
        _color_arr = arr
        color_label = f"predicted ({short_name(color_cell_line)})"

elif color_mode == _CM_MEAS and color_cell_line:
    arr = _measured_for(color_cell_line)
    if arr is not None:
        _color_arr = arr
        color_label = f"measured log2FC ({color_cell_line})"

elif color_mode == _CM_RESID and color_cell_line:
    if MODE_MODELS:
        pred = _predicted_for(_slot_by_name(color_cell_line))
        meas = _measured_for(models_ct)
        resid_tag = f"{short_name(color_cell_line)} − {models_ct}"
    else:
        pred = _predicted_for(_slot_by_ct(color_cell_line))
        meas = _measured_for(color_cell_line)
        resid_tag = color_cell_line
    if pred is not None and meas is not None:
        _color_arr = (pred - meas).astype(np.float32)
        color_label = f"residual ({resid_tag})"

elif color_mode == "score (y-axis)":
    _color_arr = np.asarray(scores, dtype=np.float32)
    color_label = score_label

elif color_mode == _CM_XAXIS:
    _color_arr = np.asarray(x, dtype=np.float32)
    color_label = _xaxis_label.replace("   [func]", "").strip()

if _color_arr is None:
    sl_subset_for_color = [loaded[i] for i in ABC]
    _color_stack = []
    _color_slot_names: list[str] = []
    for s in sl_subset_for_color:
        arr = _predicted_for(s)
        if arr is None:
            continue
        _color_stack.append(arr)
        _color_slot_names.append(short_name(s["name"]))
    if _color_stack:
        _color_arr = np.nanmean(np.vstack(_color_stack), axis=0)
        color_label = f"mean activity ({', '.join(_color_slot_names)})"
    else:
        _color_arr = np.full(common_csv.shape[0], np.nan, dtype=np.float32)
        color_label = "mean activity (n/a)"

valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(_color_arr)

_x = np.ascontiguousarray(x[valid], dtype=np.float64)
_y = np.ascontiguousarray(y[valid], dtype=np.float64)
_custom = np.ascontiguousarray(common_csv[valid], dtype=np.int64)
_color = np.ascontiguousarray(_color_arr[valid], dtype=np.float64)
_marg_color = _color

# Resolve vmin/vmax per the clipping mode.
if plot_clip_mode == "auto (2–98%)":
    if _color.size and np.isfinite(_color).any():
        _lo, _hi = np.nanpercentile(_color, [2, 98])
        _vmin, _vmax = float(_lo), float(_hi)
    else:
        _vmin = _vmax = None
elif plot_clip_mode == "manual":
    _vmin, _vmax = plot_vmin, plot_vmax
else:  # full range
    if _color.size and np.isfinite(_color).any():
        _vmin = float(np.nanmin(_color))
        _vmax = float(np.nanmax(_color))
    else:
        _vmin = _vmax = None

fig = _scatter_fig(
    score_label, color_label, _xaxis_label,
    hashlib.sha1(_x.tobytes()).hexdigest()[:16],
    hashlib.sha1(_y.tobytes()).hexdigest()[:16],
    hashlib.sha1(_custom.tobytes()).hexdigest()[:16],
    hashlib.sha1(_color.tobytes()).hexdigest()[:16],
    hashlib.sha1(_marg_color.tobytes()).hexdigest()[:16],
    plot_colorscale, _vmin, _vmax,
    plot_fig_w, plot_fig_h, plot_marg_x, plot_marg_y,
    plot_xmin, plot_xmax, plot_ymin, plot_ymax,
    int(highlight_csv),
    _x, _y, _custom, _color, _marg_color,
)

with _center:
    event = st.plotly_chart(
        fig,
        use_container_width=False,
        on_select="rerun",
        selection_mode=("points",),
        key="scatter",
    )


# --- selected point: csv row ---
sel_csv: int | None = None
sel = getattr(event, "selection", None) if event is not None else None
if sel and sel.get("points"):
    pts = [p for p in sel["points"] if p.get("customdata") is not None]
    if pts:
        pt = pts[0]
        if pt.get("customdata") is not None:
            sel_csv = int(pt["customdata"])

st.markdown("---")
if sel_csv is None:
    st.info("Click a point above to display its attribution logos.")
    st.stop()


# --- predicted vs measured ---
display_slots = [loaded[i] for i in ABC]
row = library.iloc[sel_csv]
display_full_name = str(row.get("name", ""))
st.markdown(f"### `{short_name(display_full_name)}`  · csv row `{sel_csv}`")
if len(display_full_name) > NAME_MAX:
    st.caption(display_full_name)

cols = st.columns(len(display_slots))
for col, s in zip(cols, display_slots):
    npz_idx = int(s["attr_csv_to_npz"][sel_csv])
    preds = _load_pred(s["path"], s["pred_key"]) if s.get("pred_key") else np.array([])
    pred = float(preds[npz_idx]) if preds.size and 0 <= npz_idx < len(preds) else float("nan")
    meas_col = s.get("log2fc_col", "")
    meas = float(row[meas_col]) if meas_col and meas_col in library.columns else float("nan")
    with col:
        st.markdown(f"**{s['cell_type']}**  ·  _{short_name(s['name'])}_")
        c1, c2 = st.columns(2)
        c1.metric("predicted", f"{pred:.3f}" if np.isfinite(pred) else "—")
        c2.metric("measured (log2FC)", f"{meas:.3f}" if np.isfinite(meas) else "—")


# --- attribution logos ---
seq_full = str(row.get("sequence", "") or "")


def _hits_signature(hits: list[dict]) -> tuple:
    """Hashable summary of a finemo hit list for figure-cache keying."""
    return tuple((int(h.get("start", -1)), int(h.get("end", -1)), str(h.get("motif", "")))
                 for h in (hits or []))


st.markdown("#### Attribution logos")
_need_finemo = any(s.get("finemo_tsv") for s in display_slots)
if _need_finemo:
    _name_keys = tuple(library["name"].astype(str).tolist())
    _name_vals = tuple(range(len(library)))
for s in display_slots:
    npz_idx = int(s["attr_csv_to_npz"][sel_csv])
    if npz_idx < 0:
        st.warning(f"{short_name(s['name'])}: no attribution for this CSV row.")
        continue
    try:
        attr_shape = _cached_attr_shape(s["path"], s["key"])
    except Exception as e:
        st.error(f"{short_name(s['name'])}: {e}")
        continue
    if npz_idx >= attr_shape[0]:
        st.warning(f"{short_name(s['name'])}: npz idx {npz_idx} out of range.")
        continue
    attr_L = attr_shape[2]
    wt_oh = seq_to_onehot(seq_full, length=attr_L) if show_wt_logo else None
    # Lazy-load finemo only here, on click.
    fm_path = s.get("finemo_tsv", "")
    fm = _cached_finemo(fm_path) if fm_path else {}
    fm_path_for_pid = s.get("finemo_tsv", "")
    if fm_path_for_pid:
        pid_map = _cached_finemo_csv_to_pid(fm_path_for_pid, _name_keys, _name_vals)
    else:
        pid_map = None
    if pid_map is not None:
        finemo_pid = int(pid_map[sel_csv])
    else:
        finemo_pid = npz_idx  # fallback (only correct when npz and finemo orderings agree)
    raw_hits = fm.get(finemo_pid, []) if fm and finemo_pid >= 0 else []
    hits = _hits_to_local(
        raw_hits,
        float(row.get("start_hg38", float("nan"))),
        attr_L,
    )
    plot_hits = hits if show_finemo_hits else []
    title = f"{s['cell_type']} · {short_name(s['name'])}"
    if hits and show_finemo_hits:
        title += f"  ·  {len(hits)} finemo hits"
    try:
        attr_row = load_attr_row(s["path"], s["key"], npz_idx)
        png = cached_attribution_png(
            path=s["path"], key=s["key"], idx=int(npz_idx),
            hits_signature=(_hits_signature(plot_hits), bool(show_finemo_hits)), show_wt_logo=show_wt_logo,
            attr=attr_row, wt_onehot=wt_oh, hits=plot_hits, title=title,
            proj_only_first=ENHANCER_LEN,
        )
    except Exception as e:
        st.error(f"{short_name(s['name'])}: {e}")
        continue
    st.image(png, use_container_width=True)
