"""k-CEE attribution browser.

Streamlit UI to browse pre-computed AlphaGenome attribution maps.
2D or 3D comparison; click a point on the scatter to see logos with
finemo underlines.

Run:
    uv run streamlit run app.py
"""
import hashlib
import re
from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from kcee_ui.loader import load_attr_file, list_attr_keys
from kcee_ui.scoring import cossim_score, eigenmaps_score, deviation_from_shared, dev_from_shared_eig, attr_to_importance
from kcee_ui.plotting import plot_attribution, cached_attribution_png
from kcee_ui.data import load_library, seq_to_onehot
from kcee_ui.finemo import load_finemo_hits
from kcee_ui.defaults import (
    DEFAULT_LIBRARY_CSV, infer_insert_offset, METHOD_DISPLAY,
    family_names, family_slug, methods_for_family,
    families_with_multiple_methods_anywhere, cts_for_family_with_multiple_methods,
    slots_for_family_method, slots_for_ct_method, slots_for_family_ct,
    cts_eligible_for_models_mode, methods_at_ct_across_families,
)
from kcee_ui.cache import mmap_array, cached_npy, cache_dir, cache_size_mb, clear_cache, _mtime, load_attr_row
from kcee_ui.alignment import csv_to_npz_for_slot, assert_slot_aligned, assert_pair_aligned, AlignmentError


st.set_page_config(page_title="k-CEE attribution browser", layout="wide")

# Center the default Streamlit "Running…" status widget and replace its
# contents with a large circular CSS spinner so the load-up indicator is
# obvious instead of a tiny top-right badge / thin white bar.
st.markdown(
    """
    <style>
    @keyframes kcee-spin { to { transform: rotate(360deg); } }
    [data-testid="stStatusWidget"] {
        position: fixed !important;
        top: 50% !important;
        left: 50% !important;
        right: auto !important;
        transform: translate(-50%, -50%) !important;
        z-index: 999999 !important;
        background: rgba(255, 255, 255, 0.97) !important;
        padding: 28px 36px 22px !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.20) !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
        min-width: 160px !important;
    }
    /* Hide the default tiny icon/text Streamlit puts inside the widget. */
    [data-testid="stStatusWidget"] > div { all: unset !important; }
    [data-testid="stStatusWidget"] svg,
    [data-testid="stStatusWidget"] button { display: none !important; }
    /* The circular spinner itself. */
    [data-testid="stStatusWidget"]::before {
        content: "";
        display: block;
        width: 56px;
        height: 56px;
        border: 6px solid #e6e6e6;
        border-top-color: #ff4b4b;
        border-radius: 50%;
        animation: kcee-spin 0.9s linear infinite;
    }
    [data-testid="stStatusWidget"]::after {
        content: "Loading…";
        font-size: 0.95rem;
        color: #444;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_title_col, _mode_col = st.columns([5, 1])
with _title_col:
    st.title("k-CEE attribution browser")
    # Filled after the sidebar pickers resolve so the tag reflects the current
    # (family / cell type / method) selection.
    _title_tag_slot = st.empty()
with _mode_col:
    st.write("")  # vertical padding so the radio lines up with the title row
    app_mode = st.radio(
        "mode", ["kcee", "SEAM"], index=0, horizontal=True, key="app_mode",
        label_visibility="collapsed",
        help="kcee: per-row attribution browser across families/methods/cell lines. "
             "SEAM: foreground/background viewer for the 1059-seq SEAM space (Pablo's AG models, HepG2+K562 only).",
    )

if app_mode == "SEAM":
    from kcee_ui import seam
    seam.render()
    st.stop()

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
def _cached_csv_onehot(csv_path: str, attr_L: int, insert_offset: int) -> np.ndarray:
    """(N_csv, 4, attr_L) one-hot built from library['sequence'], where
    column 0 corresponds to insert position `insert_offset`. NaN sequences
    become zero rows. Cached per (csv_path, attr_L, insert_offset)."""
    lib = _cached_library(csv_path)
    seqs = lib["sequence"].values
    n = len(seqs)
    out = np.zeros((n, 4, attr_L), dtype=np.float32)
    for i, s in enumerate(seqs):
        if isinstance(s, str) and s:
            out[i] = seq_to_onehot(s, length=attr_L, offset=insert_offset)
    return out


@st.cache_data(show_spinner="projecting attributions to importance…")
def _cached_importance(path: str, key: str, csv_path: str,
                       map_hash: str, insert_offset: int,
                       _csv_to_npz: np.ndarray) -> np.ndarray:
    """(N_npz, L) importance for a slot. Built from raw attribution and the
    library one-hot reordered into npz row order via attr_csv_to_npz.
    `insert_offset` aligns the library 230bp insert with the saved attribution
    coordinates (see defaults.py and .ui-guy/wt_alignment.md)."""
    deps = (path, key, _mtime(path), csv_path, map_hash, int(insert_offset), "v2")

    def _go():
        attr = np.asarray(_cached_load(path, key))  # (N_npz, 4, L)
        attr_L = int(attr.shape[2])
        oh_csv = _cached_csv_onehot(csv_path, attr_L, int(insert_offset))  # (N_csv, 4, L)
        n_npz = int(attr.shape[0])
        oh_npz = np.zeros((n_npz, 4, attr_L), dtype=np.float32)
        ok = _csv_to_npz >= 0
        csv_idx = np.nonzero(ok)[0]
        npz_idx = _csv_to_npz[ok]
        oh_npz[npz_idx] = oh_csv[csv_idx]
        # Replace NaN attribution rows with zeros so NaN doesn't poison
        # downstream scores. (See .ui-guy/nan_rows.md.) The per-row plot path
        # detects NaN separately and surfaces a warning.
        if not np.isfinite(attr).all():
            attr = np.where(np.isfinite(attr), attr, np.float32(0.0))
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
    "k-condition", ["cell lines", "models", "methods"], index=0, key="k_condition",
    help="'cell lines' fixes (family, method), varies cell line. "
         "'models' fixes (cell line, method), varies model family. "
         "'methods' fixes (family, cell line), varies attribution method.",
)
# Single-cell-type modes share downstream label/axis logic with the original
# 'models' mode; cell-lines mode is the odd one out.
MODE_MODELS = (k_condition != "cell lines")


# --- sidebar: library ---
# Path is captured here at sidebar render; the actual CSV (18MB) is loaded
# lazily at first real use (mapping construction below) so empty-slot startups
# are cheap.
st.sidebar.header("Library")
csv_path = st.sidebar.text_input("library CSV", value=DEFAULT_LIBRARY_CSV)
library: pd.DataFrame | None = None
library_load_error: str | None = None


# --- sidebar: family / cell type / method pickers ---
# Each k-condition fixes two of (family, cell type, method) via dropdowns and
# leaves the third varying — that's the A/B/C axis. Pickers are filtered to
# combinations whose H5 actually exists on disk; method is not a "data source"
# but an axis, so the per-method Koo entries no longer appear at the top level.
models_ct: str | None = None
_source_slots: list[dict] = []
_slot_key_tag: str = "empty"
_title_tag: str = ""
if k_condition == "cell lines":
    fams = family_names()
    if not fams:
        st.sidebar.error("No attribution families found on disk.")
    else:
        st.sidebar.header("Family")
        family = st.sidebar.selectbox("model family", fams, index=0, key="family_cl")
        method_opts = methods_for_family(family)
        st.sidebar.header("Method")
        method = st.sidebar.selectbox(
            "attribution method", method_opts, index=0,
            format_func=lambda m: METHOD_DISPLAY.get(m, m),
            key=f"method_cl__{family_slug(family)}",
        )
        _source_slots = slots_for_family_method(family, method)
        _slot_key_tag = f"cl__{family_slug(family)}__{method}"
        _title_tag = f"{family} · {METHOD_DISPLAY.get(method, method)} · comparing cell lines"
elif k_condition == "models":
    cts = cts_eligible_for_models_mode()
    if not cts:
        st.sidebar.warning("No (cell type, method) pair has >=2 families yet.")
    else:
        st.sidebar.header("Cell type")
        models_ct = st.sidebar.selectbox(
            "cell type", cts, index=0, key="models_ct",
            help="Cell types with >=1 method present in >=2 families.",
        )
        method_opts = methods_at_ct_across_families().get(models_ct, [])
        st.sidebar.header("Method")
        method = st.sidebar.selectbox(
            "attribution method", method_opts, index=0,
            format_func=lambda m: METHOD_DISPLAY.get(m, m),
            key=f"method_mdl__{models_ct}",
        )
        _source_slots = slots_for_ct_method(models_ct, method)
        _slot_key_tag = f"mdl__{models_ct}__{method}"
        _title_tag = f"{models_ct} · {METHOD_DISPLAY.get(method, method)} · comparing models"
        if len(_source_slots) < 2:
            st.sidebar.warning(f"Only {len(_source_slots)} family at {models_ct}/{METHOD_DISPLAY.get(method, method)}.")
else:  # methods
    fams = families_with_multiple_methods_anywhere()
    if not fams:
        st.sidebar.warning("No family has >=2 methods on disk yet.")
    else:
        st.sidebar.header("Family")
        family = st.sidebar.selectbox("model family", fams, index=0, key="family_mth")
        ct_opts = cts_for_family_with_multiple_methods(family)
        st.sidebar.header("Cell type")
        ct = st.sidebar.selectbox(
            "cell type", ct_opts, index=0,
            key=f"ct_mth__{family_slug(family)}",
        )
        models_ct = ct  # downstream meas/resid logic keys off models_ct
        _source_slots = slots_for_family_ct(family, ct)
        _slot_key_tag = f"mth__{family_slug(family)}__{ct}"
        _title_tag = f"{family} · {ct} · comparing methods"
if _title_tag:
    _title_tag_slot.caption(_title_tag)

slots: list[dict] = []
for d in _source_slots:
    path = d.get("path", "")
    if not path or not Path(path).exists():
        slots.append({**d, "path": "", "predictions": None, "finemo": {}, "n_attr": 0})
        continue
    try:
        keys = _cached_keys(path)
    except Exception as e:
        st.sidebar.error(f"{d.get('model') or d.get('cell_type')}: keys: {e}")
        slots.append({**d, "path": "", "predictions": None, "finemo": {}, "n_attr": 0})
        continue
    if not keys:
        slots.append({**d, "path": "", "predictions": None, "finemo": {}, "n_attr": 0})
        continue
    key = d["key"] if d.get("key") in keys else keys[0]
    name = d.get("model") or key
    try:
        n_attr = int(_cached_attr_shape(path, key)[0])
    except Exception:
        n_attr = 0
    slots.append({**d, "path": path, "key": key, "name": name,
                  "pred_key": d.get("pred_key", ""),
                  "finemo_tsv": d.get("finemo_tsv", ""),
                  "n_attr": n_attr})

loaded = [s for s in slots if s.get("path")]
if loaded:
    st.sidebar.caption("Loaded: " + ", ".join(short_name(s["name"]) for s in loaded))


# --- per-slot mappings ---
# attr_csv_to_npz: csv_row -> row in deeplift_attributions.npz (seq_valid filter for
#   K562/HepG2 (56978), identity for WTC11 (56980)).
# finemo_csv_to_pid: csv_row -> peak_id in finemo hits.tsv. Built from the canonical
#   regions.npz (sibling of hits.tsv) which holds peak_name in finemo order; this
#   ordering differs from the deeplift order because regions.npz has additional
#   per-CT filtering (e.g. K562_log2FC.notna()).

def _build_attr_csv_to_npz(slot, library_df: pd.DataFrame) -> np.ndarray:
    """Delegate to kcee_ui.alignment so unknown drop policies raise instead of
    silently aliasing rows. Old behavior aliased LegNet (n=56975) onto rows
    [0..56974] of the CSV, misaligning row 18321 onwards."""
    return csv_to_npz_for_slot(slot, library_df, strict=True)


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
        try:
            s["attr_csv_to_npz"] = _build_attr_csv_to_npz(s, library)
            assert_slot_aligned(s, library, s["attr_csv_to_npz"])
        except AlignmentError as e:
            st.sidebar.error(f"alignment failure: {e}")
            s["attr_csv_to_npz"] = np.full(len(library), -1, dtype=np.int64)
        s["covered_csv"] = np.nonzero(s["attr_csv_to_npz"] >= 0)[0]


# --- sidebar: comparison mode ---
st.sidebar.header("Comparison")
dim = st.sidebar.radio("dimension", ["2D", "3D"], horizontal=True)

ABC: list[int] = []  # indices into `loaded`
# Per-mode label for the A/B/C selectboxes: the varying axis ("cell line" /
# "family" / "method"). Downstream code still reads s["name"] for plot titles.
def _picker_label(s: dict) -> str:
    if k_condition == "cell lines":
        return s.get("cell_type") or short_name(s["name"])
    if k_condition == "models":
        return s.get("family") or short_name(s["name"])
    if k_condition == "methods":
        return METHOD_DISPLAY.get(s.get("method"), s.get("method") or short_name(s["name"]))
    return short_name(s["name"])
display_names = [_picker_label(s) for s in loaded]
if loaded:
    # Re-key per source/cell-type tag so changing the dropdown above resets the
    # A/B/C selection to the new defaults instead of holding a stale integer
    # in session_state (which caused A/B to look frozen until clicked).
    if dim == "2D":
        a_idx = st.sidebar.selectbox("A", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=0, key=f"abc_a__{_slot_key_tag}")
        b_idx = st.sidebar.selectbox("B", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=min(1, len(loaded) - 1), key=f"abc_b__{_slot_key_tag}")
        ABC = [a_idx, b_idx]
    else:
        if len(loaded) < 3:
            st.sidebar.warning("3D needs 3 loaded slots.")
        a_idx = st.sidebar.selectbox("A", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=0, key=f"abc_a3__{_slot_key_tag}")
        b_idx = st.sidebar.selectbox("B", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=min(1, len(loaded) - 1), key=f"abc_b3__{_slot_key_tag}")
        c_idx = st.sidebar.selectbox("C", range(len(loaded)), format_func=lambda i: display_names[i],
                                     index=min(2, len(loaded) - 1), key=f"abc_c3__{_slot_key_tag}")
        ABC = [a_idx, b_idx, c_idx]


# Plot controls container is created here so the library-category picker (which
# must run before scoring to populate annot_filter_mask) and the rest of the
# plot controls (rendered after scoring) share the same bordered panel.
_ctrl, _center = st.columns([1, 4])
_pc = _ctrl.container(border=True)
_pc.markdown("**Plot controls**")

# --- plot controls: library annotation (color / filter by any library column) ---
# Pick any library CSV column (e.g. `category` with values like 'promoter',
# 'putative enhancer, HepG2') to (a) filter common_csv to a subset and/or
# (b) color the scatter by that column. Filter applies BEFORE scoring so
# scores stay aligned to the filtered common_csv.
annot_col: str | None = None
annot_filter_mask: np.ndarray | None = None  # bool, length = len(library)
annot_codes: np.ndarray | None = None        # int32, length = len(library); -1 if NaN
annot_values: list[str] = []                 # ordered unique values; index = code
if library is not None:
    _eligible_annot_cols = [
        c for c in library.columns
        if c not in ("sequence", "csv_row", "name")
        and not c.endswith("_log2FC")
        and not c.endswith("_hg38")
        and library[c].nunique(dropna=True) <= 200
    ]
    if _eligible_annot_cols:
        _pc.markdown("**library category**")
        annot_col = _pc.selectbox(
            "column", ["(none)"] + _eligible_annot_cols, index=0,
            key=f"annot_col__{_slot_key_tag}",
            help="Library CSV column for filter/color (e.g. 'category').",
        )
        if annot_col == "(none)":
            annot_col = None
        if annot_col is not None:
            annot_values = sorted(library[annot_col].dropna().astype(str).unique().tolist())
            _picked = _pc.multiselect(
                f"filter {annot_col} (empty = all)",
                annot_values, default=[],
                key=f"annot_vals__{annot_col}",
            )
            _vi = {v: i for i, v in enumerate(annot_values)}
            annot_codes = (
                library[annot_col].astype("string").map(_vi).fillna(-1).astype("int32").to_numpy()
            )
            if _picked:
                _picked_codes = np.array([_vi[v] for v in _picked if v in _vi], dtype=np.int32)
                annot_filter_mask = np.isin(annot_codes, _picked_codes)


# --- score over common CSV rows ---
def _common_csv(slots_subset: list[dict]) -> np.ndarray:
    if not slots_subset:
        return np.array([], dtype=np.int64)
    return reduce(np.intersect1d, [s["covered_csv"] for s in slots_subset])


def _map_hash(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.int64).tobytes()).hexdigest()[:16]


def _var_window(insert_offset: int, attr_L: int) -> tuple[int, int]:
    """Var region (200 bp = insert[15:215]) expressed in attribution-array
    coordinates given a slot's insert_offset and attribution length.

    Koo lab / LegNet (var-only saved): insert_offset=15, attr_L=200 -> (0, 200).
    Pablo full construct:              insert_offset=0,  attr_L=281 -> (15, 215).
    Pablo var-only regen (when it lands at attr_L=200): insert_offset=15 -> (0, 200).
    """
    lo = max(0, 15 - int(insert_offset))
    hi = min(int(attr_L), lo + 200)
    return lo, hi


def _resolve_insert_offset(slot: dict, attr_L: int) -> int:
    """Pick insert_offset per slot. attr_L (read from the file) takes
    precedence: a slot whose static `insert_offset` predicts a different
    layout than the file actually has gets corrected at runtime so we don't
    silently misalign when an attribution file is regenerated with a new
    layout (e.g. Pablo models switching 281 -> 200).
    """
    inferred = infer_insert_offset(int(attr_L))
    declared = slot.get("insert_offset")
    if declared is None:
        return inferred
    declared = int(declared)
    if declared != inferred:
        return inferred
    return declared


def _slot_var_window(slot: dict) -> tuple[int, int]:
    attr_L = int(_cached_attr_shape(slot["path"], slot["key"])[2])
    return _var_window(_resolve_insert_offset(slot, attr_L), attr_L)


@st.cache_data(show_spinner="computing cossim…")
def _cossim_full(path_a: str, key_a: str, path_b: str, key_b: str,
                 csv_path: str, a_map_hash: str, b_map_hash: str,
                 ins_off_a: int, ins_off_b: int,
                 _a_map: np.ndarray, _b_map: np.ndarray) -> np.ndarray:
    """Per-CSV-row cossim on z-normalized importance over the var region
    (the slice is computed per-slot from insert_offset). NaN where either
    slot doesn't cover the row."""
    deps = ("cossim_imp", path_a, key_a, _mtime(path_a),
            path_b, key_b, _mtime(path_b),
            csv_path, a_map_hash, b_map_hash,
            int(ins_off_a), int(ins_off_b), "v2")

    def _go():
        assert_pair_aligned(("a", _a_map), ("b", _b_map))
        imp_a = _cached_importance(path_a, key_a, csv_path, a_map_hash, int(ins_off_a), _a_map)
        imp_b = _cached_importance(path_b, key_b, csv_path, b_map_hash, int(ins_off_b), _b_map)
        a_lo, a_hi = _var_window(int(ins_off_a), int(imp_a.shape[1]))
        b_lo, b_hi = _var_window(int(ins_off_b), int(imp_b.shape[1]))
        common = (_a_map >= 0) & (_b_map >= 0)
        out = np.full(_a_map.shape[0], np.nan, dtype=np.float32)
        if int(common.sum()) > 0:
            sub_a = imp_a[_a_map[common]][:, a_lo:a_hi]
            sub_b = imp_b[_b_map[common]][:, b_lo:b_hi]
            out[common] = cossim_score(sub_a, sub_b, start=0, stop=min(sub_a.shape[1], sub_b.shape[1]))
        return out

    return cached_npy("cossim_full", deps, _go)


@st.cache_data(show_spinner="computing eigenMaps…")
def _eigenmaps_full(path_a: str, key_a: str, path_b: str, key_b: str,
                    csv_path: str, a_map_hash: str, b_map_hash: str,
                    ins_off_a: int, ins_off_b: int,
                    _a_map: np.ndarray, _b_map: np.ndarray) -> np.ndarray:
    """Per-CSV-row EigenMaps[var_ratio*r] on z-normalized importance over
    the var region (per-slot). NaN where either slot doesn't cover the row."""
    deps = ("eig_imp", path_a, key_a, _mtime(path_a),
            path_b, key_b, _mtime(path_b),
            csv_path, a_map_hash, b_map_hash,
            int(ins_off_a), int(ins_off_b), "v2")

    def _go():
        assert_pair_aligned(("a", _a_map), ("b", _b_map))
        imp_a = _cached_importance(path_a, key_a, csv_path, a_map_hash, int(ins_off_a), _a_map)
        imp_b = _cached_importance(path_b, key_b, csv_path, b_map_hash, int(ins_off_b), _b_map)
        a_lo, a_hi = _var_window(int(ins_off_a), int(imp_a.shape[1]))
        b_lo, b_hi = _var_window(int(ins_off_b), int(imp_b.shape[1]))
        common = (_a_map >= 0) & (_b_map >= 0)
        out = np.full(_a_map.shape[0], np.nan, dtype=np.float32)
        if int(common.sum()) > 0:
            sub_a = imp_a[_a_map[common]][:, a_lo:a_hi]
            sub_b = imp_b[_b_map[common]][:, b_lo:b_hi]
            out[common] = eigenmaps_score(sub_a, sub_b, start=0, stop=min(sub_a.shape[1], sub_b.shape[1]))
        return out

    return cached_npy("eigenmaps_full", deps, _go)


@st.cache_data(show_spinner="computing dev_from_shared…")
def _dev_full(paths: tuple[tuple[str, str], ...], csv_path: str,
              map_hashes: tuple[str, ...], insert_offsets: tuple[int, ...],
              _maps: tuple[np.ndarray, ...]) -> np.ndarray:
    """Per-CSV-row deviation on WT×attr importance over the var region
    (sliced per-slot from insert_offset). NaN where any slot doesn't cover
    the row. Matches cossim/eigenmaps: (N, L) importance, not (N, 4, L)."""
    deps = ("dev_imp",) + tuple((p, k, _mtime(p), mh, int(off))
                                 for (p, k), mh, off in zip(paths, map_hashes, insert_offsets)) \
                       + (csv_path, len(_maps), "v3")

    def _go():
        assert_pair_aligned(*[(f"slot{i}", m) for i, m in enumerate(_maps)])
        imps_full: list[np.ndarray] = []
        for (p, k), mh, off, m in zip(paths, map_hashes, insert_offsets, _maps):
            imp = _cached_importance(p, k, csv_path, mh, int(off), m)
            lo, hi = _var_window(int(off), int(imp.shape[1]))
            imps_full.append(imp[:, lo:hi])
        common = np.ones(_maps[0].shape[0], dtype=bool)
        for m in _maps:
            common &= (m >= 0)
        out = np.full(_maps[0].shape[0], np.nan, dtype=np.float32)
        if int(common.sum()) > 0:
            sub = [imp[m[common]] for imp, m in zip(imps_full, _maps)]
            # NaN rows -> 0 so they don't poison the shared-direction calc.
            sub = [np.where(np.isfinite(x), x, np.float32(0.0)) for x in sub]
            # Trim to the shortest var window so all stacks line up.
            min_L = min(s.shape[1] for s in sub)
            sub = [s[:, :min_L] for s in sub]
            out[common] = deviation_from_shared(sub)
        return out

    return cached_npy("dev_full", deps, _go)


@st.cache_data(show_spinner="computing deviation from shared (eig)…")
def _dev_eig_full(paths: tuple[tuple[str, str], ...], csv_path: str,
                  map_hashes: tuple[str, ...], insert_offsets: tuple[int, ...],
                  weighted: bool,
                  _maps: tuple[np.ndarray, ...]) -> np.ndarray:
    """Per-CSV-row deviation-from-shared via eigendecomposition of the
    per-sequence cell-type-by-cell-type covariance of z-normalized importance.
    `weighted=False` -> cossim-style (unweighted); `weighted=True` -> EigenMaps-
    style (multiplied by var_ratio_1). Range [0, 1] either way."""
    deps = ("dev_eig",) + tuple((p, k, _mtime(p), mh, int(off))
                                 for (p, k), mh, off in zip(paths, map_hashes, insert_offsets)) \
                       + (csv_path, len(_maps), bool(weighted), "v1")

    def _go():
        assert_pair_aligned(*[(f"slot{i}", m) for i, m in enumerate(_maps)])
        imps_full: list[np.ndarray] = []
        for (p, k), mh, off, m in zip(paths, map_hashes, insert_offsets, _maps):
            imp = _cached_importance(p, k, csv_path, mh, int(off), m)
            lo, hi = _var_window(int(off), int(imp.shape[1]))
            imps_full.append(imp[:, lo:hi])
        common = np.ones(_maps[0].shape[0], dtype=bool)
        for m in _maps:
            common &= (m >= 0)
        out = np.full(_maps[0].shape[0], np.nan, dtype=np.float32)
        if int(common.sum()) > 0:
            sub = [imp[m[common]] for imp, m in zip(imps_full, _maps)]
            sub = [np.where(np.isfinite(x), x, np.float32(0.0)) for x in sub]
            min_L = min(s.shape[1] for s in sub)
            sub = [s[:, :min_L] for s in sub]
            out[common] = dev_from_shared_eig(sub, weighted=bool(weighted))
        return out

    return cached_npy("dev_eig_full", deps, _go)


scores: np.ndarray | None = None  # aligned to `common_csv` (below)
score_label = "—"
score_cmap = "Viridis"
score_zmid: float | None = None
common_csv: np.ndarray = np.array([], dtype=np.int64)

if loaded and ABC and library is not None:
    sl_subset = [loaded[i] for i in ABC]
    common_csv = _common_csv(sl_subset)
    if annot_filter_mask is not None and len(common_csv):
        common_csv = common_csv[annot_filter_mask[common_csv]]
    if dim == "2D":
        score_mode = st.sidebar.selectbox("score (mech axis)", ["cossim", "EigenMaps"], index=0)
        a, b = sl_subset[0], sl_subset[1]
        if ABC[0] == ABC[1]:
            st.sidebar.warning("A and B are the same slot.")
        elif score_mode == "cossim":
            try:
                a_map = a["attr_csv_to_npz"]
                b_map = b["attr_csv_to_npz"]
                a_L = int(_cached_attr_shape(a["path"], a["key"])[2])
                b_L = int(_cached_attr_shape(b["path"], b["key"])[2])
                scores_full = _cossim_full(
                    a["path"], a["key"], b["path"], b["key"],
                    csv_path,
                    _map_hash(a_map), _map_hash(b_map),
                    _resolve_insert_offset(a, a_L), _resolve_insert_offset(b, b_L),
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
                a_L = int(_cached_attr_shape(a["path"], a["key"])[2])
                b_L = int(_cached_attr_shape(b["path"], b["key"])[2])
                scores_full = _eigenmaps_full(
                    a["path"], a["key"], b["path"], b["key"],
                    csv_path,
                    _map_hash(a_map), _map_hash(b_map),
                    _resolve_insert_offset(a, a_L), _resolve_insert_offset(b, b_L),
                    a_map, b_map,
                )
                scores = scores_full[common_csv]
                score_label = "EigenMaps[var_ratio*r]"
                score_cmap = "Inferno"
            except Exception as e:
                st.sidebar.error(str(e))
    else:  # 3D — mech axis is "deviation from shared" via eigendecomposition
        score_mode = st.sidebar.selectbox(
            "score (mech axis)", ["cossim", "EigenMaps"],
            index=0, key="score_mode_3d",
            help=(
                "Deviation from shared: eigendecompose per-sequence covariance of z-normalised "
                "importance across the 3 cell types, return 1 - |EI_1 · shared_dir|. "
                "EigenMaps weights this by var_ratio_1 (matches the 2D eigenmaps_score = var_ratio*r convention)."
            ),
        )
        if len(set(ABC)) >= 2:
            try:
                paths = tuple((s["path"], s["key"]) for s in sl_subset)
                maps = tuple(s["attr_csv_to_npz"] for s in sl_subset)
                map_hashes = tuple(_map_hash(m) for m in maps)
                insert_offsets = tuple(
                    _resolve_insert_offset(s, int(_cached_attr_shape(s["path"], s["key"])[2]))
                    for s in sl_subset
                )
                weighted = (score_mode == "EigenMaps")
                scores_full = _dev_eig_full(paths, csv_path, map_hashes, insert_offsets,
                                            weighted, maps)
                scores = scores_full[common_csv]
                _names = ", ".join(short_name(s["name"]) for s in sl_subset)
                score_label = f"deviation from shared [{score_mode}]({_names})"
                score_cmap = "Inferno" if weighted else "Magma"
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


def _hits_to_local(hits, start_hg38, attr_L, insert_offset: int = 0):
    """Convert finemo hits (genomic coords) to attribution-array coords.

    `start_hg38` marks the genomic start of the 200bp variable region (per the
    library CSV — stop_hg38-start_hg38 ~= 200). So `hit.start - start_hg38`
    gives the hit position in var coords. To land in attribution-array coords
    we add `(15 - insert_offset)` (the var offset within the saved attribution):
        Koo lab / LegNet (insert_offset=15): shift = 0  (attr is var-only)
        Pablo (insert_offset=0):              shift = 15 (attr is full construct)
    Hits outside [0, attr_L) are dropped.
    """
    out = []
    if hits is None or not np.isfinite(start_hg38):
        return out
    peak_start = int(start_hg38)
    var_in_attr = 15 - int(insert_offset)
    for h in hits:
        s = int(h["start"]) - peak_start + var_in_attr
        e = int(h["end"]) - peak_start + var_in_attr
        if e <= 0 or s >= attr_L:
            continue
        out.append({**h, "start": max(0, s), "end": min(e, attr_L)})
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
if k_condition == "methods" and len(ABC) >= 2:
    # Same model, same cell line — predictions are identical, so there's no
    # functional axis. Collapse x to 0 to make the methods scatter purely
    # attribution-driven.
    x = np.zeros(common_csv.shape[0], dtype=np.float32)
    _xaxis_label = "(no functional axis — predictions identical across methods)"
elif MODE_MODELS and len(ABC) >= 2:
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
                 dragmode: str, boxes_hash: str,
                 dot_size: int,
                 discrete: bool,
                 _x: np.ndarray, _y: np.ndarray, _custom: np.ndarray,
                 _color: np.ndarray, _marg_color: np.ndarray,
                 _boxes: list | None = None) -> go.Figure:
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

    if discrete:
        marker = dict(size=dot_size, color=_color, opacity=0.7)
        _hover = "csv_row=%{customdata}<br>x=%{x:.3f}<br>mech=%{y:.3f}<extra></extra>"
    else:
        marker = dict(
            size=dot_size,
            color=_color,
            colorscale=colorscale,
            opacity=0.7,
            colorbar=dict(title=dict(text=color_label, side="right")),
        )
        if vmin is not None:
            marker["cmin"] = float(vmin)
        if vmax is not None:
            marker["cmax"] = float(vmax)
        _hover = (
            "csv_row=%{customdata}<br>x=%{x:.3f}"
            "<br>mech=%{y:.3f}<br>color=%{marker.color:.3f}<extra></extra>"
        )

    scatter = go.Scattergl(
        x=_x,
        y=_y,
        mode="markers",
        marker=marker,
        customdata=_custom,
        hovertemplate=_hover,
        selected=dict(marker=dict(opacity=0.7)),
        unselected=dict(marker=dict(opacity=0.7)),
        name="",
    )

    if has_marg:
        fig.add_trace(scatter, row=scatter_row, col=scatter_col)
        if marg_x != "none":
            out = None if discrete else _binned_mean(_x, _marg_color, bins=30)
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
            out = None if discrete else _binned_mean(_y, _marg_color, bins=30)
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
        dragmode=dragmode,
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
    for _i, b in enumerate(_boxes or []):
        shape_kw = dict(
            type="rect", x0=b["x0"], y0=b["y0"], x1=b["x1"], y1=b["y1"],
            line=dict(color=b["color"], width=2),
            fillcolor=b["color"], opacity=0.10, layer="above",
        )
        if has_marg:
            fig.add_shape(row=2, col=1, **shape_kw)
        else:
            fig.add_shape(**shape_kw)
        m = (_x >= b["x0"]) & (_x <= b["x1"]) & (_y >= b["y0"]) & (_y <= b["y1"])
        if int(m.sum()) > 0:
            overlay = go.Scattergl(
                x=_x[m], y=_y[m], mode="markers",
                marker=dict(size=dot_size, color=b["color"], opacity=0.9),
                customdata=_custom[m],
                hovertemplate="csv_row=%{customdata}<extra></extra>",
                showlegend=False, name=f"box {_i + 1}",
            )
            if has_marg:
                fig.add_trace(overlay, row=2, col=1)
            else:
                fig.add_trace(overlay)
    return fig


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

with _pc:
    # Colorscale follows the per-score-type default (RdBu_r / Inferno / Magma).
    plot_colorscale = score_cmap
    plot_dot_size = int(st.slider("dot size", 1, 12, 4, 1, key="pc_dot_size"))
    # Build the x/y axis chooser pool. Each option maps to an array aligned to common_csv.
    _axis_pool: dict[str, np.ndarray | None] = {"auto": None}
    for s in (loaded[i] for i in ABC):
        col = f"{s['cell_type']}_log2FC"
        if col in library.columns:
            _axis_pool[f"measured {s['cell_type']}_log2FC"] = (
                library[col].iloc[common_csv].to_numpy(dtype=np.float32, copy=False)
            )
    for s in (loaded[i] for i in ABC):
        if s.get("pred_key"):
            _ppred = _predicted_for(s)
            if _ppred is not None:
                _axis_pool[f"predicted ({short_name(s['name'])})"] = _ppred
    _axis_pool["score (mech axis)"] = np.asarray(scores, dtype=np.float32)
    if "HepG2_log2FC" in library.columns and "K562_log2FC" in library.columns:
        _hk = library[["HepG2_log2FC", "K562_log2FC"]].iloc[common_csv].values
        _axis_pool["log2FC HepG2 − K562"] = (_hk[:, 0] - _hk[:, 1]).astype(np.float32)
    _abc_slots = [loaded[i] for i in ABC]
    if len(_abc_slots) >= 2 and _abc_slots[0].get("pred_key") and _abc_slots[1].get("pred_key"):
        _pa = _predicted_for(_abc_slots[0])
        _pb = _predicted_for(_abc_slots[1])
        if _pa is not None and _pb is not None:
            _axis_pool["pred(A) − pred(B)"] = (_pa - _pb).astype(np.float32)
    _axis_opts = list(_axis_pool.keys())
    x_axis_choice = st.selectbox("x-axis", _axis_opts, index=0, key="pc_xaxis")
    y_axis_choice = st.selectbox("y-axis", _axis_opts, index=0, key="pc_yaxis")

    # Unified color pool: continuous entries reuse axis arrays; residual entries
    # are pred-minus-measured; library categorical columns produce discrete entries.
    _color_pool: dict[str, tuple] = {}
    for _k, _v in _axis_pool.items():
        if _k == "auto" or _v is None:
            continue
        _color_pool[_k] = ("continuous", _v)
    # residuals: one per ABC slot with pred_key + matching log2FC column
    for s in (loaded[i] for i in ABC):
        if not s.get("pred_key"):
            continue
        if MODE_MODELS:
            if not models_ct or f"{models_ct}_log2FC" not in library.columns:
                continue
            _pred = _predicted_for(s)
            _meas = _measured_for(models_ct)
            _tag = f"{short_name(s['name'])} − {models_ct}"
        else:
            _ct = s.get("cell_type")
            if not _ct or f"{_ct}_log2FC" not in library.columns:
                continue
            _pred = _predicted_for(s)
            _meas = _measured_for(_ct)
            _tag = f"{short_name(s['name'])} − {_ct}"
        if _pred is None or _meas is None:
            continue
        _color_pool[f"residual ({_tag})"] = ("continuous", (_pred - _meas).astype(np.float32))
    # legacy "average magnitude" — mean of predictions across ABC
    _avg_stack = []
    for s in (loaded[i] for i in ABC):
        _ap = _predicted_for(s)
        if _ap is not None:
            _avg_stack.append(_ap)
    if _avg_stack:
        _color_pool["average magnitude"] = (
            "continuous", np.nanmean(np.vstack(_avg_stack), axis=0).astype(np.float32),
        )
    # discrete library columns: ≤20 unique non-null values, eligible columns only
    _PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
                "#CC79A7", "#999999", "#882255", "#117733", "#88CCEE", "#AA4499"]
    _NA_HEX = "#BDBDBD"
    if library is not None:
        for _col in library.columns:
            if _col in ("sequence", "csv_row", "name"):
                continue
            if _col.endswith("_log2FC") or _col.endswith("_hg38"):
                continue
            _vals = library[_col].dropna()
            _nu = _vals.nunique()
            if _nu == 0 or _nu > 20:
                continue
            _cats = sorted(_vals.astype(str).unique().tolist())
            _palette_map = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(_cats)}
            _per_row_cat = library[_col].astype("string").iloc[common_csv].fillna("NA").to_numpy()
            _per_row_hex = np.array(
                [_palette_map.get(str(v), _NA_HEX) for v in _per_row_cat], dtype="<U7"
            )
            _color_pool[f"library: {_col}"] = ("discrete", _per_row_hex, _palette_map)

    _color_opts = ["auto (mean activity)"] + list(_color_pool.keys())
    color_mode = st.selectbox("color by", _color_opts, index=0, key="pc_color")

    _color_arr = None
    _color_is_discrete = False
    _color_legend: list[tuple[str, str]] = []
    color_label = "—"
    if color_mode == "auto (mean activity)":
        if "average magnitude" in _color_pool:
            _color_arr = _color_pool["average magnitude"][1]
            color_label = "mean activity"
        else:
            _color_arr = np.full(common_csv.shape[0], np.nan, dtype=np.float32)
            color_label = "mean activity (n/a)"
    else:
        _entry = _color_pool.get(color_mode)
        if _entry is not None and _entry[0] == "continuous":
            _color_arr = np.asarray(_entry[1], dtype=np.float32)
            color_label = color_mode
        elif _entry is not None and _entry[0] == "discrete":
            _color_arr = _entry[1]
            _color_is_discrete = True
            color_label = color_mode
            _color_legend = [(k, v) for k, v in _entry[2].items()]
    if _color_is_discrete and _color_legend:
        _rows = []
        for _cat, _hex in _color_legend[:12]:
            _txt = str(_cat)
            if len(_txt) > 28:
                _txt = _txt[:27] + "…"
            _rows.append(
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"background:{_hex};border-radius:2px;margin-right:4px;'></span>{_txt}"
            )
        st.markdown("<br>".join(_rows), unsafe_allow_html=True)
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

# Apply x/y axis overrides BEFORE the `valid` mask is built so finite-filtering
# sees the user-chosen arrays. Axis titles are also updated to reflect the choice.
if x_axis_choice != "auto" and _axis_pool.get(x_axis_choice) is not None:
    x = np.asarray(_axis_pool[x_axis_choice], dtype=np.float32)
    _xaxis_label = x_axis_choice
if y_axis_choice != "auto" and _axis_pool.get(y_axis_choice) is not None:
    y = np.asarray(_axis_pool[y_axis_choice], dtype=np.float32)
    score_label = y_axis_choice

# Shared color array (aligned to common_csv) was resolved inside _pc above.
# Discrete colors are per-row hex strings; continuous is a numeric array.
if _color_is_discrete:
    valid = np.isfinite(x) & np.isfinite(y)
else:
    if _color_arr is None:
        _color_arr = np.full(common_csv.shape[0], np.nan, dtype=np.float32)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(_color_arr)

_x = np.ascontiguousarray(x[valid], dtype=np.float64)
_y = np.ascontiguousarray(y[valid], dtype=np.float64)
_custom = np.ascontiguousarray(common_csv[valid], dtype=np.int64)
if _color_is_discrete:
    _color = np.ascontiguousarray(_color_arr[valid])
    _marg_color = _color
    _vmin = _vmax = None
else:
    _color = np.ascontiguousarray(_color_arr[valid], dtype=np.float64)
    _marg_color = _color
    if _color.size and np.isfinite(_color).any():
        _lo, _hi = np.nanpercentile(_color, [2, 98])
        _vmin, _vmax = float(_lo), float(_hi)
    else:
        _vmin = _vmax = None

st.session_state.setdefault("isolate_mode", False)
st.session_state.setdefault("isolate_boxes", [])
st.session_state.setdefault("isolate_next_id", 1)
st.session_state.setdefault("isolate_last_box_hash", None)
_BOX_PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
                "#0072B2", "#D55E00", "#CC79A7", "#999999"]
_isolate_mode = bool(st.session_state["isolate_mode"])
_boxes = list(st.session_state["isolate_boxes"])
_dragmode = "select" if _isolate_mode else "zoom"
_boxes_hash = hashlib.sha1(
    repr([(b["_uid"], round(b["x0"], 6), round(b["x1"], 6),
           round(b["y0"], 6), round(b["y1"], 6), b["color"]) for b in _boxes]).encode()
).hexdigest()[:16]

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
    _dragmode, _boxes_hash,
    int(plot_dot_size),
    bool(_color_is_discrete),
    _x, _y, _custom, _color, _marg_color,
    _boxes=_boxes,
)

with _center:
    _tb_left, _tb_mid, _tb_right = st.columns([1, 6, 2])
    with _tb_left:
        st.toggle("isolate", key="isolate_mode")
    with _tb_mid:
        if _boxes:
            _chip_cols = st.columns([1] * len(_boxes) + [2])
            for _ci, _b in enumerate(_boxes):
                _bm = ((_x >= _b["x0"]) & (_x <= _b["x1"]) &
                       (_y >= _b["y0"]) & (_y <= _b["y1"]))
                _bn = int(_bm.sum())
                with _chip_cols[_ci]:
                    st.markdown(
                        f"<span style='display:inline-block;width:10px;height:10px;"
                        f"background:{_b['color']};border-radius:2px;margin-right:4px;'></span>"
                        f"box {_ci + 1} · {_bn}",
                        unsafe_allow_html=True,
                    )
                    # Delete uses the stable internal _uid (not the displayed
                    # 1-based position) so widget keys don't collide and
                    # numbering renumbers naturally after deletion.
                    if st.button("✕", key=f"del_box_{_b['_uid']}"):
                        st.session_state["isolate_boxes"] = [
                            x for x in st.session_state["isolate_boxes"] if x["_uid"] != _b["_uid"]
                        ]
                        st.rerun()
    with _tb_right:
        _no_boxes = (len(_boxes) == 0)
        with st.popover("download lib", disabled=_no_boxes):
            _box_opts = [b["_uid"] for b in _boxes]
            _uid_to_pos = {b["_uid"]: i + 1 for i, b in enumerate(_boxes)}
            _box_labels = {
                b["_uid"]: f"box {i + 1} · {int(((_x >= b['x0']) & (_x <= b['x1']) & (_y >= b['y0']) & (_y <= b['y1'])).sum())} rows · {b['color']}"
                for i, b in enumerate(_boxes)
            }
            _sel_box_uids = st.multiselect(
                "include boxes", options=_box_opts,
                default=_box_opts,
                format_func=lambda u: _box_labels.get(u, str(u)),
                key="dl_boxes",
            )
            # Compact column scheme: rename verbose library columns + per-slot prediction
            # arrays for every selected (ABC) slot. Defaults match the "input-like"
            # set (seq_idx, name, chr, start, stop, sequence) + predictions.
            _rename = {"chr_hg38": "chr", "start_hg38": "start", "stop_hg38": "stop"}
            _lib_cols = ["name", "chr_hg38", "start_hg38", "stop_hg38", "sequence",
                         "HepG2_log2FC", "K562_log2FC", "WTC11_log2FC", "category", "str_hg38"]
            _lib_cols = [c for c in _lib_cols if c in library.columns]
            _pred_arrays: dict[str, np.ndarray] = {}
            # Per-slot full hypothetical attribution; resolved lazily per-box.
            _attr_slots: dict[str, dict] = {}
            for s in (loaded[i] for i in ABC):
                _sname = short_name(s["name"])
                if s.get("pred_key"):
                    _p = _predicted_for(s)
                    if _p is not None:
                        _pred_arrays[f"pred_{_sname}"] = _p
                _attr_slots[f"attr_{_sname}"] = s
            _all_opts = ["seq_idx"] + _lib_cols + list(_pred_arrays.keys()) \
                      + list(_attr_slots.keys())
            _default_cols = ["seq_idx", "name", "chr_hg38", "start_hg38", "stop_hg38", "sequence"] \
                          + list(_pred_arrays.keys()) + list(_attr_slots.keys())
            _default_cols = [c for c in _default_cols if c in _all_opts]
            _sel_cols = st.multiselect(
                "columns",
                options=_all_opts,
                default=_default_cols,
                format_func=lambda c: _rename.get(c, c),
                key="dl_cols",
                help="`attr_<slot>` = full hypothetical attribution (4×L = 800 floats per row, space-separated, row-major A→T). Importance is sequence·attr — derive downstream if needed.",
            )
            _round = int(st.number_input("float decimals", value=4, min_value=0, max_value=8,
                                          step=1, key="dl_round"))
            # Pre-resolve mmap'd attribution for the slots whose attr_<slot> column
            # the user picked. mmap'd reads are O(1); per-row slicing happens inside
            # the box loop below.
            _attr_views: dict[str, tuple[dict, np.ndarray]] = {}
            for c in _sel_cols:
                if c in _attr_slots:
                    s = _attr_slots[c]
                    try:
                        _attr_views[c] = (s, np.asarray(_cached_load(s["path"], s["key"])))
                    except Exception as e:
                        st.warning(f"{c}: {e}")
            _frames = []
            _unique_rows: set[int] = set()
            for _b in _boxes:
                if _b["_uid"] not in _sel_box_uids:
                    continue
                _bm = ((_x >= _b["x0"]) & (_x <= _b["x1"]) &
                       (_y >= _b["y0"]) & (_y <= _b["y1"]))
                _rows = _custom[_bm]
                _rows_pos = np.nonzero(_bm)[0]  # positions in common_csv (for prediction arrays)
                if _rows.size == 0:
                    continue
                _df = pd.DataFrame(index=range(_rows.size))
                _fmt = f"%.{_round}f"
                for c in _sel_cols:
                    if c == "seq_idx":
                        _df["seq_idx"] = _rows
                    elif c in library.columns:
                        _df[_rename.get(c, c)] = library[c].iloc[_rows].to_numpy()
                    elif c in _pred_arrays:
                        _df[c] = _pred_arrays[c][_rows_pos]
                    elif c in _attr_views:
                        s, _attr = _attr_views[c]
                        _npz = s["attr_csv_to_npz"][_rows]
                        # Flatten (4, L) row-major into a single space-separated string.
                        _df[c] = [
                            " ".join(_fmt % v for v in np.asarray(_attr[int(i)]).ravel())
                            if 0 <= int(i) < _attr.shape[0] else ""
                            for i in _npz
                        ]
                _df["box_id"] = _uid_to_pos[_b["_uid"]]
                _frames.append(_df)
                _unique_rows.update(int(r) for r in _rows.tolist())
            st.caption(f"{len(_sel_box_uids)} box(es), {len(_unique_rows)} unique rows")
            if _frames:
                _combined = pd.concat(_frames, axis=0, ignore_index=True)
                _csv = _combined.to_csv(index=False, float_format=f"%.{_round}f").encode()
                st.download_button(
                    "download CSV",
                    data=_csv,
                    file_name="kcee_subsets.csv",
                    mime="text/csv",
                )
        if not _no_boxes:
            if st.button("clear boxes", key="clear_boxes"):
                st.session_state["isolate_boxes"] = []
                st.session_state["isolate_last_box_hash"] = None
                st.rerun()

    # Per-mode chart key: when k-condition or slot selection changes, the
    # scatter's customdata shape/contents change. Reusing a single "scatter"
    # key across modes leaves Streamlit holding stale selection state, so
    # clicks (and the red highlight lookup) silently miss in models mode.
    _abc_tag = "-".join(str(i) for i in ABC) if ABC else "none"
    _scatter_key = f"scatter__{_slot_key_tag}__{_abc_tag}"
    _sel_modes = ("points", "box") if _isolate_mode else ("points",)
    event = st.plotly_chart(
        fig,
        use_container_width=False,
        on_select="rerun",
        selection_mode=_sel_modes,
        key=_scatter_key,
    )


# --- isolate-mode: capture a new box selection ---
# Hash latest box geometry and only append on change, else every rerun
# re-adds the same selection.
if _isolate_mode and event is not None:
    _ev_sel = getattr(event, "selection", None)
    _ev_boxes = (_ev_sel.get("box") if _ev_sel else None) or []
    if _ev_boxes:
        _bx = _ev_boxes[-1]
        _xs = _bx.get("x") or []
        _ys = _bx.get("y") or []
        if len(_xs) == 2 and len(_ys) == 2:
            _x0, _x1 = sorted([float(_xs[0]), float(_xs[1])])
            _y0, _y1 = sorted([float(_ys[0]), float(_ys[1])])
            _h = f"{round(_x0,6)},{round(_x1,6)},{round(_y0,6)},{round(_y1,6)}"
            if _h != st.session_state.get("isolate_last_box_hash"):
                _nid = int(st.session_state["isolate_next_id"])
                _color = _BOX_PALETTE[(_nid - 1) % len(_BOX_PALETTE)]
                st.session_state["isolate_boxes"].append(
                    {"_uid": _nid, "x0": _x0, "x1": _x1, "y0": _y0, "y1": _y1, "color": _color}
                )
                st.session_state["isolate_next_id"] = _nid + 1
                st.session_state["isolate_last_box_hash"] = _h
                st.rerun()

# --- selected point: csv row ---
sel_csv: int | None = None
sel = getattr(event, "selection", None) if event is not None else None
if not _isolate_mode and sel and sel.get("points"):
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
_need_finemo = show_finemo_hits and any(s.get("finemo_tsv") for s in display_slots)
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
    insert_offset = _resolve_insert_offset(s, attr_L)
    var_lo, var_hi = _var_window(insert_offset, attr_L)
    wt_oh = seq_to_onehot(seq_full, length=attr_L, offset=insert_offset) if show_wt_logo else None
    # Lazy-load finemo only when the user has it enabled.
    if show_finemo_hits:
        fm_path = s.get("finemo_tsv", "")
        fm = _cached_finemo(fm_path) if fm_path else {}
        if fm_path:
            pid_map = _cached_finemo_csv_to_pid(fm_path, _name_keys, _name_vals)
            finemo_pid = int(pid_map[sel_csv])
        else:
            finemo_pid = npz_idx  # fallback (only correct when npz and finemo orderings agree)
        raw_hits = fm.get(finemo_pid, []) if fm and finemo_pid >= 0 else []
        hits = _hits_to_local(
            raw_hits,
            float(row.get("start_hg38", float("nan"))),
            attr_L,
            insert_offset=insert_offset,
        )
    else:
        hits = []
    plot_hits = hits
    title = f"{s['cell_type']} · {short_name(s['name'])}"
    if hits and show_finemo_hits:
        title += f"  ·  {len(hits)} finemo hits"
    try:
        with st.spinner(f"loading attribution map · {short_name(s['name'])}…"):
            attr_row = load_attr_row(s["path"], s["key"], npz_idx)
            if not np.isfinite(attr_row).all():
                st.warning(f"{short_name(s['name'])}: row {npz_idx} has NaN attribution "
                           f"(known bad rows for the standardtorch file: 18321, 18322). Skipping plot.")
                continue
            png = cached_attribution_png(
                path=s["path"], key=s["key"], idx=int(npz_idx),
                hits_signature=(_hits_signature(plot_hits), bool(show_finemo_hits)), show_wt_logo=show_wt_logo,
                attr=attr_row, wt_onehot=wt_oh, hits=plot_hits, title=title,
                proj_only_first=ENHANCER_LEN,
                crop=(var_lo, var_hi),
            )
    except Exception as e:
        st.error(f"{short_name(s['name'])}: {e}")
        continue
    st.image(png, use_container_width=True)
