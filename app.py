"""k-CEE attribution browser.

Streamlit UI to browse pre-computed AlphaGenome attribution maps.
2D or 3D comparison; click a point on the scatter to see logos with
finemo underlines.

Run:
    uv run streamlit run app.py
"""
from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kcee_ui.loader import load_attr_file, list_attr_keys
from kcee_ui.scoring import cossim_score, eigenmaps_score, deviation_from_shared
from kcee_ui.plotting import plot_attribution
from kcee_ui.data import load_library, seq_to_onehot
from kcee_ui.finemo import load_finemo_hits
from kcee_ui.defaults import DEFAULT_SLOTS, DEFAULT_EIGEN_PKL, DEFAULT_LIBRARY_CSV


st.set_page_config(page_title="k-CEE attribution browser", layout="wide")
st.title("k-CEE attribution browser")

N_SLOTS = 3
ENHANCER_LEN = 230
SEQ_FLANK_PAD = 15  # library seqs span start_hg38..stop_hg38 plus 15bp each side.
NAME_MAX = 24


def short_name(name: str) -> str:
    name = str(name or "")
    return name if len(name) <= NAME_MAX else name[:NAME_MAX - 1] + "…"


# --- caches ---
@st.cache_data(show_spinner="loading attributions…")
def _cached_load(path: str, key: str) -> np.ndarray:
    return load_attr_file(path, key)


@st.cache_data(show_spinner=False)
def _cached_keys(path: str) -> list[str]:
    return list_attr_keys(path)


@st.cache_data(show_spinner=False)
def _cached_eigenmaps(path: str, key: str) -> np.ndarray:
    return eigenmaps_score(path, key)


@st.cache_data(show_spinner=False)
def _cached_library(path: str) -> pd.DataFrame:
    return load_library(path)


@st.cache_data(show_spinner=False)
def _load_pred(path: str, key: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npz":
        with np.load(p) as d:
            if key in d.files:
                return np.asarray(d[key], dtype=np.float32)
    elif p.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            if key in f:
                return np.asarray(f[key][:], dtype=np.float32)
    return np.array([])


@st.cache_data(show_spinner="loading finemo hits…")
def _cached_finemo(tsv_path: str) -> dict[int, list[dict]]:
    if not tsv_path or not Path(tsv_path).exists():
        return {}
    return load_finemo_hits(tsv_path)


# --- sidebar: library ---
st.sidebar.header("Library")
csv_path = st.sidebar.text_input("library CSV", value=DEFAULT_LIBRARY_CSV)
library: pd.DataFrame | None = None
if csv_path and Path(csv_path).exists():
    try:
        library = _cached_library(csv_path)
        st.sidebar.caption(f"{len(library)} rows")
    except Exception as e:
        st.sidebar.error(f"CSV load: {e}")


# --- sidebar: model slots (config + lightweight metadata only) ---
st.sidebar.header("Available models")
slots: list[dict] = []
for i in range(N_SLOTS):
    d = DEFAULT_SLOTS[i] if i < len(DEFAULT_SLOTS) else {
        "cell_type": f"slot{i+1}", "model": "", "key": "", "pred_key": "",
        "log2fc_col": "", "path": "", "finemo_tsv": "",
    }
    with st.sidebar.expander(f"Slot {i + 1} — {d['cell_type']}", expanded=(i < 2)):
        path = st.text_input("attr file (.npz / .h5)", value=d["path"], key=f"path_{i}")
        finemo_tsv = st.text_input("finemo hits.tsv", value=d.get("finemo_tsv", ""), key=f"fm_{i}")
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
        key = st.selectbox("attribution key", keys, index=default_key_idx, key=f"key_{i}")
        name = st.text_input("display name", value=d["model"] or key, key=f"name_{i}")
        preds = _load_pred(path, d["pred_key"]) if d.get("pred_key") else np.array([])
        fm = _cached_finemo(finemo_tsv) if finemo_tsv else {}
        n_attr = int(preds.shape[0]) if preds.size else 0
        st.caption(f"pred {preds.shape if preds.size else '—'} · finemo {len(fm)}")
        slots.append({**d, "path": path, "key": key, "name": name,
                      "predictions": preds, "finemo": fm, "n_attr": n_attr})

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


if loaded and library is not None:
    name_keys = tuple(library["name"].astype(str).tolist())
    name_vals = tuple(range(len(library)))
    for s in loaded:
        s["attr_csv_to_npz"] = _build_attr_csv_to_npz(s, library)
        s["covered_csv"] = np.nonzero(s["attr_csv_to_npz"] >= 0)[0]
        s["finemo_csv_to_pid"] = _cached_finemo_csv_to_pid(
            s.get("finemo_tsv", ""), name_keys, name_vals
        )


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


@st.cache_data(show_spinner="computing cossim…")
def _cossim_aligned(path_a: str, key_a: str, path_b: str, key_b: str,
                    npz_a: tuple, npz_b: tuple) -> np.ndarray:
    a = _cached_load(path_a, key_a)[list(npz_a)]
    b = _cached_load(path_b, key_b)[list(npz_b)]
    return cossim_score(a, b)


@st.cache_data(show_spinner="computing dev_from_shared…")
def _dev_aligned(paths: tuple[tuple[str, str], ...], idxs: tuple[tuple, ...]) -> np.ndarray:
    arrs = [_cached_load(p, k)[list(ix)] for (p, k), ix in zip(paths, idxs)]
    return deviation_from_shared(arrs)


scores: np.ndarray | None = None  # aligned to `common_csv` (below)
score_label = "—"
common_csv: np.ndarray = np.array([], dtype=np.int64)

if loaded and ABC and library is not None:
    sl_subset = [loaded[i] for i in ABC]
    common_csv = _common_csv(sl_subset)
    if dim == "2D":
        score_mode = st.sidebar.selectbox("score (mech axis)", ["cossim", "eigenMaps"], index=0)
        a, b = sl_subset[0], sl_subset[1]
        if ABC[0] == ABC[1]:
            st.sidebar.warning("A and B are the same slot.")
        elif score_mode == "cossim":
            try:
                scores = _cossim_aligned(
                    a["path"], a["key"], b["path"], b["key"],
                    tuple(a["attr_csv_to_npz"][common_csv].tolist()),
                    tuple(b["attr_csv_to_npz"][common_csv].tolist()),
                )
                score_label = f"cossim({short_name(a['name'])}, {short_name(b['name'])})"
            except Exception as e:
                st.sidebar.error(str(e))
        else:  # eigenMaps
            pkl = st.sidebar.text_input("eigen_analysis.pkl", value=DEFAULT_EIGEN_PKL, key="eig_pkl")
            eig_key = st.sidebar.text_input("score key", value="EI_1 var x r", key="eig_key")
            if pkl and Path(pkl).exists():
                try:
                    eig_full = _cached_eigenmaps(pkl, eig_key)
                    # eigen pkl has its own length; align by truncating to min over common_csv positions in a's npz space
                    a_npz = a["attr_csv_to_npz"][common_csv]
                    valid = a_npz < len(eig_full)
                    common_csv = common_csv[valid]
                    scores = eig_full[a_npz[valid]]
                    score_label = f"eigenMaps[{eig_key}]"
                except Exception as e:
                    st.sidebar.error(str(e))
            elif pkl:
                st.sidebar.warning("Path not found")
    else:  # 3D
        if len(set(ABC)) >= 2:
            try:
                paths = tuple((s["path"], s["key"]) for s in sl_subset)
                idxs = tuple(tuple(s["attr_csv_to_npz"][common_csv].tolist()) for s in sl_subset)
                scores = _dev_aligned(paths, idxs)
                score_label = f"dev_from_shared({', '.join(short_name(s['name']) for s in sl_subset)})"
            except Exception as e:
                st.sidebar.error(str(e))
        else:
            st.sidebar.warning("Pick at least 2 distinct slots.")


# --- sidebar: display options ---
st.sidebar.header("Display")
show_wt_logo = st.sidebar.checkbox("WT-projected logo (attr × onehot)", value=False)


def _hits_to_local(hits, start_hg38, stop_hg38, strand, attr_L):
    out = []
    if hits is None or not np.isfinite(start_hg38) or not np.isfinite(stop_hg38):
        return out
    seq_g_start = start_hg38 - SEQ_FLANK_PAD
    seq_g_end = stop_hg38 + SEQ_FLANK_PAD
    for h in hits:
        hs, he = int(h["start"]), int(h["end"])
        if max(hs, he) < attr_L:
            s, e = hs, he
        elif strand == "-":
            s = int(seq_g_end - he)
            e = int(seq_g_end - hs)
        else:
            s = int(hs - seq_g_start)
            e = int(he - seq_g_start)
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
hk_cols = library[["HepG2_log2FC", "K562_log2FC"]].iloc[common_csv].values
x = hk_cols[:, 0] - hk_cols[:, 1]
y = scores
valid = np.isfinite(x) & np.isfinite(y)

fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=x[valid],
        y=y[valid],
        mode="markers",
        marker=dict(
            size=4,
            color=y[valid],
            colorscale="Inferno",
            opacity=0.6,
            colorbar=dict(title=score_label),
        ),
        customdata=common_csv[valid],
        hovertemplate="csv_row=%{customdata}<br>log2FC(H/K)=%{x:.3f}<br>mech=%{y:.3f}<extra></extra>",
        name="",
    )
)
fig.update_layout(
    xaxis_title="log2FC (HepG2 / K562)   [func]",
    yaxis_title=f"{score_label}   [mech]",
    height=520,
    margin=dict(l=40, r=20, t=30, b=40),
    template="plotly_white",
    dragmode="zoom",
)

event = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode=("points",),
    key="scatter",
)


# --- selected point: csv row ---
sel_csv: int | None = None
sel = getattr(event, "selection", None) if event is not None else None
if sel and sel.get("points"):
    pt = sel["points"][0]
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
    pred = float(s["predictions"][npz_idx]) if s["predictions"] is not None and 0 <= npz_idx < len(s["predictions"]) else float("nan")
    meas_col = s.get("log2fc_col", "")
    meas = float(row[meas_col]) if meas_col and meas_col in library.columns else float("nan")
    with col:
        st.markdown(f"**{s['cell_type']}**  ·  _{short_name(s['name'])}_")
        c1, c2 = st.columns(2)
        c1.metric("predicted", f"{pred:.3f}" if np.isfinite(pred) else "—")
        c2.metric("measured (log2FC)", f"{meas:.3f}" if np.isfinite(meas) else "—")


# --- attribution logos ---
seq_full = str(row.get("sequence", "") or "")

st.markdown("#### Attribution logos")
for s in display_slots:
    npz_idx = int(s["attr_csv_to_npz"][sel_csv])
    if npz_idx < 0:
        st.warning(f"{short_name(s['name'])}: no attribution for this CSV row.")
        continue
    try:
        attr = _cached_load(s["path"], s["key"])
    except Exception as e:
        st.error(f"{short_name(s['name'])}: {e}")
        continue
    if npz_idx >= attr.shape[0]:
        st.warning(f"{short_name(s['name'])}: npz idx {npz_idx} out of range.")
        continue
    attr_L = attr.shape[2]
    wt_oh = seq_to_onehot(seq_full, length=attr_L) if show_wt_logo else None
    pid_map = s.get("finemo_csv_to_pid")
    if pid_map is not None:
        finemo_pid = int(pid_map[sel_csv])
    else:
        finemo_pid = npz_idx  # fallback (only correct when npz and finemo orderings agree)
    raw_hits = s["finemo"].get(finemo_pid, []) if s.get("finemo") and finemo_pid >= 0 else []
    hits = _hits_to_local(
        raw_hits,
        float(row.get("start_hg38", float("nan"))),
        float(row.get("stop_hg38", float("nan"))),
        str(row.get("str_hg38", "+") or "+"),
        attr_L,
    )
    title = f"{s['cell_type']} · {short_name(s['name'])}"
    if hits:
        title += f"  ·  {len(hits)} finemo hits"
    fig = plot_attribution(
        attr[npz_idx],
        wt_onehot=wt_oh,
        hits=hits,
        title=title,
        proj_only_first=ENHANCER_LEN,
    )
    st.pyplot(fig, clear_figure=True)
