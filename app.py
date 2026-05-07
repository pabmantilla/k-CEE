"""k-CEE attribution browser.

Streamlit UI to browse pre-computed attribution maps from up to 3 models,
with a 2D mech/func scatter, click-to-select, finemo-underlined logos,
and a WT-projected (onehot * attr) toggle.

Run:
    uv run streamlit run app.py
"""
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
ENHANCER_LEN = 230  # CSV sequences are 230bp; npz attr is 281bp. Project only the first 230.


# --- caches ---
@st.cache_data(show_spinner=False)
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
def _cached_predictions(path: str, key: str) -> np.ndarray:
    if key not in _cached_keys_any(path):
        return np.array([])
    return load_attr_file.__wrapped__(path, key) if False else _load_pred(path, key)


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


@st.cache_data(show_spinner=False)
def _cached_keys_any(path: str) -> list[str]:
    p = Path(path)
    keys: list[str] = []
    if p.suffix == ".npz":
        with np.load(p) as d:
            keys = list(d.files)
    elif p.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(p, "r") as f:
            f.visititems(lambda n, o: keys.append(n))
    return keys


@st.cache_data(show_spinner=False)
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


# --- sidebar: model slots ---
st.sidebar.header("Model slots")
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
            slots.append({**d, "path": "", "attr": None, "predictions": None, "finemo": {}})
            if path:
                st.warning("Path not found")
            continue
        try:
            keys = _cached_keys(path)
        except Exception as e:
            st.error(f"keys: {e}")
            slots.append({**d, "path": "", "attr": None, "predictions": None, "finemo": {}})
            continue
        if not keys:
            st.warning("No 3D arrays found")
            slots.append({**d, "path": "", "attr": None, "predictions": None, "finemo": {}})
            continue
        default_key_idx = keys.index(d["key"]) if d["key"] in keys else 0
        key = st.selectbox("attribution key", keys, index=default_key_idx, key=f"key_{i}")
        name = st.text_input("display name", value=d["model"] or key, key=f"name_{i}")
        try:
            attr = _cached_load(path, key)
            preds = _load_pred(path, d["pred_key"]) if d.get("pred_key") else np.array([])
            fm = _cached_finemo(finemo_tsv)
            st.caption(f"attr {attr.shape} · pred {preds.shape if preds.size else '—'} · finemo {len(fm)}")
            slots.append({**d, "path": path, "key": key, "name": name,
                          "attr": attr, "predictions": preds, "finemo": fm})
        except Exception as e:
            st.error(str(e))
            slots.append({**d, "path": "", "attr": None, "predictions": None, "finemo": {}})

loaded = [s for s in slots if s["attr"] is not None]


# --- sidebar: scoring ---
st.sidebar.header("Score (mech axis)")
mode_options = ["cossim", "eigenMaps", "dev_from_shared"]
mode = st.sidebar.selectbox("mode", mode_options, index=0)

scores: np.ndarray | None = None
score_label = mode

if mode == "cossim":
    if len(loaded) >= 2:
        names = [s["name"] for s in loaded]
        a_idx = st.sidebar.selectbox("A", range(len(loaded)), format_func=lambda i: names[i], index=0, key="cs_a")
        b_idx = st.sidebar.selectbox("B", range(len(loaded)), format_func=lambda i: names[i],
                                     index=min(1, len(loaded) - 1), key="cs_b")
        if a_idx != b_idx:
            a, b = loaded[a_idx]["attr"], loaded[b_idx]["attr"]
            n = min(a.shape[0], b.shape[0])
            scores = cossim_score(a[:n], b[:n])
            score_label = f"cossim({names[a_idx]}, {names[b_idx]})"
        else:
            st.sidebar.warning("Pick two different slots.")
    else:
        st.sidebar.info("cossim needs at least 2 loaded slots.")
elif mode == "eigenMaps":
    pkl = st.sidebar.text_input("eigen_analysis.pkl", value=DEFAULT_EIGEN_PKL, key="eig_pkl")
    eig_key = st.sidebar.text_input("score key", value="EI_1 var x r", key="eig_key")
    if pkl and Path(pkl).exists():
        try:
            scores = _cached_eigenmaps(pkl, eig_key)
            score_label = f"eigenMaps[{eig_key}]"
        except Exception as e:
            st.sidebar.error(str(e))
    elif pkl:
        st.sidebar.warning("Path not found")
elif mode == "dev_from_shared":
    if len(loaded) >= 2:
        names = [s["name"] for s in loaded]
        picked = st.sidebar.multiselect(
            "cell types to compare",
            list(range(len(loaded))),
            default=list(range(len(loaded))),
            format_func=lambda i: names[i],
            key="dev_pick",
        )
        if len(picked) >= 2:
            attrs_picked = [loaded[i]["attr"] for i in picked]
            scores = deviation_from_shared(attrs_picked)
            score_label = f"dev_from_shared({', '.join(names[i] for i in picked)})"
        else:
            st.sidebar.warning("Pick at least 2.")
    else:
        st.sidebar.info("dev_from_shared needs at least 2 loaded slots.")


# --- sidebar: display options ---
st.sidebar.header("Display")
show_wt_logo = st.sidebar.checkbox("WT-projected logo (attr × onehot)", value=False)
seq_pad = st.sidebar.number_input(
    "seq flank pad (bp, each side)",
    min_value=0, max_value=200, value=15, step=1,
    help="Library sequences extend start_hg38..stop_hg38 plus this much padding on each side. "
         "Used to convert genomic finemo hits to sequence-local coords.",
)


def _hits_to_local(hits, start_hg38, stop_hg38, strand, attr_L, pad):
    out = []
    if hits is None or not np.isfinite(start_hg38) or not np.isfinite(stop_hg38):
        return out
    seq_g_start = start_hg38 - pad
    seq_g_end = stop_hg38 + pad
    for h in hits:
        hs, he = int(h["start"]), int(h["end"])
        # If already small (looks local), assume local.
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


# --- main: gating message centered ---
if not loaded or library is None:
    msg = "Load a library CSV and at least one model in the sidebar to begin."
    if not loaded and library is not None:
        msg = "Load at least one model slot to begin."
    elif loaded and library is None:
        msg = "Set the library CSV path to begin."
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


# --- main: 2D scatter (func x mech) ---
n_seq = min(s["attr"].shape[0] for s in loaded)
n_seq = min(n_seq, len(library))
if scores is not None:
    n_seq = min(n_seq, len(scores))

x = (library["HepG2_log2FC"].values[:n_seq] - library["K562_log2FC"].values[:n_seq])
y = scores[:n_seq] if scores is not None else np.zeros(n_seq, dtype=np.float32)
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
        customdata=np.nonzero(valid)[0],
        hovertemplate="idx=%{customdata}<br>log2FC(H/K)=%{x:.3f}<br>mech=%{y:.3f}<extra></extra>",
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

# --- selected sequence ---
sel_idx: int | None = None
sel = getattr(event, "selection", None) if event is not None else None
if sel and sel.get("points"):
    pt = sel["points"][0]
    if pt.get("customdata") is not None:
        sel_idx = int(pt["customdata"])

st.markdown("---")
if sel_idx is None:
    st.info("Click a point to display attribution logos for that sequence.")
    st.stop()

# --- main: predicted vs measured per cell type ---
st.markdown(f"### Sequence index `{sel_idx}` — `{library.iloc[sel_idx].get('name', '')}`")
cols = st.columns(len(loaded))
for col, s in zip(cols, loaded):
    pred = float(s["predictions"][sel_idx]) if s["predictions"] is not None and sel_idx < len(s["predictions"]) else float("nan")
    meas_col = s.get("log2fc_col", "")
    meas = float(library.iloc[sel_idx][meas_col]) if meas_col and meas_col in library.columns else float("nan")
    with col:
        st.markdown(f"**{s['cell_type']}**  ·  _{s['name']}_")
        c1, c2 = st.columns(2)
        c1.metric("predicted", f"{pred:.3f}" if np.isfinite(pred) else "—")
        c2.metric("measured (log2FC)", f"{meas:.3f}" if np.isfinite(meas) else "—")

# --- main: attribution logos with finemo underlines ---
seq_full = str(library.iloc[sel_idx].get("sequence", "") or "")
attr_L = loaded[0]["attr"].shape[2]
wt_oh = seq_to_onehot(seq_full, length=attr_L) if show_wt_logo else None

st.markdown("#### Attribution logos")
for s in loaded:
    if sel_idx >= s["attr"].shape[0]:
        st.warning(f"{s['name']}: index {sel_idx} out of range ({s['attr'].shape[0]}).")
        continue
    raw_hits = s["finemo"].get(sel_idx, []) if s.get("finemo") else []
    row = library.iloc[sel_idx]
    hits = _hits_to_local(
        raw_hits,
        float(row.get("start_hg38", float("nan"))),
        float(row.get("stop_hg38", float("nan"))),
        str(row.get("str_hg38", "+") or "+"),
        s["attr"].shape[2],
        seq_pad,
    )
    title = f"{s['cell_type']} · {s['name']}"
    if hits:
        title += f"  ·  {len(hits)} finemo hits"
    fig = plot_attribution(
        s["attr"][sel_idx],
        wt_onehot=wt_oh,
        hits=hits,
        title=title,
        proj_only_first=ENHANCER_LEN,
    )
    st.pyplot(fig, clear_figure=True)
