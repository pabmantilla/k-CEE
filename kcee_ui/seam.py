"""SEAM mode: foreground / background viewer mirroring the kcee mech/func layout.

Data lives under
`/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/SEAM_target_spaces/`.
1059 sequences total (353 per condition: diff-diff / same-diff / same-same).
HepG2 + K562 only; only Pablo's AG models. WTC11 + other model families are
not represented in SEAM space.

Each sequence contributes up to 3 points on the mech (cossim HepG2 vs K562) ×
func (log2FC HepG2 − K562) scatter:
  - WT attribution         (gray)
  - SEAM foreground scaled (red)
  - SEAM background scaled (blue)
A view toggle lets you show all three, or restrict to one type.
Click a point to display the corresponding (ct × type) attribution logos.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from kcee_ui.data import seq_to_onehot
from kcee_ui.plotting import fast_logo
from kcee_ui.scoring import attr_to_importance, cossim_score

SEAM_ROOT = Path(
    "/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/SEAM_target_spaces"
)
FG_DIR = SEAM_ROOT / "results" / "foregrounds"
LIB_PKL = SEAM_ROOT / "libraries" / "hippo_target_library.pkl"

CELL_TYPES = ("HepG2", "K562")
TYPES = ("wt", "foreground", "background")
TYPE_LABEL = {"wt": "WT", "foreground": "foreground", "background": "background"}
TYPE_COLOR = {"wt": "#888888", "foreground": "#d62728", "background": "#1f77b4"}
TYPE_FILE = {
    "wt": "wt_attribution.npy",
    "foreground": "foreground_scaled.npy",
    "background": "average_background_scaled.npy",
}
VIEW_OPTIONS = ("all three", "foregrounds only", "backgrounds only", "WT only")
VIEW_TO_TYPES = {
    "all three":         ("wt", "foreground", "background"),
    "foregrounds only":  ("foreground",),
    "backgrounds only":  ("background",),
    "WT only":           ("wt",),
}


@st.cache_data(show_spinner=False)
def _load_library() -> pd.DataFrame:
    with open(LIB_PKL, "rb") as f:
        return pickle.load(f)["df"].reset_index(drop=True)


@st.cache_data(show_spinner="loading SEAM maps (1059 × 3 × 2)…")
def _stack_maps() -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray, list[int]]:
    """Return ({(type, ct): (N, 4, 230)}, wt_onehot (N, 4, 230), seq_idx list)."""
    df = _load_library()
    seq_idxs = [int(x) for x in df["seq_idx"].values]
    N = len(seq_idxs)
    maps: dict[tuple[str, str], np.ndarray] = {}
    for ct in CELL_TYPES:
        for tp in TYPES:
            arr = np.zeros((N, 4, 230), dtype=np.float32)
            fname = TYPE_FILE[tp]
            for i, sid in enumerate(seq_idxs):
                p = FG_DIR / ct / str(sid) / fname
                arr[i] = np.load(p).T
            maps[(tp, ct)] = arr
    onehot = np.zeros((N, 4, 230), dtype=np.float32)
    for i, seq in enumerate(df["sequence"].astype(str).tolist()):
        onehot[i] = seq_to_onehot(seq, length=230, offset=0)
    return maps, onehot, seq_idxs


@st.cache_data(show_spinner=False)
def _compute_scores() -> dict[str, np.ndarray]:
    """Per-(type) cossim(HepG2, K562) over the variable insert region."""
    maps, onehot, _ = _stack_maps()
    out: dict[str, np.ndarray] = {}
    for tp in TYPES:
        imp_h = attr_to_importance(maps[(tp, "HepG2")], onehot)
        imp_k = attr_to_importance(maps[(tp, "K562")], onehot)
        out[tp] = cossim_score(imp_h, imp_k)
    return out


def _load_fg_pack(ct: str, seq_idx: int) -> dict:
    d = FG_DIR / ct / str(seq_idx)
    return {
        "wt":         np.load(d / TYPE_FILE["wt"]),
        "foreground": np.load(d / TYPE_FILE["foreground"]),
        "background": np.load(d / TYPE_FILE["background"]),
        "ref_cluster_idx": int(np.load(d / "ref_cluster_idx.npy")),
    }


def _make_scatter(df: pd.DataFrame, mask: np.ndarray, scores: dict[str, np.ndarray],
                  visible_types: tuple[str, ...]) -> go.Figure:
    log2fc_diff = (df["HepG2_log2FC"].values - df["K562_log2FC"].values).astype(np.float32)
    fig = go.Figure()
    for tp in visible_types:
        y = scores[tp][mask]
        x = log2fc_diff[mask]
        cd = np.stack([
            df.loc[mask, "seq_idx"].values.astype(np.int64),
            np.array([{"wt": 0, "foreground": 1, "background": 2}[tp]] * int(mask.sum()), dtype=np.int64),
        ], axis=-1)
        hover = (
            "seq_idx=%{customdata[0]}<br>"
            f"type={TYPE_LABEL[tp]}<br>"
            "cossim(HepG2,K562)=%{y:.3f}<br>"
            "log2FC HepG2 − K562=%{x:.3f}<extra></extra>"
        )
        fig.add_trace(go.Scattergl(
            x=x, y=y,
            mode="markers",
            name=TYPE_LABEL[tp],
            marker=dict(color=TYPE_COLOR[tp], size=6, opacity=0.65, line=dict(width=0)),
            customdata=cd,
            hovertemplate=hover,
        ))
    fig.update_layout(
        xaxis_title="log2FC HepG2 − K562   [func]",
        yaxis_title="cossim(HepG2 attr, K562 attr)   [mech]",
        height=520,
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", y=1.08, x=0),
        dragmode="zoom",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#bbb", line_width=1)
    fig.add_vline(x=0, line_dash="dot", line_color="#bbb", line_width=1)
    return fig


def _render_maps(seq_idx: int, row: pd.Series, view: str, picked_type: str | None) -> None:
    """Bottom-of-page logos. Layout:
        - 'all three' view: 3 rows (WT/fg/bg) × 2 cols (HepG2/K562)
        - one-type view:    1 row  (picked type) × 2 cols (HepG2/K562)
    """
    if view == "all three":
        rows = ("wt", "foreground", "background")
    else:
        rows = (VIEW_TO_TYPES[view][0],)

    fig_h = 2.1 * len(rows) + 0.4
    fig, axes = plt.subplots(len(rows), 2, figsize=(18, fig_h), sharex=True, squeeze=False)
    for col, ct in enumerate(CELL_TYPES):
        try:
            pack = _load_fg_pack(ct, seq_idx)
        except FileNotFoundError:
            for r in range(len(rows)):
                axes[r, col].text(0.5, 0.5, f"missing for {ct}",
                                  ha="center", va="center", transform=axes[r, col].transAxes)
                axes[r, col].set_axis_off()
            continue
        all_vals = np.concatenate([pack[k].ravel() for k in rows])
        yabs = float(np.max(np.abs(all_vals))) * 1.05 or 1.0
        ylim = (-yabs, yabs)
        for r, key in enumerate(rows):
            label = f"{ct} · {TYPE_LABEL[key]}"
            if key == "background":
                label += f" (ref cluster {pack['ref_cluster_idx']})"
            highlight = (picked_type == key)
            fast_logo(pack[key], ax=axes[r, col], ylim=ylim)
            axes[r, col].set_title(label, fontsize=10,
                                   color=TYPE_COLOR[key] if highlight else "black",
                                   fontweight="bold" if highlight else "normal")
            axes[r, col].set_ylabel("attr" if col == 0 else "")
    axes[-1, 0].set_xlabel("position (230 bp)")
    axes[-1, 1].set_xlabel("position (230 bp)")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render() -> None:
    df = _load_library()
    scores = _compute_scores()

    # --- sidebar pickers ---
    st.sidebar.header("SEAM view")
    view = st.sidebar.radio(
        "show",
        VIEW_OPTIONS,
        index=0,
        key="seam_view",
        help="all three: gray WT + red foreground + blue background per seq. "
             "Otherwise just one type per seq.",
    )
    cond_options = ("all", "diff-diff", "same-diff", "same-same")
    cond = st.sidebar.selectbox("condition filter", cond_options, index=0, key="seam_cond")

    # --- mask ---
    mask = np.ones(len(df), dtype=bool)
    if cond != "all":
        mask &= (df["condition"].values == cond)
    finite = np.isfinite(df["HepG2_log2FC"].values) & np.isfinite(df["K562_log2FC"].values)
    mask &= finite

    visible_types = VIEW_TO_TYPES[view]

    st.markdown(f"**{int(mask.sum())} sequences** · view: `{view}` · condition: `{cond}`")
    st.caption(
        "mech (y): cossim(HepG2 attr, K562 attr) on z-normalised importance over the 200-bp var region.  "
        "func (x): measured log2FC HepG2 − K562.  "
        f"colors: { ' / '.join(f'<span style=\"color:{TYPE_COLOR[t]}\">{TYPE_LABEL[t]}</span>' for t in TYPES) }.",
        unsafe_allow_html=True,
    )

    fig = _make_scatter(df, mask, scores, visible_types)
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode=("points",),
        key=f"seam_scatter__{view}__{cond}",
    )

    # --- click handler ---
    sel_seq_idx: int | None = None
    sel_type: str | None = None
    sel = getattr(event, "selection", None) if event is not None else None
    if sel and sel.get("points"):
        pts = [p for p in sel["points"] if p.get("customdata") is not None]
        if pts:
            cd = pts[0]["customdata"]
            sel_seq_idx = int(cd[0])
            sel_type = {0: "wt", 1: "foreground", 2: "background"}.get(int(cd[1]))

    st.markdown("---")
    if sel_seq_idx is None:
        st.info("Click a point above to display attribution logos for that sequence.")
        return

    row = df[df["seq_idx"] == sel_seq_idx].iloc[0]
    name = str(row.get("name", ""))
    name_short = (name[:80] + "…") if len(name) > 80 else name
    st.markdown(f"### `{name_short}`  · seq_idx `{sel_seq_idx}`  · condition `{row['condition']}`")
    if len(name) > 80:
        st.caption(name)
    st.caption(
        f"EI_1 var x r **{row['EI_1 var x r']:+.3f}**  ·  "
        f"pred_HepG2 **{row['pred_HepG2']:+.3f}**  ·  pred_K562 **{row['pred_K562']:+.3f}**  ·  "
        f"log2FC HepG2 **{row['HepG2_log2FC']:+.3f}**  ·  log2FC K562 **{row['K562_log2FC']:+.3f}**"
    )

    _render_maps(sel_seq_idx, row, view, sel_type)

    st.caption(
        "SEAM space covers 1059 sequences (Pablo's AG models, HepG2 + K562 only). "
        "Other model families / WTC11 are not represented."
    )
