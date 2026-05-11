"""SEAM mode: foreground / background viewer mirroring the kcee mech/func layout.

Data lives under
`/grid/koo/home/pmantill/projects/Virtual_Experiments/Hippo_axis/Hippo_dependency_mpra/SEAM_target_spaces/`.
1059 sequences (353 per condition: diff-diff / same-diff / same-same).
HepG2 + K562 only; only Pablo's AG models. WTC11 + other model families are
not represented in SEAM space.

Each sequence contributes up to 3 points on the mech (cossim HepG2 vs K562) ×
func (log2FC HepG2 − K562) scatter:
  - WT attribution         (gray, circle)
  - SEAM foreground scaled (red, triangle-up)
  - SEAM background scaled (blue, triangle-down)
A view toggle restricts to one type or shows all three. Plot controls mirror
the kcee browser: colorscale, color clipping, color-by, marginals, manual axis
limits, figure dims, highlight-by-seq_idx.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
TYPE_SYMBOL = {"wt": "circle", "foreground": "triangle-up", "background": "triangle-down"}
# wt/foreground come from single npy files; background is the intra-cluster bg
# at the ref cluster (cluster_backgrounds[ref_cluster_idx]) per the SEAM
# foreground viewer notebook — NOT the averaged background.
TYPE_FILE = {
    "wt": "wt_attribution.npy",
    "foreground": "foreground_scaled.npy",
}


def _load_intra_bg(seq_dir: Path) -> np.ndarray:
    """SEAM intra-cluster background = cluster_backgrounds[ref_cluster_idx]."""
    ref_idx = int(np.load(seq_dir / "ref_cluster_idx.npy"))
    cluster_bgs = np.load(seq_dir / "cluster_backgrounds.npy")  # (n_clusters, 230, 4)
    return cluster_bgs[ref_idx].astype(np.float32)
VIEW_OPTIONS = ("all three", "foregrounds only", "backgrounds only", "WT only")
VIEW_TO_TYPES = {
    "all three":         ("wt", "foreground", "background"),
    "foregrounds only":  ("foreground",),
    "backgrounds only":  ("background",),
    "WT only":           ("wt",),
}
TYPE_CODE = {"wt": 0, "foreground": 1, "background": 2}
CODE_TO_TYPE = {v: k for k, v in TYPE_CODE.items()}

_COLOR_MODES = (
    "by type (WT/fg/bg)",
    "EI_1 var x r",
    "log2FC HepG2 − K562",
    "log2FC HepG2",
    "log2FC K562",
    "predicted HepG2",
    "predicted K562",
    "predicted HepG2 − K562",
    "condition",
)


@st.cache_data(show_spinner=False)
def _load_library() -> pd.DataFrame:
    with open(LIB_PKL, "rb") as f:
        return pickle.load(f)["df"].reset_index(drop=True)


@st.cache_data(show_spinner="loading SEAM maps (1059 × 3 × 2)…")
def _stack_maps() -> tuple[dict[tuple[str, str], np.ndarray], np.ndarray, list[int]]:
    df = _load_library()
    seq_idxs = [int(x) for x in df["seq_idx"].values]
    N = len(seq_idxs)
    maps: dict[tuple[str, str], np.ndarray] = {}
    for ct in CELL_TYPES:
        for tp in ("wt", "foreground"):
            arr = np.zeros((N, 4, 230), dtype=np.float32)
            fname = TYPE_FILE[tp]
            for i, sid in enumerate(seq_idxs):
                arr[i] = np.load(FG_DIR / ct / str(sid) / fname).T
            maps[(tp, ct)] = arr
        # background = intra-cluster bg at ref cluster
        arr = np.zeros((N, 4, 230), dtype=np.float32)
        for i, sid in enumerate(seq_idxs):
            arr[i] = _load_intra_bg(FG_DIR / ct / str(sid)).T
        maps[("background", ct)] = arr
    onehot = np.zeros((N, 4, 230), dtype=np.float32)
    for i, seq in enumerate(df["sequence"].astype(str).tolist()):
        onehot[i] = seq_to_onehot(seq, length=230, offset=0)
    return maps, onehot, seq_idxs


@st.cache_data(show_spinner=False)
def _compute_scores() -> dict[str, np.ndarray]:
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
        "background": _load_intra_bg(d),
        "ref_cluster_idx": int(np.load(d / "ref_cluster_idx.npy")),
    }


def _binned_mean(values: np.ndarray, color: np.ndarray, bins: int = 30):
    finite = np.isfinite(values) & np.isfinite(color)
    v, c = values[finite], color[finite]
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


def _per_seq_color(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, str]:
    """Per-seq color array + colorbar label. Returns NaNs where unavailable."""
    n = len(df)
    if mode == "EI_1 var x r":
        return df["EI_1 var x r"].to_numpy(dtype=np.float32), "EI_1 var x r"
    if mode == "log2FC HepG2 − K562":
        return (df["HepG2_log2FC"] - df["K562_log2FC"]).to_numpy(dtype=np.float32), "log2FC HepG2 − K562"
    if mode == "log2FC HepG2":
        return df["HepG2_log2FC"].to_numpy(dtype=np.float32), "log2FC HepG2"
    if mode == "log2FC K562":
        return df["K562_log2FC"].to_numpy(dtype=np.float32), "log2FC K562"
    if mode == "predicted HepG2":
        return df["pred_HepG2"].to_numpy(dtype=np.float32), "predicted HepG2"
    if mode == "predicted K562":
        return df["pred_K562"].to_numpy(dtype=np.float32), "predicted K562"
    if mode == "predicted HepG2 − K562":
        return (df["pred_HepG2"] - df["pred_K562"]).to_numpy(dtype=np.float32), "predicted HepG2 − K562"
    if mode == "condition":
        cond_to_code = {"diff-diff": 0, "same-diff": 1, "same-same": 2}
        return df["condition"].map(cond_to_code).to_numpy(dtype=np.float32), "condition (0=dd,1=sd,2=ss)"
    return np.full(n, np.nan, dtype=np.float32), ""


def _resolve_clip(values: np.ndarray, mode: str, manual: tuple[float, float] | None
                  ) -> tuple[float | None, float | None]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None, None
    if mode == "auto (2–98%)":
        return float(np.percentile(v, 2)), float(np.percentile(v, 98))
    if mode == "manual" and manual is not None:
        return manual
    return None, None


def _scatter_fig(df: pd.DataFrame, mask: np.ndarray, scores: dict[str, np.ndarray],
                 visible_types: tuple[str, ...], color_mode: str, colorscale: str,
                 vmin: float | None, vmax: float | None,
                 fig_w: int, fig_h: int, marg_x: str, marg_y: str,
                 xmin: float | None, xmax: float | None,
                 ymin: float | None, ymax: float | None,
                 highlight_seq: int) -> tuple[go.Figure, list[str]]:
    """Returns (figure, trace_types) where trace_types[i] is the SEAM type
    ('wt'/'foreground'/'background') of trace i in the scatter cell, so the
    Streamlit click handler can map curve_number → type without relying on
    multi-element customdata (which Streamlit collapses to a scalar)."""
    log2fc_diff = (df["HepG2_log2FC"].values - df["K562_log2FC"].values).astype(np.float32)
    seq_idxs = df["seq_idx"].values.astype(np.int64)

    has_marg = (marg_x != "none") or (marg_y != "none")
    if has_marg:
        fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True,
                            row_heights=[0.2, 0.8], column_widths=[0.8, 0.2],
                            horizontal_spacing=0.02, vertical_spacing=0.02)
        scatter_row, scatter_col = 2, 1
    else:
        fig = go.Figure()
        scatter_row = scatter_col = None

    use_value_color = (color_mode != "by type (WT/fg/bg)")
    value_color, color_label = _per_seq_color(df, color_mode) if use_value_color else (None, "")

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_color: list[np.ndarray] = []
    trace_types: list[str] = []

    for ti, tp in enumerate(visible_types):
        m = mask & np.isfinite(log2fc_diff) & np.isfinite(scores[tp])
        if value_color is not None:
            m = m & np.isfinite(value_color)
        if not m.any():
            continue
        xs = log2fc_diff[m]
        ys = scores[tp][m]
        sids = seq_idxs[m]
        cd = sids.astype(np.int64)  # 1D: just seq_idx; type comes from trace
        if use_value_color:
            cval = value_color[m]
            marker = dict(
                size=6,
                color=cval,
                colorscale=colorscale,
                opacity=0.7,
                symbol=TYPE_SYMBOL[tp],
                line=dict(width=0),
                showscale=(ti == 0),
                colorbar=dict(title=color_label) if ti == 0 else None,
            )
            if vmin is not None:
                marker["cmin"] = float(vmin)
            if vmax is not None:
                marker["cmax"] = float(vmax)
            all_color.append(cval.astype(np.float64))
        else:
            marker = dict(
                size=6,
                color=TYPE_COLOR[tp],
                opacity=0.65,
                symbol=TYPE_SYMBOL[tp],
                line=dict(width=0),
            )
        hover = (
            "seq_idx=%{customdata}<br>"
            f"type={TYPE_LABEL[tp]}<br>"
            "x=%{x:.3f}<br>y=%{y:.3f}"
            + ("<br>color=%{marker.color:.3f}" if use_value_color else "")
            + "<extra></extra>"
        )
        trace = go.Scattergl(
            x=xs, y=ys, mode="markers", name=TYPE_LABEL[tp],
            marker=marker, customdata=cd, hovertemplate=hover,
        )
        if has_marg:
            fig.add_trace(trace, row=scatter_row, col=scatter_col)
        else:
            fig.add_trace(trace)
        trace_types.append(tp)
        all_x.append(xs.astype(np.float64))
        all_y.append(ys.astype(np.float64))

    # Marginals (computed on the union of all visible points)
    if has_marg and all_x:
        x_all = np.concatenate(all_x)
        y_all = np.concatenate(all_y)
        marg_color = np.concatenate(all_color) if (use_value_color and all_color) else None

        def _bars(values, colorvals, hn, orient):
            out = _binned_mean(values, colorvals, bins=30) if colorvals is not None else None
            if out is not None:
                centers, widths, counts, means = out
                marker = dict(color=means, colorscale=colorscale,
                              cmin=vmin, cmax=vmax, showscale=False)
            else:
                out2 = _binned_counts(values, bins=30)
                if out2 is None:
                    return None
                centers, widths, counts = out2
                marker = dict(color="#888")
            total = counts.sum() or 1.0
            if hn == "density":
                vals = counts / (total * widths)
            elif hn == "probability":
                vals = counts / total
            else:
                vals = counts
            if orient == "v":
                return go.Bar(x=centers, y=vals, width=widths, marker=marker, showlegend=False, name="")
            return go.Bar(x=vals, y=centers, width=widths, orientation="h", marker=marker, showlegend=False, name="")

        if marg_x != "none":
            b = _bars(x_all, marg_color, marg_x, "v")
            if b is not None:
                fig.add_trace(b, row=1, col=1)
        if marg_y != "none":
            b = _bars(y_all, marg_color, marg_y, "h")
            if b is not None:
                fig.add_trace(b, row=2, col=2)
        fig.update_xaxes(title_text="log2FC HepG2 − K562   [func]", row=2, col=1)
        fig.update_yaxes(title_text="cossim(HepG2, K562)   [mech]", row=2, col=1)
    else:
        fig.update_layout(
            xaxis_title="log2FC HepG2 − K562   [func]",
            yaxis_title="cossim(HepG2, K562)   [mech]",
        )

    fig.update_layout(
        width=int(fig_w), height=int(fig_h),
        margin=dict(l=40, r=20, t=30, b=40),
        template="plotly_white",
        dragmode="zoom",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    if has_marg:
        if xmin is not None and xmax is not None:
            fig.update_xaxes(range=[xmin, xmax], row=2, col=1)
        if ymin is not None and ymax is not None:
            fig.update_yaxes(range=[ymin, ymax], row=2, col=1)
    else:
        if xmin is not None and xmax is not None:
            fig.update_xaxes(range=[xmin, xmax])
        if ymin is not None and ymax is not None:
            fig.update_yaxes(range=[ymin, ymax])

    # reference lines at 0
    if has_marg:
        fig.add_hline(y=0, line_dash="dot", line_color="#bbb", line_width=1, row=2, col=1)
        fig.add_vline(x=0, line_dash="dot", line_color="#bbb", line_width=1, row=2, col=1)
    else:
        fig.add_hline(y=0, line_dash="dot", line_color="#bbb", line_width=1)
        fig.add_vline(x=0, line_dash="dot", line_color="#bbb", line_width=1)

    if highlight_seq is not None and highlight_seq >= 0:
        for tp in visible_types:
            m = mask & (seq_idxs == int(highlight_seq))
            m &= np.isfinite(log2fc_diff) & np.isfinite(scores[tp])
            if not m.any():
                continue
            hl = go.Scatter(
                x=[float(log2fc_diff[m][0])], y=[float(scores[tp][m][0])],
                mode="markers",
                marker=dict(size=18, color="rgba(0,0,0,0)",
                            line=dict(color="#e63946", width=3),
                            symbol=TYPE_SYMBOL[tp]),
                hovertemplate=f"seq_idx={int(highlight_seq)}<br>type={TYPE_LABEL[tp]}<extra></extra>",
                showlegend=False, name="highlight",
            )
            if has_marg:
                fig.add_trace(hl, row=scatter_row, col=scatter_col)
            else:
                fig.add_trace(hl)
    return fig, trace_types


def _render_maps(seq_idx: int, view: str, picked_type: str | None,
                 wt_onehot: np.ndarray | None = None) -> None:
    """Bottom-of-page logos. If `wt_onehot` (230, 4) is given, each map is
    projected onto the WT base at each position (attr * onehot), matching the
    kcee `WT-projected logo (attr × onehot)` toggle."""
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
        plot_maps = {k: (pack[k] * wt_onehot if wt_onehot is not None else pack[k]) for k in rows}
        all_vals = np.concatenate([plot_maps[k].ravel() for k in rows])
        yabs = float(np.max(np.abs(all_vals))) * 1.05 or 1.0
        ylim = (-yabs, yabs)
        for r, key in enumerate(rows):
            label = f"{ct} · {TYPE_LABEL[key]}"
            if key == "background":
                label += f" (ref cluster {pack['ref_cluster_idx']})"
            if wt_onehot is not None:
                label += " · WT-projected"
            highlight = (picked_type == key)
            fast_logo(plot_maps[key], ax=axes[r, col], ylim=ylim)
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

    # Sidebar: SEAM-specific selectors
    st.sidebar.header("SEAM view")
    view = st.sidebar.radio("show", VIEW_OPTIONS, index=0, key="seam_view",
                            help="all three: gray WT + red foreground + blue background per seq.")
    cond_options = ("all", "diff-diff", "same-diff", "same-same")
    cond = st.sidebar.selectbox("condition filter", cond_options, index=0, key="seam_cond")
    show_wt_logo = st.sidebar.checkbox("WT-projected logo (attr × onehot)", value=False,
                                       key="seam_show_wt_logo",
                                       help="Multiply each attribution map by the WT one-hot "
                                            "(same toggle as kcee mode).")

    mask = np.ones(len(df), dtype=bool)
    if cond != "all":
        mask &= (df["condition"].values == cond)
    finite = np.isfinite(df["HepG2_log2FC"].values) & np.isfinite(df["K562_log2FC"].values)
    mask &= finite
    visible_types = VIEW_TO_TYPES[view]

    st.markdown(f"**{int(mask.sum())} sequences** · view: `{view}` · condition: `{cond}`")
    legend = " / ".join(f'<span style="color:{TYPE_COLOR[t]}">{TYPE_LABEL[t]}</span>' for t in TYPES)
    st.caption(
        f"mech (y): cossim(HepG2 attr, K562 attr) on z-normalised importance over the 200-bp var region.  "
        f"func (x): measured log2FC HepG2 − K562.  type colors: {legend}.  "
        f"marker shape: circle=WT, △=foreground, ▽=background.",
        unsafe_allow_html=True,
    )

    _ctrl, _center = st.columns([1, 4])
    with _ctrl:
        with st.container(border=True):
            st.markdown("**Plot controls**")
            color_mode = st.selectbox("color by", _COLOR_MODES, index=0, key="seam_pc_color")
            plot_colorscale = st.selectbox(
                "colorscale",
                ["Viridis", "Plasma", "Magma", "Turbo", "Cividis", "RdBu_r", "Inferno"],
                index=0, key="seam_pc_cmap",
            )
            plot_clip_mode = st.radio("color clipping", ["auto (2–98%)", "manual", "full range"],
                                      index=0, key="seam_pc_clip")
            plot_vmin: float | None = None
            plot_vmax: float | None = None
            if plot_clip_mode == "manual":
                plot_vmin = float(st.number_input("vmin", value=0.0, key="seam_pc_vmin"))
                plot_vmax = float(st.number_input("vmax", value=1.0, key="seam_pc_vmax"))
            plot_fig_w = int(st.slider("figure width (px)", 400, 2400, 900, 50, key="seam_pc_w"))
            plot_fig_h = int(st.slider("figure height (px)", 300, 1200, 600, 50, key="seam_pc_h"))
            _MARG = ["none", "counts", "density", "probability"]
            plot_marg_x = st.selectbox("marginal x", _MARG, index=0, key="seam_pc_mx")
            plot_marg_y = st.selectbox("marginal y", _MARG, index=0, key="seam_pc_my")
            plot_auto_lims = st.checkbox("auto axis limits", value=True, key="seam_pc_autolims")
            plot_xmin = plot_xmax = plot_ymin = plot_ymax = None
            if not plot_auto_lims:
                _xc1, _xc2 = st.columns(2)
                plot_xmin = float(_xc1.number_input("xmin", value=-3.0, step=0.1, key="seam_pc_xmin"))
                plot_xmax = float(_xc2.number_input("xmax", value=3.0, step=0.1, key="seam_pc_xmax"))
                _yc1, _yc2 = st.columns(2)
                plot_ymin = float(_yc1.number_input("ymin", value=-1.0, step=0.1, key="seam_pc_ymin"))
                plot_ymax = float(_yc2.number_input("ymax", value=1.0, step=0.1, key="seam_pc_ymax"))
            highlight_seq = int(st.number_input("highlight seq_idx", value=-1, step=1,
                                                key="seam_pc_highlight",
                                                help="-1 to disable; otherwise show a red ring at this seq."))

    # Resolve vmin/vmax from the global pool of visible color values when not manual.
    if color_mode != "by type (WT/fg/bg)" and plot_clip_mode != "manual":
        per_seq, _ = _per_seq_color(df, color_mode)
        pool = []
        for tp in visible_types:
            m = mask & np.isfinite(per_seq) & np.isfinite(scores[tp])
            if m.any():
                pool.append(per_seq[m])
        if pool:
            allc = np.concatenate(pool)
            if plot_clip_mode == "auto (2–98%)":
                plot_vmin = float(np.percentile(allc, 2))
                plot_vmax = float(np.percentile(allc, 98))

    with _center:
        fig, trace_types = _scatter_fig(
            df, mask, scores, visible_types, color_mode, plot_colorscale,
            plot_vmin, plot_vmax, plot_fig_w, plot_fig_h, plot_marg_x, plot_marg_y,
            plot_xmin, plot_xmax, plot_ymin, plot_ymax, highlight_seq,
        )
        event = st.plotly_chart(
            fig, use_container_width=False, on_select="rerun",
            selection_mode=("points",),
            key=f"seam_scatter__{view}__{cond}__{color_mode}__{plot_marg_x}__{plot_marg_y}",
        )

    # --- click handler ---
    # customdata is 1D seq_idx; type comes from the trace's curve_number lookup.
    # Streamlit collapses multi-element customdata to a scalar, so we can't pack
    # both fields into customdata directly.
    sel_seq_idx: int | None = None
    sel_type: str | None = None
    sel = getattr(event, "selection", None) if event is not None else None
    if sel and sel.get("points"):
        pts = [p for p in sel["points"] if p.get("customdata") is not None]
        if pts:
            pt = pts[0]
            cd = pt["customdata"]
            sel_seq_idx = int(cd[0]) if isinstance(cd, (list, tuple, np.ndarray)) else int(cd)
            cn = pt.get("curve_number")
            if cn is not None and 0 <= int(cn) < len(trace_types):
                sel_type = trace_types[int(cn)]

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

    wt_oh = None
    if show_wt_logo:
        wt_oh = seq_to_onehot(str(row["sequence"]), length=230, offset=0).T  # (230, 4)
    _render_maps(sel_seq_idx, view, sel_type, wt_onehot=wt_oh)

    st.caption(
        "SEAM space covers 1059 sequences (Pablo's AG models, HepG2 + K562 only). "
        "Other model families / WTC11 are not represented."
    )
