"""Attribution logo plotting with white background, optional WT projection,
and optional finemo hit underlines.

Logo glyphs are rendered with `fast_logo`, a cached-geometry implementation
that reuses pre-computed Matplotlib TextPath outlines for A/C/G/T. This
avoids the per-call setup cost of logomaker and is roughly an order of
magnitude faster for the (4, L) inputs the viewer renders on every click.
"""
from __future__ import annotations

import hashlib
import io

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath


_DNA_COLORS: dict[str, tuple[float, float, float]] = {
    "A": (0.0, 0.5, 0.0),
    "C": (0.0, 0.0, 1.0),
    "G": (1.0, 0.65, 0.0),
    "T": (1.0, 0.0, 0.0),
}


class _GlyphCache:
    def __init__(self) -> None:
        self.verts: dict[str, np.ndarray] = {}
        self.codes: dict[str, np.ndarray] = {}
        self.xmin: dict[str, float] = {}
        self.ymin: dict[str, float] = {}
        self.w: dict[str, float] = {}
        self.h: dict[str, float] = {}
        self.flip_verts: dict[str, np.ndarray] = {}
        self.flip_ymin: dict[str, float] = {}
        self.flip_h: dict[str, float] = {}
        self.ref_w: float = 0.0
        self.ready: bool = False

    def build(
        self,
        font_name: str = "sans",
        font_weight: str = "bold",
        ref_char: str = "E",
    ) -> None:
        if self.ready:
            return
        fp = fm.FontProperties(family=font_name, weight=font_weight)
        for ch in "ACGT":
            tp = TextPath((0, 0), ch, size=1, prop=fp)
            ext = tp.get_extents()
            v = np.array(tp.vertices, dtype=np.float64)
            self.verts[ch] = v
            self.codes[ch] = np.array(tp.codes, dtype=np.uint8)
            self.xmin[ch] = float(ext.xmin)
            self.ymin[ch] = float(ext.ymin)
            self.w[ch] = float(ext.width)
            self.h[ch] = float(ext.height)
            fv = v.copy()
            fv[:, 1] = -fv[:, 1]
            self.flip_verts[ch] = fv
            self.flip_ymin[ch] = float(fv[:, 1].min())
            self.flip_h[ch] = float(fv[:, 1].max()) - self.flip_ymin[ch]
        self.ref_w = TextPath((0, 0), ref_char, size=1, prop=fp).get_extents().width
        self.ready = True


_CACHE = _GlyphCache()


def fast_logo(
    values: np.ndarray,
    ax: Axes,
    width: float = 0.95,
    height_scale: float = 1.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Render a single attribution logo on a Matplotlib axis.

    `values` is (L, 4) with columns A/C/G/T. Positive values stack upward;
    negative values are drawn flipped and stack downward.
    """
    _CACHE.build()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Expected values shape (L, 4), got {values.shape}")

    seq_len = values.shape[0]
    chars = list("ACGT")
    patches: list[PathPatch] = []
    facecolors: list[tuple[float, float, float]] = []
    y_min = 0.0
    y_max = 0.0
    eps = max(1e-6, 1e-4 * float(np.max(np.abs(values))) if values.size else 0.0)

    for pos in range(seq_len):
        vs = values[pos] * float(height_scale)
        order = np.argsort(vs)
        vs_sorted = vs[order]
        cs = [chars[i] for i in order]

        floor = float(np.sum(vs_sorted[vs_sorted < 0]))
        pos_min = floor

        for v, ch in zip(vs_sorted, cs):
            h = abs(float(v))
            if h < eps:
                continue
            ceiling = floor + h
            flip = v < 0
            bx = pos - width / 2.0

            if flip:
                vt = _CACHE.flip_verts[ch]
                oy, oh = _CACHE.flip_ymin[ch], _CACHE.flip_h[ch]
            else:
                vt = _CACHE.verts[ch]
                oy, oh = _CACHE.ymin[ch], _CACHE.h[ch]
            ow = _CACHE.w[ch]
            ox = _CACHE.xmin[ch]

            hstretch = min(width / ow, width / _CACHE.ref_w)
            cw = hstretch * ow
            shift = (width - cw) / 2.0
            vstretch = h / oh

            new_verts = vt.copy()
            new_verts[:, 0] = (vt[:, 0] - ox) * hstretch + bx + shift
            new_verts[:, 1] = (vt[:, 1] - oy) * vstretch + floor

            patches.append(PathPatch(MplPath(new_verts, _CACHE.codes[ch])))
            facecolors.append(_DNA_COLORS[ch])
            floor = ceiling

        pos_max = floor
        y_min = min(y_min, pos_min)
        y_max = max(y_max, pos_max)

    pc = PatchCollection(
        patches,
        match_original=False,
        facecolors=facecolors,
        edgecolors="none",
        linewidths=0,
    )
    ax.add_collection(pc)
    ax.set_xlim(-0.5, seq_len - 0.5)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        if y_max == y_min:
            y_max = y_min + 1.0
        pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)


def plot_attribution(
    attr: np.ndarray,
    wt_onehot: np.ndarray | None = None,
    hits: list[dict] | None = None,
    title: str = "",
    figsize=(14, 1.8),
    proj_only_first: int | None = None,
    crop: tuple[int, int] | None = None,
):
    """Plot a (4, L) attribution map as a sequence logo on a white background.

    Args:
        attr: (4, L) attribution.
        wt_onehot: (4, L) one-hot. If given, attr is projected to WT base.
        hits: list of {start, end, motif, strand} in the input attr frame.
        title: figure title.
        proj_only_first: if set, project only the first N positions.
        crop: (start, stop) slice into the input attr/wt_onehot to display.
            Hit coordinates are shifted by -start and clipped to [0, stop-start].
            If None, the full attr is shown.
    """
    assert attr.shape[0] == 4, f"Expected (4, L), got {attr.shape}"
    crop_start = 0
    if crop is not None:
        crop_start, crop_stop = int(crop[0]), int(crop[1])
        crop_start = max(0, crop_start)
        crop_stop = min(attr.shape[1], crop_stop)
        if crop_stop > crop_start:
            attr = attr[:, crop_start:crop_stop]
            if wt_onehot is not None:
                wt_onehot = wt_onehot[:, crop_start:crop_stop]
    L = attr.shape[1]
    a = attr.copy()
    if wt_onehot is not None:
        wt = wt_onehot
        if proj_only_first is not None:
            mask_pos = np.zeros(L, dtype=bool)
            mask_pos[: min(proj_only_first, L)] = True
            wt_has = wt.sum(axis=0) > 0
            do_proj = mask_pos & wt_has
        else:
            wt_has = wt.sum(axis=0) > 0
            do_proj = wt_has
        if do_proj.any():
            a[:, do_proj] = a[:, do_proj] * wt[:, do_proj]
    if hits and crop_start > 0:
        shifted: list[dict] = []
        for h in hits:
            s = int(h["start"]) - crop_start
            e = int(h["end"]) - crop_start
            if e <= 0 or s >= L:
                continue
            shifted.append({**h, "start": max(0, s), "end": min(L, e)})
        hits = shifted

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    fast_logo(a.T, ax=ax)

    if hits:
        ymin = float(np.minimum(a.min(), 0.0))
        ymax = float(np.maximum(a.max(), 0.0))
        span = max(ymax - ymin, 1e-6)
        underline_y = ymin - 0.10 * span
        ax.set_ylim(ymin - 0.20 * span, ymax + 0.05 * span)
        for h in hits:
            s = max(0, int(h["start"]))
            e = min(L, int(h["end"]))
            if e <= s:
                continue
            ax.plot([s, e - 1], [underline_y, underline_y], lw=2.5, color="#222", solid_capstyle="butt")
            label = h.get("motif", "")
            if label.startswith("pos_patterns.pattern_"):
                label = "p" + label.split("pattern_")[-1]
            elif label.startswith("neg_patterns.pattern_"):
                label = "n" + label.split("pattern_")[-1]
            ax.text((s + e - 1) / 2, underline_y - 0.03 * span, label,
                    ha="center", va="top", fontsize=7, color="#444")

    ax.set_title(title, fontsize=10, color="#111")
    ax.set_xlabel("Position", color="#111")
    ax.set_ylabel("Attr", color="#111")
    ax.tick_params(colors="#111")
    _ylo, _yhi = ax.get_ylim()
    if _ylo < 0 < _yhi:
        ax.set_yticks([_ylo, 0.0, _yhi])
    else:
        ax.set_yticks([_ylo, (_ylo + _yhi) / 2.0, _yhi])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.2g}"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.22)
    return fig


def cached_attribution_png(*, path: str, key: str, idx: int, hits_signature: tuple, show_wt_logo: bool, **plot_kwargs) -> bytes:
    """Render plot_attribution to PNG bytes, cached in session_state by inputs.
    hits_signature must be hashable (e.g. tuple of (start, end, motif) tuples).
    plot_kwargs are forwarded to plot_attribution(...) AS-IS.
    """
    cache = st.session_state.setdefault("_attr_png_cache", {})
    sig_hash = hashlib.md5(repr(hits_signature).encode()).hexdigest()[:12]
    cache_key = (path, key, idx, sig_hash, bool(show_wt_logo))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    fig = plot_attribution(**plot_kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    data = buf.getvalue()
    if len(cache) > 64:
        cache.pop(next(iter(cache)))
    cache[cache_key] = data
    return data
