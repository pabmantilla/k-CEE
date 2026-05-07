"""Attribution logo plotting with white background, optional WT projection,
and optional finemo hit underlines."""
import matplotlib.pyplot as plt
import numpy as np

try:
    import logomaker
    HAVE_LOGOMAKER = True
except ImportError:
    HAVE_LOGOMAKER = False


ALPHABET = ["A", "C", "G", "T"]
NT_COLOR = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}


def plot_attribution(
    attr: np.ndarray,
    wt_onehot: np.ndarray | None = None,
    hits: list[dict] | None = None,
    title: str = "",
    figsize=(14, 1.8),
    proj_only_first: int | None = None,
):
    """Plot a (4, L) attribution map as a sequence logo on a white background.

    Args:
        attr: (4, L) attribution.
        wt_onehot: (4, L) one-hot. If given, attr is projected: only the
            attribution at the WT base is shown per position. If a position
            has no WT base (all zeros), it is skipped (raw attr stays).
        hits: list of {start, end, motif, strand}. Drawn as underline brackets.
        title: figure title.
        proj_only_first: if set, project only the first N positions; positions
            >= N keep the raw attribution. Useful when WT seq covers only the
            enhancer region (e.g. 230) but attr covers a longer span (e.g. 281).
    """
    assert attr.shape[0] == 4, f"Expected (4, L), got {attr.shape}"
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

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    if HAVE_LOGOMAKER:
        import pandas as pd
        df = pd.DataFrame(a.T, columns=ALPHABET)
        logomaker.Logo(df, ax=ax, color_scheme="classic")
    else:
        x = np.arange(L)
        bottom_pos = np.zeros(L)
        bottom_neg = np.zeros(L)
        for i, nt in enumerate(ALPHABET):
            v = a[i]
            pos = np.where(v > 0, v, 0.0)
            neg = np.where(v < 0, v, 0.0)
            ax.bar(x, pos, bottom=bottom_pos, color=NT_COLOR[nt], width=1.0)
            ax.bar(x, neg, bottom=bottom_neg, color=NT_COLOR[nt], width=1.0)
            bottom_pos = bottom_pos + pos
            bottom_neg = bottom_neg + neg

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
            color = "#222"
            ax.plot([s, e - 1], [underline_y, underline_y], lw=2.5, color=color, solid_capstyle="butt")
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
    for spine in ax.spines.values():
        spine.set_color("#222")
    fig.tight_layout()
    return fig
