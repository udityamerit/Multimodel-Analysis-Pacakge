"""Shared plotting helper: a simple ranked bar chart for comparing many models at once."""
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

_BAR = "#8fb3c9"      # every model
_BEST = "#1f5f8b"     # the best model
_INK = "#2b2b2b"


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.3f}" if abs(value) < 1000 else f"{value:,.0f}"


def comparison_figure(
    names: Sequence[str],
    values: Sequence[float],
    metric: str,
    title: str,
    lower_is_better: bool = False,
    footnote: Optional[str] = None,
):
    """Horizontal bar chart of one metric, best model on top, value printed at the end of each bar.

    Returns the figure; the caller saves / shows it.
    """
    names = [str(n) for n in names]
    vals = np.array([np.nan if v is None else float(v) for v in values])
    # best first; missing values go last
    sort_key = np.where(np.isfinite(vals), vals if lower_is_better else -vals, np.inf)
    order = np.argsort(sort_key, kind="stable")
    names = [names[i] for i in order]
    vals = vals[order]
    n = len(names)

    fig, ax = plt.subplots(figsize=(9.5, max(3.6, 0.42 * n + 1.6)))
    ys = np.arange(n)
    colors = [_BEST if i == 0 else _BAR for i in range(n)]
    ax.barh(ys, np.nan_to_num(vals, nan=0.0), color=colors, height=0.68)

    finite = vals[np.isfinite(vals)]
    lo = min(0.0, finite.min()) if finite.size else 0.0
    hi = max(0.0, finite.max()) if finite.size else 1.0
    pad = (hi - lo) * 0.012 or 0.01
    for y, v in zip(ys, vals):
        ax.text(max(v, 0.0) + pad if np.isfinite(v) else pad, y, _fmt(v), va="center", ha="left",
                fontsize=10, color=_INK, fontweight="bold" if y == 0 else "normal")

    ax.set_xlim(lo - (hi - lo) * 0.02 if lo < 0 else 0, hi + (hi - lo) * 0.12 or 1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=10.5, color=_INK)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric}  ({'lower' if lower_is_better else 'higher'} is better)", fontsize=10)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=_INK, pad=12)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if footnote:
        fig.text(0.01, 0.005, footnote, ha="left", va="bottom", fontsize=8.5, color="#777777")
    fig.tight_layout(rect=(0, 0.03 if footnote else 0, 1, 1))
    return fig
