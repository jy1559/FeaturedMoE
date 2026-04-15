from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import seaborn as sns

PAPER_COLORS = {
    "RouteRec": "#0F766E",
    "shared_ffn": "#4C78A8",
    "hidden_only": "#F58518",
    "mixed": "#E45756",
    "behavior_only": "#0F766E",
    "SASRec": "#4C78A8",
    "BSARec": "#72B7B2",
    "DuoRec": "#F58518",
    "FEARec": "#54A24B",
    "SIGMA": "#B279A2",
    "GRU4Rec": "#9D755D",
    "TiSASRec": "#ECA82C",
}

DEFAULT_SEQUENCE = [
    "#0F766E",
    "#4C78A8",
    "#F58518",
    "#E45756",
    "#54A24B",
    "#72B7B2",
    "#B279A2",
    "#9D755D",
    "#ECA82C",
]


def build_palette(labels: Iterable[str] | None = None) -> dict[str, str] | list[str]:
    if labels is None:
        return DEFAULT_SEQUENCE.copy()

    palette: dict[str, str] = {}
    fallback_index = 0
    for label in labels:
        if label in PAPER_COLORS:
            palette[label] = PAPER_COLORS[label]
            continue
        palette[label] = DEFAULT_SEQUENCE[fallback_index % len(DEFAULT_SEQUENCE)]
        fallback_index += 1
    return palette


def set_paper_theme(context: str = "notebook", font_scale: float = 1.0) -> None:
    sns.set_theme(
        style="whitegrid",
        context=context,
        font_scale=font_scale,
        palette=DEFAULT_SEQUENCE,
    )
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )