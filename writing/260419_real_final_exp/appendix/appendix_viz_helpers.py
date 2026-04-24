from __future__ import annotations

import json
from collections.abc import Iterable
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

PALETTE = {
    "ink": "#20313B",
    "muted": "#66757D",
    "grid": "#D9D8D2",
    "paper": "#FCFCFA",
    "badge": "#F3EEE4",
    "route": "#0F766E",
    "blue": "#5B7C99",
    "orange": "#D28B36",
    "rose": "#C96567",
    "gold": "#D9B44A",
    "plum": "#8E6C8A",
}

DATASET_LABELS = {
    "beauty": "Beauty",
    "foursquare": "Foursquare",
    "KuaiRecLargeStrictPosV2_0.2": "KuaiRec",
    "lastfm0.03": "LastFM",
    "movielens1m": "ML-1M",
    "retail_rocket": "Retail Rocket",
    "beauty_to_kuairec": "Beauty→KuaiRec",
}

OURS_SUFFIX = " (ours)"

SHARED_DATASET_METRIC_LIMITS = {
    "KuaiRecLargeStrictPosV2_0.2": {
        "ndcg20": (0.2666+0.05, 0.3538+0.05),
        "hr10": (0.3061+0.05, 0.3669+0.05),
    },
    "foursquare": {
        "ndcg20": (0.1580+0.05, 0.2184+0.05),
        "hr10": (0.2180+0.05, 0.3215+0.05),
    },
}
SUBFIG_SIZE = (4.45, 3.3)
LEGEND_STRIP_SIZE = (8.25, 0.82)
LEGEND_STRIP_HALF_SIZE = (4.1, 0.8)


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.12)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": PALETTE["paper"],
            "axes.edgecolor": PALETTE["muted"],
            "axes.linewidth": 0.9,
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 16.2,
            "axes.titleweight": "semibold",
            "axes.labelsize": 15.6,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 13.0,
            "ytick.labelsize": 13.0,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.72,
            "legend.frameon": True,
            "legend.fontsize": 13.2,
            "legend.facecolor": "white",
            "legend.edgecolor": PALETTE["muted"],
            "legend.framealpha": 0.92,
            "figure.dpi": 140,
            "savefig.dpi": 260,
        }
    )


def resolve_data_path(name: str | Path) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return DATA_DIR / path


def load_csv(name: str) -> pd.DataFrame:
    path = resolve_data_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing export: {path}")
    return pd.read_csv(path)


def load_json(name: str) -> dict:
    path = resolve_data_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing export: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preview_frame(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = list(df.columns)
    print(f"rows={len(df):,} cols={len(cols)}")
    print(cols)
    return df.head(n)


def dataset_label(raw: str) -> str:
    return DATASET_LABELS.get(raw, raw)


def clean_axes(ax: plt.Axes, grid_axis: str = "y") -> plt.Axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["muted"])
    ax.spines["bottom"].set_color(PALETTE["muted"])
    ax.grid(axis=grid_axis, color=PALETTE["grid"], alpha=0.72)
    return ax


def single_subfigure_axes(figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize or SUBFIG_SIZE)
    fig.subplots_adjust(left=0.18, right=0.835, bottom=0.19, top=0.91)
    clean_axes(ax)
    return fig, ax


def legend_strip_axes(figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize or LEGEND_STRIP_SIZE)
    ax.axis("off")
    return fig, ax


def half_legend_strip_axes(figsize: tuple[float, float] | None = None) -> tuple[plt.Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize or LEGEND_STRIP_HALF_SIZE)
    ax.axis("off")
    return fig, ax


def mark_ours_first(order: list[str]) -> list[str]:
    if not order:
        return []
    marked = list(order)
    marked[0] = f"{marked[0]}{OURS_SUFFIX}"
    return marked


def y_limits_like_q2(values: Iterable[float], lower_pad: float = 0.55, upper_pad: float = 0.18) -> tuple[float, float]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return (0.0, 1.0)
    lo = float(series.min())
    hi = float(series.max())
    span = max(hi - lo, 0.004)
    return lo - span * lower_pad, hi + span * upper_pad


def _canonical_metric_key(metric_name: str) -> str | None:
    name = str(metric_name).strip().lower()
    if name in {"test_ndcg20", "test_ndcg_20", "ndcg20", "ndcg@20"}:
        return "ndcg20"
    if name in {"test_hit10", "test_hr10", "hr10", "hr@10", "hit10", "hit@10"}:
        return "hr10"
    return None


def _single_dataset_key(df: pd.DataFrame) -> str | None:
    if "dataset" not in df.columns:
        return None
    datasets = df["dataset"].dropna().astype(str).unique().tolist()
    if len(datasets) != 1:
        return None
    return datasets[0]


def _round_axis_limit(value: float, direction: str, decimals: int = 4) -> float:
    scale = 10 ** decimals
    if direction == "down":
        return math.floor(value * scale) / scale
    return math.ceil(value * scale) / scale


def shared_dataset_metric_limits(
    df: pd.DataFrame,
    metric_name: str,
    values: Iterable[float],
    lower_pad: float = 0.55,
    upper_pad: float = 0.18,
) -> tuple[float, float]:
    dataset_key = _single_dataset_key(df)
    metric_key = _canonical_metric_key(metric_name)
    if dataset_key is None or metric_key is None:
        return y_limits_like_q2(values, lower_pad=lower_pad, upper_pad=upper_pad)

    dataset_limits = SHARED_DATASET_METRIC_LIMITS.get(dataset_key, {})
    base_limits = dataset_limits.get(metric_key)
    if base_limits is None:
        return y_limits_like_q2(values, lower_pad=lower_pad, upper_pad=upper_pad)

    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return base_limits

    data_min = float(series.min())
    data_max = float(series.max())

    # 기본적으로 CANONICAL 범위를 사용하되, 
    # 데이터가 CANONICAL 범위를 벗어나는 경우에만 확장합니다.
    lower = min(base_limits[0], data_min)
    upper = max(base_limits[1], data_max)

    return _round_axis_limit(lower, "down"), _round_axis_limit(upper, "up")


def remove_panel_label_space(ax: plt.Axes) -> None:
    ax.set_title(ax.get_title(), pad=8)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.97,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["ink"],
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
        ),
        zorder=10,
    )


def annotate_demo(fig: plt.Figure, df: pd.DataFrame) -> None:
    if "data_status" not in df.columns or df.empty:
        return
    status = str(df["data_status"].iloc[0]).strip()
    if not status:
        return
    fig.text(
        0.012,
        0.988,
        f"data_status={status}",
        ha="left",
        va="top",
        fontsize=8.5,
        color=PALETTE["muted"],
        bbox={"boxstyle": "round,pad=0.22", "facecolor": PALETTE["badge"], "edgecolor": PALETTE["grid"]},
    )


def metric_limits(values: Iterable[float], padding: float = 0.12, floor: float | None = None) -> tuple[float, float]:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return (0.0, 1.0)
    lo = float(series.min())
    hi = float(series.max())
    span = max(hi - lo, 0.01)
    pad = span * padding
    lower = lo - pad
    upper = hi + pad
    if floor is not None:
        lower = max(lower, floor)
    return lower, upper


def palette_for(labels: Iterable[str]) -> dict[str, str]:
    fallback = [PALETTE["route"], PALETTE["blue"], PALETTE["orange"], PALETTE["rose"], PALETTE["gold"], PALETTE["plum"]]
    mapping: dict[str, str] = {}
    for idx, label in enumerate(labels):
        label_text = str(label).lower()
        if "full" in label_text or "semantic" in label_text or "route" in label_text:
            mapping[str(label)] = PALETTE["route"]
        elif "no" in label_text or "baseline" in label_text or "sasrec" in label_text:
            mapping[str(label)] = PALETTE["blue"]
        elif "consistency" in label_text or "top-2" in label_text or "memory" in label_text:
            mapping[str(label)] = PALETTE["orange"]
        elif "z-loss" in label_text or "shuffle" in label_text or "tempo" in label_text:
            mapping[str(label)] = PALETTE["rose"]
        else:
            mapping[str(label)] = fallback[idx % len(fallback)]
    return mapping


def metric_legend_handles() -> list[object]:
    return [
        Patch(facecolor="#8FA6DE", edgecolor="#6178AE", alpha=0.92, label="NDCG@20"),
        Line2D([0], [0], color="#C33245", marker="o", linewidth=2.2, markersize=6.5, label="HR@10"),
    ]


def add_metric_legend(ax: plt.Axes, loc: str = "lower right") -> None:
    ax.legend(
        handles=metric_legend_handles(),
        loc=loc,
        frameon=True,
        fancybox=False,
        borderpad=0.48,
        labelspacing=0.32,
        handlelength=2.0,
        handletextpad=0.62,
        prop={"size": plt.rcParams["legend.fontsize"] - 1},
    )


def add_legend_strip(
    ax: plt.Axes,
    category_labels: list[str],
    category_colors: list[str],
    ncol: int | None = None,
) -> None:
    handles = [
        Patch(facecolor=color, edgecolor="white", linewidth=0.8, label=label)
        for label, color in zip(category_labels, category_colors)
    ]
    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        ncol=ncol or len(handles),
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.75,
        borderaxespad=0.0,
        prop={"size": plt.rcParams["legend.fontsize"]},
    )


def bar_line_panel(
    df: pd.DataFrame,
    category_col: str,
    bar_col: str,
    line_col: str,
    ax: plt.Axes,
    order: list[str] | None = None,
    bar_label: str = "NDCG@20",
    line_label: str = "HR@10",
    xrotation: int = 20,
    palette_override: dict[str, str] | None = None,
    show_xticklabels: bool = True,
    add_metric_legend_box: bool = False,
    metric_legend_loc: str = "lower right",
    bar_limits: tuple[float, float] | None = None,
    line_limits: tuple[float, float] | None = None,
) -> tuple[plt.Axes, plt.Axes]:
    """Dual-axis panel: coloured bars for *bar_col*, black line+markers for *line_col*.
    Matches the main-body Q2 style.  No title or panel letter is added — add externally if needed."""
    work = df.copy().dropna(subset=[category_col])
    work[category_col] = work[category_col].astype(str)
    order_values = order if order is not None else work[category_col].unique().tolist()

    palette = palette_override or palette_for(order_values)
    x = np.arange(len(order_values), dtype=float)

    bar_vals = [float(pd.to_numeric(work[work[category_col] == lbl][bar_col], errors="coerce").mean())
                if lbl in work[category_col].values else np.nan for lbl in order_values]
    line_vals = [float(pd.to_numeric(work[work[category_col] == lbl][line_col], errors="coerce").mean())
                 if lbl in work[category_col].values else np.nan for lbl in order_values]

    ax.bar(x, bar_vals, width=0.62,
           color=[palette[lbl] for lbl in order_values],
           alpha=0.82, edgecolor="white", linewidth=0.8, zorder=2)
    twin = ax.twinx()
    twin.plot(x, line_vals, color="#C33245", marker="o",
              linewidth=2.2, markersize=6.5, markeredgecolor=PALETTE["ink"], markeredgewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        order_values if show_xticklabels else [""] * len(order_values),
        rotation=xrotation,
        ha="right" if xrotation else "center",
    )
    if not show_xticklabels:
        ax.tick_params(axis="x", length=0)
    ax.set_ylabel(bar_label)
    twin.set_ylabel(line_label)
    ax.set_ylim(*(bar_limits or shared_dataset_metric_limits(work, bar_col, bar_vals)))
    twin.set_ylim(*(line_limits or shared_dataset_metric_limits(work, line_col, line_vals, lower_pad=0.45, upper_pad=0.18)))
    ax.margins(x=0.06)
    clean_axes(ax)
    twin.grid(False)
    twin.spines["top"].set_visible(False)
    twin.spines["right"].set_color(PALETTE["muted"])
    if add_metric_legend_box:
        add_metric_legend(ax, loc=metric_legend_loc)
    return ax, twin
