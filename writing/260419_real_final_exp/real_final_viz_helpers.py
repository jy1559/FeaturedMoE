from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing export: {path}")
    return pd.read_csv(path)


def preview_frame(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = list(df.columns)
    print(f"rows={len(df):,} cols={len(cols)}")
    print(cols)
    return df.head(n)


def paper_axes(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    return fig, ax
