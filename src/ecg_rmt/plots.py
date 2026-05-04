from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .rmt import marchenko_pastur_pdf


COLORS = {
    "normal": "#2F6F73",
    "abnormal": "#B45F3C",
    "blue": "#3B5B8A",
    "green": "#4F8A5B",
    "gold": "#C99A19",
    "red": "#B9412A",
    "gray": "#4F5B62",
}


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 250,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_class_balance(frame: pd.DataFrame, output: Path) -> None:
    set_plot_style()
    counts = frame["target_name"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(counts.index, counts.values, color=[COLORS["abnormal"], COLORS["normal"]])
    ax.set_ylabel("records")
    ax.set_title("Target balance")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def save_covariance_spectrum(spectrum: pd.DataFrame, output: Path) -> None:
    set_plot_style()
    eigenvalues = spectrum["eigenvalue"].to_numpy(float)
    components = spectrum["component"].to_numpy(float)
    mp_lower = float(spectrum["mp_lower"].iloc[0])
    mp_upper = float(spectrum["mp_upper"].iloc[0])
    ratio = float(spectrum["aspect_ratio_p_over_n"].iloc[0])
    outlier = eigenvalues > mp_upper

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    rank_ax, hist_ax = axes
    rank_ax.axhspan(mp_lower, mp_upper, color=COLORS["blue"], alpha=0.12)
    rank_ax.scatter(components[~outlier], eigenvalues[~outlier], s=8, color=COLORS["blue"])
    rank_ax.scatter(components[outlier], eigenvalues[outlier], s=10, color=COLORS["red"])
    rank_ax.axhline(mp_upper, color=COLORS["red"], linestyle="--", linewidth=1)
    rank_ax.set_yscale("log")
    rank_ax.set_xlabel("component rank")
    rank_ax.set_ylabel("eigenvalue")
    rank_ax.set_title("Standardized covariance spectrum")

    visible = eigenvalues[eigenvalues <= max(mp_upper * 2.5, np.percentile(eigenvalues, 90))]
    hist_ax.hist(visible, bins=40, density=True, alpha=0.7, color=COLORS["blue"])
    xs = np.linspace(max(mp_lower, 1e-9), mp_upper, 400)
    hist_ax.plot(xs, marchenko_pastur_pdf(xs, ratio=ratio), color=COLORS["red"])
    hist_ax.axvspan(mp_lower, mp_upper, color=COLORS["blue"], alpha=0.12)
    hist_ax.set_xlabel("eigenvalue")
    hist_ax.set_ylabel("density")
    hist_ax.set_title("MP bulk window")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
