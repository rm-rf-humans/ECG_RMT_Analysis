from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg_rmt.rmt import marchenko_pastur_pdf


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "recovered_modal"
METRICS = ARTIFACTS / "metrics"
RMT = ARTIFACTS / "rmt"
TABLE = ARTIFACTS / "data" / "modeling_table.csv.gz"
PAPER_FIGURES = ROOT / "paper" / "figures"
PRESENTATION_FIGURES = ROOT / "presentation" / "png_figures"

COLORS = {
    "normal": "#2F6F73",
    "abnormal": "#B45F3C",
    "blue": "#3B5B8A",
    "green": "#4F8A5B",
    "gold": "#C99A19",
    "red": "#B9412A",
    "gray": "#4F5B62",
    "light": "#D8DEE4",
}


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 250,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Modal artifact: {path}")
    return path


def _save(fig: plt.Figure, name: str) -> None:
    for directory in [PAPER_FIGURES, PRESENTATION_FIGURES]:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / name, bbox_inches="tight")
    plt.close(fig)


def _family(column: str) -> str:
    if column.startswith("corr_"):
        return "cross-lead correlation"
    if column.startswith("global_"):
        return "global signal"
    if column in {"age", "sex", "height", "weight"}:
        return "demographics"
    if column.startswith("lead_"):
        feature = "_".join(column.split("_")[2:])
        if feature.startswith("fft_"):
            return "lead spectral"
        if feature.startswith("band_"):
            return "lead band-power"
        if feature.startswith("derivative") or feature in {"rms", "energy", "zero_crossing_rate"}:
            return "lead morphology"
        return "lead amplitude"
    return "metadata flags"


def preprocessing_findings() -> None:
    table = pd.read_csv(_require(TABLE), low_memory=False)
    summary = json.loads(_require(METRICS / "data_summary.json").read_text())

    split = pd.DataFrame(summary["split_counts"]).T.loc[["train", "validation", "test"]]
    split = split.rename(columns={"diagnostic_abnormality": "abnormal"})
    inventory = pd.Series(
        {
            "per-lead waveform": len([c for c in table.columns if c.startswith("lead_")]),
            "cross-lead correlation": len([c for c in table.columns if c.startswith("corr_")]),
            "metadata": 7,
            "global waveform": len([c for c in table.columns if c.startswith("global_")]),
        }
    ).sort_values()
    signal_cols = [
        c for c in table.columns if c.startswith("lead_") or c.startswith("corr_") or c.startswith("global_")
    ]
    signal_missing = table[signal_cols].isna().mean()
    missing = pd.Series(
        {
            "height": table["height"].isna().mean() * 100,
            "weight": table["weight"].isna().mean() * 100,
            "nurse": table["nurse"].isna().mean() * 100,
            "site": table["site"].isna().mean() * 100,
            "max signal feature": signal_missing.max() * 100,
            "mean signal feature": signal_missing.mean() * 100,
        }
    ).sort_values()
    age_groups = [
        table.loc[table["target_name"] == "normal", "age"].dropna().to_numpy(),
        table.loc[table["target_name"] == "diagnostic_abnormality", "age"].dropna().to_numpy(),
    ]

    numeric = table.select_dtypes(include=[np.number])
    candidate_cols = [
        c for c in signal_cols + ["age", "sex", "height", "weight", "nurse", "site"]
        if c in numeric.columns and float(numeric[c].std(skipna=True)) > 0
    ]
    target_corr = numeric[candidate_cols].corrwith(table["target"]).abs().dropna().sort_values(ascending=False)
    family = target_corr.groupby(target_corr.index.map(_family)).mean().sort_values()
    top = target_corr.head(10).sort_values()

    fig = plt.figure(figsize=(8.75, 5.55))
    axes = fig.subplots(2, 3)
    split_ax, inventory_ax, missing_ax, age_ax, family_ax, top_ax = axes.ravel()

    bottom = np.zeros(len(split))
    for column, color in [("normal", COLORS["normal"]), ("abnormal", COLORS["abnormal"])]:
        values = split[column].to_numpy()
        split_ax.bar(split.index, values, bottom=bottom, color=color, label=column)
        bottom += values
    for idx, total in enumerate(bottom):
        split_ax.text(idx, total + 350, f"{split.iloc[idx]['abnormal'] / total:.1%}", ha="center", fontsize=8)
    split_ax.set_title("Split prevalence")
    split_ax.set_ylabel("records")
    split_ax.legend(frameon=False)

    inventory_ax.barh(inventory.index, inventory.values, color=COLORS["blue"])
    inventory_ax.set_title("Feature inventory before encoding")
    inventory_ax.set_xlabel("raw columns")
    for y, value in enumerate(inventory.values):
        inventory_ax.text(value + 5, y, f"{int(value)}", va="center", fontsize=8)

    missing_ax.barh(missing.index, missing.values, color=COLORS["gold"])
    missing_ax.set_xscale("symlog", linthresh=0.01)
    missing_ax.set_xlim(0, 100)
    missing_ax.set_title("Missingness before imputation")
    missing_ax.set_xlabel("missing records (%)")
    for y, value in enumerate(missing.values):
        label = f"{value:.4f}%" if value < 0.01 else f"{value:.3g}%"
        missing_ax.text(max(value * 1.1, 0.018), y, label, va="center", fontsize=7)

    box = age_ax.boxplot(
        age_groups,
        tick_labels=["normal", "abnormal"],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], [COLORS["normal"], COLORS["abnormal"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for idx, values in enumerate(age_groups, start=1):
        median = np.median(values)
        age_ax.text(idx, median + 4, f"median {median:.0f}", ha="center", fontsize=7)
    age_ax.set_title("Age by constructed target")
    age_ax.set_ylabel("age")

    family_ax.barh(family.index, family.values, color=COLORS["green"])
    family_ax.set_title("Association by feature family")
    family_ax.set_xlabel("mean |corr(target, feature)|")
    for y, value in enumerate(family.values):
        family_ax.text(value + 0.004, y, f"{value:.3f}", va="center", fontsize=7)

    top_ax.barh([name.replace("_", " ") for name in top.index], top.values, color=COLORS["red"])
    top_ax.set_title("Top univariate target associations")
    top_ax.set_xlabel("|Pearson correlation|")
    for y, value in enumerate(top.values):
        top_ax.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=7)

    fig.text(
        0.01,
        0.01,
        "Rows: 21,388; raw feature map: 523 columns; preprocessed design matrix: 594 columns after one-hot expansion.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "paper_preprocessing_findings.png")


def dataset_spectrum() -> None:
    summary = json.loads(_require(METRICS / "data_summary.json").read_text())
    spectrum = pd.read_csv(_require(RMT / "covariance_spectrum.csv"))
    cov = pd.read_csv(_require(RMT / "covariance_summary.csv")).iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.1))
    split_ax, diag_ax, rank_ax, hist_ax = axes.ravel()

    split = pd.DataFrame(summary["split_counts"]).T.loc[["train", "validation", "test"]]
    bottom = np.zeros(len(split))
    for column, color, label in [
        ("normal", COLORS["normal"], "normal"),
        ("diagnostic_abnormality", COLORS["abnormal"], "diagnostic abnormality"),
    ]:
        values = split[column].to_numpy()
        split_ax.bar(split.index, values, bottom=bottom, color=color, label=label)
        bottom += values
    split_ax.set_title("Official PTB-XL fold split")
    split_ax.set_ylabel("records")
    split_ax.legend(frameon=False)

    diag = pd.Series(summary["diagnostic_class_counts"]).sort_values()
    diag_ax.barh(diag.index, diag.values, color=COLORS["blue"])
    diag_ax.set_title("Mapped diagnostic classes")
    diag_ax.set_xlabel("records with class present")
    for y, value in enumerate(diag.values):
        diag_ax.text(value + 80, y, str(int(value)), va="center", fontsize=8)

    eigenvalues = spectrum["eigenvalue"].to_numpy(float)
    components = spectrum["component"].to_numpy(float)
    mp_lower = float(spectrum["mp_lower"].iloc[0])
    mp_upper = float(spectrum["mp_upper"].iloc[0])
    ratio = float(spectrum["aspect_ratio_p_over_n"].iloc[0])
    mask = eigenvalues > mp_upper
    rank_ax.axhspan(mp_lower, mp_upper, color=COLORS["blue"], alpha=0.12, label="MP bulk")
    rank_ax.scatter(components[~mask], eigenvalues[~mask], s=8, color=COLORS["blue"], label="bulk")
    rank_ax.scatter(components[mask], eigenvalues[mask], s=10, color=COLORS["red"], label="outlier")
    rank_ax.axhline(mp_upper, color=COLORS["red"], linestyle="--", linewidth=1)
    rank_ax.axhline(mp_lower, color=COLORS["gray"], linestyle=":", linewidth=1)
    rank_ax.set_yscale("log")
    rank_ax.set_title("Standardized covariance spectrum")
    rank_ax.set_xlabel("component rank")
    rank_ax.set_ylabel("eigenvalue")
    rank_ax.text(
        0.03,
        0.08,
        f"$p/n={cov['aspect_ratio_p_over_n']:.4f}$\n{int(cov['n_signal_components'])} outliers; {cov['signal_variance_fraction']:.1%} variance",
        transform=rank_ax.transAxes,
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": COLORS["light"], "pad": 3},
    )
    rank_ax.legend(frameon=False, ncol=3)

    hist_limit = max(mp_upper * 2.25, float(np.nanpercentile(eigenvalues, 90)))
    visible = eigenvalues[eigenvalues <= hist_limit]
    hist_ax.axvspan(mp_lower, mp_upper, color=COLORS["blue"], alpha=0.12)
    hist_ax.hist(visible, bins=42, density=True, color=COLORS["blue"], alpha=0.65, label="empirical")
    xs = np.linspace(max(mp_lower, 1e-9), mp_upper, 400)
    hist_ax.plot(xs, marchenko_pastur_pdf(xs, ratio=ratio), color=COLORS["red"], label="MP density")
    hist_ax.set_title("Bulk support versus empirical spectrum")
    hist_ax.set_xlabel("eigenvalue")
    hist_ax.set_ylabel("density")
    hist_ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "paper_dataset_spectrum.png")


def model_summary() -> None:
    metrics = pd.read_csv(_require(METRICS / "model_metrics.csv"))
    diagnostics = pd.read_csv(_require(METRICS / "model_rmt_diagnostics.csv"))
    params = json.loads(_require(METRICS / "model_params.json").read_text())
    test = metrics[metrics["split"] == "test"].sort_values("roc_auc")
    diag = diagnostics[diagnostics["split"] == "test"].set_index("model").loc[test["model"]].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    y = np.arange(len(test))
    labels = test["model"].str.replace("_", " ", regex=False)
    for col, color, marker, label in [
        ("roc_auc", COLORS["blue"], "o", "ROC-AUC"),
        ("balanced_accuracy", COLORS["green"], "s", "balanced acc."),
        ("f1", COLORS["gold"], "^", "F1"),
    ]:
        axes[0].scatter(test[col], y, color=color, marker=marker, label=label)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("test score")
    axes[0].set_title("Held-out test performance")
    axes[0].legend(frameon=False)

    first, best = [], []
    for model in test["model"]:
        trials = [trial for trial in params[model]["trials"] if trial.get("value") is not None]
        first.append(float(trials[0]["value"]))
        best.append(float(params[model]["best_cv_roc_auc"]))
    for idx in y:
        axes[1].plot([first[idx], best[idx]], [idx, idx], color=COLORS["light"])
    axes[1].scatter(first, y, color=COLORS["gray"], marker="x", label="first trial")
    axes[1].scatter(best, y, color=COLORS["red"], label="best trial")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("CV ROC-AUC")
    axes[1].set_title("Optuna search trace")
    axes[1].legend(frameon=False)

    for idx, row in diag.iterrows():
        values = [row["bulk_subspace_roc_auc"], row["signal_subspace_roc_auc"], row["original_roc_auc"]]
        axes[2].plot(values, [idx, idx, idx], color=COLORS["light"])
        axes[2].scatter(values[0], idx, color=COLORS["gold"], marker="s", label="bulk" if idx == 0 else None)
        axes[2].scatter(values[1], idx, color=COLORS["green"], marker="^", label="outlier" if idx == 0 else None)
        axes[2].scatter(values[2], idx, color=COLORS["blue"], label="original" if idx == 0 else None)
        axes[2].text(0.925, idx, f"{values[1] / values[2]:.0%}", va="center", fontsize=7)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("test ROC-AUC")
    axes[2].set_title("RMT reconstruction stress test")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "paper_model_rmt_summary.png")


def model_mp_facets() -> None:
    spectrum = pd.read_csv(_require(RMT / "covariance_spectrum.csv"))
    diagnostics = pd.read_csv(_require(METRICS / "model_rmt_diagnostics.csv"))
    diagnostics = diagnostics[diagnostics["split"] == "test"].sort_values("original_roc_auc", ascending=False)
    eigenvalues = spectrum["eigenvalue"].to_numpy(float)
    components = spectrum["component"].to_numpy(float)
    mp_lower = float(spectrum["mp_lower"].iloc[0])
    mp_upper = float(spectrum["mp_upper"].iloc[0])
    ratio = float(spectrum["aspect_ratio_p_over_n"].iloc[0])
    mask = eigenvalues > mp_upper
    xs = np.linspace(mp_lower, mp_upper, 400)
    hist_limit = max(mp_upper * 2.25, float(np.nanpercentile(eigenvalues, 90)))
    visible = eigenvalues[eigenvalues <= hist_limit]
    fig, axes = plt.subplots(len(diagnostics), 3, figsize=(8.6, 8.9), gridspec_kw={"width_ratios": [1.35, 1.05, 1.0]})
    for idx, (_, row) in enumerate(diagnostics.iterrows()):
        rank_ax, hist_ax, score_ax = axes[idx]
        rank_ax.axhspan(mp_lower, mp_upper, color=COLORS["blue"], alpha=0.12)
        rank_ax.scatter(components[~mask], eigenvalues[~mask], s=5, color=COLORS["blue"], alpha=0.65)
        rank_ax.scatter(components[mask], eigenvalues[mask], s=7, color=COLORS["red"], alpha=0.85)
        rank_ax.axhline(mp_upper, color=COLORS["red"], linestyle="--", linewidth=0.8)
        rank_ax.axhline(mp_lower, color=COLORS["gray"], linestyle=":", linewidth=0.8)
        rank_ax.set_yscale("log")
        rank_ax.set_ylabel(str(row["model"]).replace("_", " "), rotation=0, ha="right", va="center", labelpad=42)
        hist_ax.axvspan(mp_lower, mp_upper, color=COLORS["blue"], alpha=0.12)
        hist_ax.hist(visible, bins=35, density=True, color=COLORS["blue"], alpha=0.6)
        hist_ax.plot(xs, marchenko_pastur_pdf(xs, ratio=ratio), color=COLORS["red"], linewidth=1.0)
        hist_ax.set_xlim(0, hist_limit)
        scores = pd.Series({"orig.": row["original_roc_auc"], "outlier": row["signal_subspace_roc_auc"], "bulk": row["bulk_subspace_roc_auc"]})
        bars = score_ax.barh(scores.index, scores.values, color=[COLORS["blue"], COLORS["green"], COLORS["gold"]])
        score_ax.set_xlim(0.48, 0.95)
        for bar in bars:
            score_ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}", va="center", fontsize=7)
        if idx == 0:
            rank_ax.set_title("Ranked spectrum")
            hist_ax.set_title("MP bulk window")
            score_ax.set_title("test ROC-AUC")
        if idx < len(diagnostics) - 1:
            rank_ax.set_xticklabels([])
            hist_ax.set_xticklabels([])
            score_ax.set_xticklabels([])
    axes[-1, 0].set_xlabel("component rank")
    axes[-1, 1].set_xlabel("eigenvalue")
    axes[-1, 2].set_xlabel("ROC-AUC")
    fig.suptitle("Marchenko-Pastur Bulk and Model-Specific Reconstruction Behavior", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.03, 0, 1, 0.985))
    _save(fig, "paper_model_mp_law_facets.png")


def filtering_experiment() -> None:
    results = pd.read_csv(_require(METRICS / "best_model_rmt_filtering.csv"))
    order = ["original_features", "mp_outlier_scores", "mp_outlier_reconstruction", "pca_plus_plus_scores"]
    labels = {
        "original_features": "original",
        "mp_outlier_scores": "MP scores",
        "mp_outlier_reconstruction": "MP reconstruction",
        "pca_plus_plus_scores": "PCA++ scores",
    }
    frame = results[results["representation"].isin(order)].copy()
    test = frame[frame["split"] == "test"].set_index("representation")
    validation = frame[frame["split"] == "validation"].set_index("representation")
    baseline = test.loc["original_features"]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    x = np.arange(len(order))
    axes[0].bar(x - 0.17, [validation.loc[item, "roc_auc"] for item in order], width=0.34, color=COLORS["blue"], alpha=0.6, label="validation")
    axes[0].bar(x + 0.17, [test.loc[item, "roc_auc"] for item in order], width=0.34, color=COLORS["red"], alpha=0.85, label="test")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([labels[item] for item in order], rotation=20, ha="right")
    axes[0].set_ylim(0.86, 0.935)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Best-model filtering experiment")
    axes[0].legend(frameon=False)

    metrics = ["roc_auc", "accuracy", "balanced_accuracy", "f1"]
    metric_labels = ["ROC-AUC", "accuracy", "balanced acc.", "F1"]
    y = np.arange(len(metrics))
    for representation, color in [
        ("mp_outlier_scores", COLORS["gold"]),
        ("mp_outlier_reconstruction", COLORS["green"]),
        ("pca_plus_plus_scores", COLORS["red"]),
    ]:
        deltas = [test.loc[representation, metric] - baseline[metric] for metric in metrics]
        axes[1].scatter(deltas, y, label=labels[representation], color=color)
        for value, yy in zip(deltas, y):
            axes[1].plot([0, value], [yy, yy], color=COLORS["light"])
    axes[1].axvline(0, color=COLORS["gray"], linestyle="--")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(metric_labels)
    axes[1].set_xlabel("test delta versus original")
    axes[1].set_title("Held-out effect")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, "paper_rmt_filtering_experiment.png")


def main() -> None:
    _setup()
    preprocessing_findings()
    dataset_spectrum()
    model_summary()
    model_mp_facets()
    filtering_experiment()


if __name__ == "__main__":
    main()
