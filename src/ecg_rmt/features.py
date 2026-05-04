from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


LEAD_NAMES = [
    "I",
    "II",
    "III",
    "AVR",
    "AVL",
    "AVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def _safe_skew(values: np.ndarray) -> float:
    centered = values - np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float(np.nanmean((centered / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    centered = values - np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float(np.nanmean((centered / std) ** 4) - 3.0)


def _lead_feature_names() -> list[str]:
    names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "q01",
        "q05",
        "q25",
        "q75",
        "q95",
        "q99",
        "iqr",
        "rms",
        "energy",
        "skew",
        "kurtosis",
        "zero_crossing_rate",
        "derivative_mean",
        "derivative_std",
        "derivative_rms",
        "fft_total_power",
        "fft_centroid_hz",
        "fft_entropy",
        "fft_dominant_hz",
        "fft_rolloff85_hz",
        "fft_rolloff95_hz",
    ]
    for band in ["baseline", "low", "qrs", "high", "very_high"]:
        names.extend([f"band_{band}_power", f"band_{band}_fraction"])
    names.append("band_low_high_ratio")
    return names


def _spectral_entropy(power: np.ndarray) -> float:
    total = float(np.nansum(power))
    if total <= 0 or not np.isfinite(total):
        return 0.0
    probs = power / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)) / np.log(len(power)))


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(np.nansum(power[mask]))


def summarize_lead(values: np.ndarray, sampling_rate: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: np.nan for name in _lead_feature_names()}

    derivative = np.diff(finite)
    freqs = np.fft.rfftfreq(finite.size, d=1.0 / sampling_rate)
    spectrum = np.fft.rfft(finite - finite.mean())
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    cumulative = np.cumsum(power)

    def rolloff(frac: float) -> float:
        if total_power <= 0:
            return 0.0
        idx = int(np.searchsorted(cumulative, frac * total_power, side="left"))
        idx = min(idx, len(freqs) - 1)
        return float(freqs[idx])

    bands = {
        "baseline": (0.0, 0.5),
        "low": (0.5, 5.0),
        "qrs": (5.0, 15.0),
        "high": (15.0, 40.0),
        "very_high": (40.0, sampling_rate / 2),
    }
    features = {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q05": float(np.quantile(finite, 0.05)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "iqr": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)),
        "rms": float(np.sqrt(np.mean(finite**2))),
        "energy": float(np.sum(finite**2)),
        "skew": _safe_skew(finite),
        "kurtosis": _safe_kurtosis(finite),
        "zero_crossing_rate": float(np.mean(np.diff(np.signbit(finite)) != 0)),
        "derivative_mean": float(np.mean(derivative)) if derivative.size else 0.0,
        "derivative_std": float(np.std(derivative)) if derivative.size else 0.0,
        "derivative_rms": (
            float(np.sqrt(np.mean(derivative**2))) if derivative.size else 0.0
        ),
        "fft_total_power": total_power,
        "fft_centroid_hz": (
            float(np.sum(freqs * power) / total_power) if total_power > 0 else 0.0
        ),
        "fft_entropy": _spectral_entropy(power),
        "fft_dominant_hz": float(freqs[int(np.argmax(power))]),
        "fft_rolloff85_hz": rolloff(0.85),
        "fft_rolloff95_hz": rolloff(0.95),
    }
    for name, (low, high) in bands.items():
        value = _band_power(freqs, power, low, high)
        features[f"band_{name}_power"] = value
        features[f"band_{name}_fraction"] = value / total_power if total_power > 0 else 0.0
    high_power = features["band_high_power"] + features["band_very_high_power"]
    features["band_low_high_ratio"] = features["band_low_power"] / max(high_power, 1e-12)
    return features


def extract_ecg_features(
    signal: np.ndarray,
    sampling_rate: int,
    lead_names: Iterable[str] = LEAD_NAMES,
) -> dict[str, float]:
    values = np.asarray(signal, dtype=float)
    if values.ndim != 2:
        raise ValueError("signal must have shape (samples, leads)")
    lead_names = list(lead_names)
    if values.shape[1] != len(lead_names):
        lead_names = [f"lead_{idx}" for idx in range(values.shape[1])]

    features: dict[str, float] = {
        "global_duration_seconds": values.shape[0] / sampling_rate,
        "global_n_samples": float(values.shape[0]),
        "global_n_leads": float(values.shape[1]),
        "global_mean_abs_amplitude": float(np.nanmean(np.abs(values))),
        "global_std_amplitude": float(np.nanstd(values)),
        "global_missing_fraction": float(np.isnan(values).mean()),
    }

    for idx, lead in enumerate(lead_names):
        for name, value in summarize_lead(values[:, idx], sampling_rate).items():
            features[f"lead_{lead}_{name}"] = value

    for left_idx, right_idx in combinations(range(values.shape[1]), 2):
        left = values[:, left_idx]
        right = values[:, right_idx]
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() < 3:
            corr = np.nan
        else:
            corr = float(np.corrcoef(left[mask], right[mask])[0, 1])
        features[f"corr_{lead_names[left_idx]}_{lead_names[right_idx]}"] = corr
    return features


def feature_frame(records: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)
