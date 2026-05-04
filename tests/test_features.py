from __future__ import annotations

import numpy as np

from ecg_rmt.features import LEAD_NAMES, extract_ecg_features, summarize_lead


def test_summarize_lead_contains_time_and_frequency_features() -> None:
    t = np.linspace(0, 1, 100, endpoint=False)
    signal = np.sin(2 * np.pi * 5 * t)
    features = summarize_lead(signal, sampling_rate=100)
    assert features["std"] > 0
    assert features["fft_total_power"] > 0
    assert abs(features["fft_dominant_hz"] - 5.0) < 1.1


def test_extract_ecg_features_has_per_lead_and_correlation_blocks() -> None:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=(100, len(LEAD_NAMES)))
    features = extract_ecg_features(signal, sampling_rate=100)
    assert "lead_I_mean" in features
    assert "lead_V6_fft_entropy" in features
    assert "corr_I_II" in features
    assert features["global_n_leads"] == 12
