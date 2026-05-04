from __future__ import annotations

import numpy as np

from ecg_rmt.rmt import (
    effective_rank,
    fit_covariance_decomposition,
    marchenko_pastur_edges,
    marchenko_pastur_pdf,
    pca_plus_plus_directions,
    reconstruct_from_components,
)


def test_marchenko_pastur_edges() -> None:
    lower, upper = marchenko_pastur_edges(0.25)
    assert np.isclose(lower, 0.25)
    assert np.isclose(upper, 2.25)


def test_marchenko_pastur_pdf_zero_outside_support() -> None:
    values = np.array([0.1, 1.0, 3.0])
    density = marchenko_pastur_pdf(values, ratio=0.25)
    assert density[0] == 0
    assert density[1] > 0
    assert density[2] == 0


def test_covariance_decomposition_orders_eigenvalues() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(80, 10))
    decomposition = fit_covariance_decomposition(matrix)
    assert np.all(np.diff(decomposition.eigenvalues) <= 1e-10)
    assert decomposition.components.shape == (10, 10)


def test_reconstruction_shape() -> None:
    rng = np.random.default_rng(9)
    matrix = rng.normal(size=(50, 8))
    decomposition = fit_covariance_decomposition(matrix)
    mask = np.zeros(8, dtype=bool)
    mask[:3] = True
    reconstructed = reconstruct_from_components(matrix, decomposition, mask)
    assert reconstructed.shape == matrix.shape


def test_effective_rank_bounds() -> None:
    rank = effective_rank(np.array([1.0, 1.0, 1.0, 1.0]))
    assert np.isclose(rank, 4.0)


def test_pca_plus_plus_direction_shape() -> None:
    rng = np.random.default_rng(11)
    signal = rng.normal(size=(60, 2))
    loading = rng.normal(size=(2, 7))
    left = signal @ loading + 0.05 * rng.normal(size=(60, 7))
    right = signal @ loading + 0.05 * rng.normal(size=(60, 7))
    directions = pca_plus_plus_directions(left, right, n_components=3)
    assert directions.shape == (3, 7)
