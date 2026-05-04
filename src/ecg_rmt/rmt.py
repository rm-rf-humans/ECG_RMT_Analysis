from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def marchenko_pastur_edges(ratio: float, sigma2: float = 1.0) -> tuple[float, float]:
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    root = np.sqrt(ratio)
    return sigma2 * (1.0 - root) ** 2, sigma2 * (1.0 + root) ** 2


def marchenko_pastur_pdf(
    eigenvalues: np.ndarray,
    ratio: float,
    sigma2: float = 1.0,
) -> np.ndarray:
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    lower, upper = marchenko_pastur_edges(ratio, sigma2=sigma2)
    density = np.zeros_like(eigenvalues, dtype=float)
    mask = (eigenvalues >= lower) & (eigenvalues <= upper) & (eigenvalues > 0)
    numerator = np.sqrt((upper - eigenvalues[mask]) * (eigenvalues[mask] - lower))
    density[mask] = numerator / (2.0 * np.pi * ratio * sigma2 * eigenvalues[mask])
    return density


@dataclass(frozen=True)
class CovarianceDecomposition:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    mp_lower: float
    mp_upper: float
    aspect_ratio: float

    @property
    def outlier_mask(self) -> np.ndarray:
        return self.eigenvalues > self.mp_upper

    @property
    def bulk_mask(self) -> np.ndarray:
        return ~self.outlier_mask

    @property
    def n_outliers(self) -> int:
        return int(self.outlier_mask.sum())


def fit_covariance_decomposition(
    matrix: np.ndarray,
    standardize: bool = True,
    sigma2: float = 1.0,
) -> CovarianceDecomposition:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    mean = np.nanmean(values, axis=0)
    centered = values - mean
    centered = np.nan_to_num(centered, nan=0.0)
    if standardize:
        scale = centered.std(axis=0, ddof=1)
        scale[scale == 0] = 1.0
        centered = centered / scale
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = components[:, order].T
    ratio = values.shape[1] / values.shape[0]
    lower, upper = marchenko_pastur_edges(ratio, sigma2=sigma2)
    return CovarianceDecomposition(mean, components, eigenvalues, lower, upper, ratio)


def covariance_spectrum_frame(decomposition: CovarianceDecomposition) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component": np.arange(1, decomposition.eigenvalues.size + 1),
            "eigenvalue": decomposition.eigenvalues,
            "mp_lower": decomposition.mp_lower,
            "mp_upper": decomposition.mp_upper,
            "aspect_ratio_p_over_n": decomposition.aspect_ratio,
            "is_outlier": decomposition.outlier_mask,
        }
    )


def effective_rank(eigenvalues: np.ndarray) -> float:
    values = np.asarray(eigenvalues, dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    probs = values / values.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return float(np.exp(entropy))


def reconstruct_from_components(
    matrix: np.ndarray,
    decomposition: CovarianceDecomposition,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    centered = np.nan_to_num(values - decomposition.mean, nan=0.0)
    kept = decomposition.components[np.asarray(mask, dtype=bool)]
    if kept.size == 0:
        return np.tile(decomposition.mean, (values.shape[0], 1))
    scores = centered @ kept.T
    return decomposition.mean + scores @ kept


def project_scores(
    matrix: np.ndarray,
    decomposition: CovarianceDecomposition,
    mask: np.ndarray,
) -> np.ndarray:
    centered = np.nan_to_num(np.asarray(matrix, dtype=float) - decomposition.mean, nan=0.0)
    return centered @ decomposition.components[np.asarray(mask, dtype=bool)].T


def pca_plus_plus_directions(
    left: np.ndarray,
    right: np.ndarray,
    n_components: int,
    ridge: float = 1e-6,
) -> np.ndarray:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired views must have the same shape")
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    denominator = max(left.shape[0] - 1, 1)
    covariance = (left.T @ left + right.T @ right) / (2.0 * denominator)
    positive = (left.T @ right + right.T @ left) / (2.0 * denominator)
    covariance = covariance + ridge * np.eye(covariance.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, ridge)
    whitening = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    whitened = whitening.T @ positive @ whitening
    score_values, score_vectors = np.linalg.eigh(whitened)
    order = np.argsort(score_values)[::-1]
    directions = whitening @ score_vectors[:, order[:n_components]]
    return directions.T
