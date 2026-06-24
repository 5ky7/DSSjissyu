"""Static and simple dynamic copula scenario generators."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.random import make_rng


@dataclass(frozen=True)
class CopulaFit:
    """Fitted two-dimensional copula parameters."""

    model: str
    correlation: float
    df: float | None = None
    rolling_window: int | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        """Return serializable parameters."""
        return asdict(self)


def fit_gaussian_copula(data: pd.DataFrame, columns: list[str]) -> CopulaFit:
    """Fit a Gaussian copula using empirical PIT values."""
    scores = _normal_scores(data, columns)
    return CopulaFit("gaussian_copula", float(scores.corr().iloc[0, 1]))


def fit_t_copula(data: pd.DataFrame, columns: list[str]) -> CopulaFit:
    """Fit a t-copula with method-of-moments tail degrees of freedom."""
    scores = _normal_scores(data, columns)
    corr = float(scores.corr().iloc[0, 1])
    df = _estimate_joint_tail_df(scores)
    return CopulaFit("t_copula", corr, df)


def fit_dynamic_gaussian_copula(
    data: pd.DataFrame,
    columns: list[str],
    rolling_window: int = 252,
) -> tuple[CopulaFit, pd.Series]:
    """Fit a dynamic Gaussian copula from rolling copula-score correlations."""
    scores = _normal_scores(data, columns)
    rolling_corr = scores[columns[0]].rolling(rolling_window).corr(scores[columns[1]])
    rolling_corr = rolling_corr.dropna().clip(-0.95, 0.95)
    if rolling_corr.empty:
        rolling_corr = pd.Series([scores.corr().iloc[0, 1]])
    fit = CopulaFit(
        "dynamic_gaussian_copula",
        float(scores.corr().iloc[0, 1]),
        rolling_window=rolling_window,
    )
    return fit, rolling_corr.reset_index(drop=True)


def simulate_copula(
    data: pd.DataFrame,
    columns: list[str],
    fit: CopulaFit,
    path_length: int,
    n_paths: int,
    seed: int | None = None,
    dynamic_correlations: pd.Series | None = None,
) -> pd.DataFrame:
    """Simulate paths from a fitted copula and empirical marginals."""
    rng = make_rng(seed)
    historical = data[columns].dropna().reset_index(drop=True)
    paths = []
    for path_id in range(n_paths):
        uniforms = _simulate_uniforms(
            fit,
            path_length,
            rng,
            dynamic_correlations=dynamic_correlations,
        )
        sampled = pd.DataFrame(
            {
                column: _empirical_ppf(historical[column], uniforms[:, i])
                for i, column in enumerate(columns)
            }
        )
        sampled.insert(0, "t", np.arange(path_length))
        sampled.insert(0, "path_id", path_id)
        sampled.insert(0, "model", fit.model)
        paths.append(sampled)
    return pd.concat(paths, ignore_index=True)


def _normal_scores(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert empirical PIT values to normal scores."""
    uniforms = pd.DataFrame(index=data.index)
    for column in columns:
        ranks = data[column].rank(method="average")
        uniforms[column] = ranks / (ranks.notna().sum() + 1.0)
    uniforms = uniforms[columns].dropna().clip(1e-6, 1.0 - 1e-6)
    return pd.DataFrame(stats.norm.ppf(uniforms), columns=columns, index=uniforms.index)


def _estimate_joint_tail_df(scores: pd.DataFrame) -> float:
    """Estimate a conservative t degrees-of-freedom from score kurtosis."""
    kurtosis = float(scores.stack().kurtosis())
    if not np.isfinite(kurtosis) or kurtosis <= 0:
        return 10.0
    return float(np.clip(4.0 + 6.0 / kurtosis, 4.5, 30.0))


def _simulate_uniforms(
    fit: CopulaFit,
    path_length: int,
    rng: np.random.Generator,
    dynamic_correlations: pd.Series | None = None,
) -> np.ndarray:
    """Simulate copula uniforms."""
    if fit.model == "dynamic_gaussian_copula":
        if dynamic_correlations is None or dynamic_correlations.empty:
            raise ValueError("dynamic_correlations are required")
        uniforms = np.empty((path_length, 2))
        start = rng.integers(0, len(dynamic_correlations))
        for t in range(path_length):
            corr = float(dynamic_correlations.iloc[(start + t) % len(dynamic_correlations)])
            z = rng.multivariate_normal(np.zeros(2), _corr_matrix(corr))
            uniforms[t] = stats.norm.cdf(z)
        return uniforms

    if fit.model == "t_copula":
        corr = _corr_matrix(fit.correlation)
        normal = rng.multivariate_normal(np.zeros(2), corr, size=path_length)
        chi = rng.chisquare(fit.df or 10.0, size=path_length)
        t_values = normal / np.sqrt(chi[:, None] / (fit.df or 10.0))
        return stats.t.cdf(t_values, df=fit.df or 10.0)

    normal = rng.multivariate_normal(
        np.zeros(2),
        _corr_matrix(fit.correlation),
        size=path_length,
    )
    return stats.norm.cdf(normal)


def _corr_matrix(correlation: float) -> np.ndarray:
    """Create a numerically stable 2x2 correlation matrix."""
    corr = float(np.clip(correlation, -0.95, 0.95))
    return np.array([[1.0, corr], [corr, 1.0]])


def _empirical_ppf(series: pd.Series, uniforms: np.ndarray) -> np.ndarray:
    """Map uniforms to empirical quantiles."""
    values = np.sort(series.dropna().to_numpy())
    probabilities = np.linspace(0.0, 1.0, len(values))
    return np.interp(uniforms, probabilities, values)
