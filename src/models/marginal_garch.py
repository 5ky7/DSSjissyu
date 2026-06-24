"""Lightweight univariate GARCH-family fitting and simulation.

This module intentionally avoids a hard dependency on the ``arch`` package.
It estimates Gaussian quasi-likelihood models with SciPy and uses Student-t
innovations during simulation to preserve fat tails.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from src.utils.random import make_rng


@dataclass(frozen=True)
class GarchFit:
    """Fitted marginal volatility model."""

    model: str
    mu: float
    omega: float
    alpha: float
    beta: float
    gamma: float
    loglikelihood: float
    aic: float
    bic: float
    nu: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a serializable representation."""
        return asdict(self)


def fit_garch_family(series: pd.Series, model: str = "garch") -> tuple[GarchFit, pd.DataFrame]:
    """Fit GARCH, GJR-GARCH, or EGARCH to a series."""
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    if len(values) < 100:
        raise ValueError("At least 100 observations are required")
    model = model.lower()
    if model not in {"garch", "gjr_garch", "egarch"}:
        raise ValueError(f"Unsupported model: {model}")

    mu0 = float(np.mean(values))
    var0 = float(np.var(values - mu0))
    start = _initial_params(model, mu0, var0)
    bounds = _bounds(model, values)
    result = optimize.minimize(
        _negative_loglikelihood,
        start,
        args=(values, model),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000},
    )
    params = result.x if result.success else start
    sigma2 = _conditional_variance(values, params, model)
    resid = values - params[0]
    standardized = resid / np.sqrt(sigma2)
    nu = _estimate_t_df(standardized)
    loglik = -_negative_loglikelihood(params, values, model)
    k = len(params) + 1
    fit = GarchFit(
        model=model,
        mu=float(params[0]),
        omega=float(params[1]),
        alpha=float(params[2]),
        beta=float(params[3]),
        gamma=float(params[4]) if len(params) > 4 else 0.0,
        loglikelihood=float(loglik),
        aic=float(2 * k - 2 * loglik),
        bic=float(np.log(len(values)) * k - 2 * loglik),
        nu=float(nu),
    )
    diagnostics = pd.DataFrame(
        {
            "x": values,
            "conditional_volatility": np.sqrt(sigma2),
            "standardized_residual": standardized,
        }
    )
    return fit, diagnostics


def simulate_garch(
    fit: GarchFit,
    path_length: int,
    n_paths: int,
    seed: int | None = None,
    initial_variance: float | None = None,
) -> pd.DataFrame:
    """Simulate univariate returns from a fitted model."""
    rng = make_rng(seed)
    variance0 = initial_variance or fit.omega / max(1e-8, 1.0 - fit.alpha - fit.beta)
    rows = []
    for path_id in range(n_paths):
        variance = max(variance0, 1e-12)
        previous_resid = 0.0
        for t in range(path_length):
            shock = rng.standard_t(fit.nu)
            shock *= np.sqrt((fit.nu - 2.0) / fit.nu) if fit.nu > 2 else 1.0
            value = fit.mu + np.sqrt(variance) * shock
            rows.append(
                {
                    "model": fit.model,
                    "path_id": path_id,
                    "t": t,
                    "value": value,
                    "conditional_volatility": np.sqrt(variance),
                }
            )
            variance = _next_variance(fit, variance, value - fit.mu, previous_resid)
            previous_resid = value - fit.mu
    return pd.DataFrame(rows)


def fit_marginals(
    df: pd.DataFrame,
    columns: list[str],
    model_names: tuple[str, ...] = ("garch", "gjr_garch", "egarch"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit several marginal models and return comparison/residual tables."""
    fits = []
    residuals = []
    for column in columns:
        best_for_residuals = None
        for model_name in model_names:
            fit, diagnostics = fit_garch_family(df[column], model_name)
            row = fit.to_dict()
            row["series"] = column
            fits.append(row)
            if best_for_residuals is None or fit.aic < best_for_residuals[0].aic:
                best_for_residuals = (fit, diagnostics)
        assert best_for_residuals is not None
        fit, diagnostics = best_for_residuals
        diagnostics = diagnostics.copy()
        diagnostics.insert(0, "series", column)
        diagnostics.insert(1, "selected_model", fit.model)
        residuals.append(diagnostics)
    return pd.DataFrame(fits), pd.concat(residuals, ignore_index=True)


def _initial_params(model: str, mu: float, variance: float) -> np.ndarray:
    """Initial parameter vector."""
    if model == "egarch":
        return np.array([mu, np.log(max(variance * 0.05, 1e-10)), 0.05, 0.9, -0.05])
    if model == "gjr_garch":
        return np.array([mu, max(variance * 0.05, 1e-10), 0.05, 0.9, 0.05])
    return np.array([mu, max(variance * 0.05, 1e-10), 0.05, 0.9])


def _bounds(model: str, values: np.ndarray) -> list[tuple[float, float]]:
    """Optimization bounds."""
    scale = max(float(np.std(values)), 1e-4)
    bounds = [(-10 * scale, 10 * scale), (1e-12, 10 * scale * scale), (1e-6, 0.5), (1e-6, 0.999)]
    if model == "egarch":
        bounds[1] = (-50.0, 10.0)
        bounds.append((-0.5, 0.5))
    elif model == "gjr_garch":
        bounds.append((0.0, 0.5))
    return bounds


def _negative_loglikelihood(params: np.ndarray, values: np.ndarray, model: str) -> float:
    """Gaussian quasi negative log-likelihood."""
    sigma2 = _conditional_variance(values, params, model)
    resid = values - params[0]
    if np.any(~np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e12
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + resid**2 / sigma2))


def _conditional_variance(values: np.ndarray, params: np.ndarray, model: str) -> np.ndarray:
    """Run variance recursion for a model."""
    mu = params[0]
    resid = values - mu
    n = len(values)
    variance = np.empty(n)
    unconditional = max(float(np.var(resid)), 1e-12)
    variance[0] = unconditional
    if model == "egarch":
        omega, alpha, beta, gamma = params[1:5]
        log_variance = np.log(unconditional)
        expected_abs_z = np.sqrt(2.0 / np.pi)
        for t in range(1, n):
            previous_variance = np.exp(np.clip(log_variance, -50.0, 50.0))
            z = resid[t - 1] / np.sqrt(max(previous_variance, 1e-12))
            log_variance = omega + beta * log_variance + alpha * (abs(z) - expected_abs_z) + gamma * z
            log_variance = float(np.clip(log_variance, -50.0, 50.0))
            variance[t] = max(np.exp(log_variance), 1e-12)
        return variance

    omega, alpha, beta = params[1:4]
    gamma = params[4] if model == "gjr_garch" else 0.0
    if alpha + beta + 0.5 * gamma >= 0.999:
        return np.full(n, np.nan)
    for t in range(1, n):
        asymmetry = gamma * resid[t - 1] ** 2 if resid[t - 1] < 0 else 0.0
        variance[t] = omega + alpha * resid[t - 1] ** 2 + asymmetry + beta * variance[t - 1]
        variance[t] = max(variance[t], 1e-12)
    return variance


def _next_variance(
    fit: GarchFit,
    current_variance: float,
    resid: float,
    previous_resid: float,
) -> float:
    """Advance one volatility step."""
    if fit.model == "egarch":
        expected_abs_z = np.sqrt(2.0 / np.pi)
        z = previous_resid / np.sqrt(max(current_variance, 1e-12))
        log_variance = (
            fit.omega
            + fit.beta * np.log(max(current_variance, 1e-12))
            + fit.alpha * (abs(z) - expected_abs_z)
            + fit.gamma * z
        )
        log_variance = float(np.clip(log_variance, -50.0, 50.0))
        return float(max(np.exp(log_variance), 1e-12))
    asymmetry = fit.gamma * resid**2 if resid < 0 else 0.0
    return float(max(fit.omega + fit.alpha * resid**2 + asymmetry + fit.beta * current_variance, 1e-12))


def _estimate_t_df(standardized: np.ndarray) -> float:
    """Estimate Student-t degrees of freedom from excess kurtosis."""
    kurtosis = pd.Series(standardized).kurtosis()
    if not np.isfinite(kurtosis) or kurtosis <= 0:
        return 30.0
    return float(np.clip(4.0 + 6.0 / kurtosis, 4.1, 60.0))
