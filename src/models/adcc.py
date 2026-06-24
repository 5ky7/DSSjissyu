"""Two-dimensional asymmetric DCC model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from src.models.dcc import _correlation_from_q, _normal_corr_loglik
from src.utils.random import make_rng


@dataclass(frozen=True)
class AdccFit:
    """Fitted ADCC parameters."""

    a: float
    b: float
    g: float
    qbar_00: float
    qbar_01: float
    qbar_11: float
    nbar_00: float
    nbar_01: float
    nbar_11: float
    loglikelihood: float

    def to_dict(self) -> dict[str, float]:
        """Return serializable parameters."""
        return asdict(self)

    @property
    def qbar(self) -> np.ndarray:
        """Return Q-bar matrix."""
        return np.array([[self.qbar_00, self.qbar_01], [self.qbar_01, self.qbar_11]])

    @property
    def nbar(self) -> np.ndarray:
        """Return N-bar matrix."""
        return np.array([[self.nbar_00, self.nbar_01], [self.nbar_01, self.nbar_11]])


def fit_adcc(standardized_residuals: pd.DataFrame) -> tuple[AdccFit, pd.DataFrame]:
    """Fit a two-dimensional ADCC model to standardized residuals."""
    z = standardized_residuals.dropna().to_numpy(dtype=float)
    if z.shape[1] != 2:
        raise ValueError("ADCC implementation expects exactly two columns")
    qbar = np.cov(z.T)
    negative = np.minimum(z, 0.0)
    nbar = np.cov(negative.T)

    def objective(theta: np.ndarray) -> float:
        a, b, g = theta
        if a < 0 or b < 0 or g < 0 or a + b + g >= 0.999:
            return 1e12
        _, loglik = adcc_correlations(z, a, b, g, qbar, nbar)
        return -loglik

    result = optimize.minimize(
        objective,
        x0=np.array([0.02, 0.92, 0.03]),
        method="L-BFGS-B",
        bounds=[(1e-6, 0.4), (1e-6, 0.999), (1e-6, 0.4)],
    )
    a, b, g = result.x if result.success else np.array([0.02, 0.92, 0.03])
    correlations, loglik = adcc_correlations(z, float(a), float(b), float(g), qbar, nbar)
    fit = AdccFit(
        a=float(a),
        b=float(b),
        g=float(g),
        qbar_00=float(qbar[0, 0]),
        qbar_01=float(qbar[0, 1]),
        qbar_11=float(qbar[1, 1]),
        nbar_00=float(nbar[0, 0]),
        nbar_01=float(nbar[0, 1]),
        nbar_11=float(nbar[1, 1]),
        loglikelihood=float(loglik),
    )
    corr_df = pd.DataFrame({"t": np.arange(len(correlations)), "conditional_corr": correlations})
    return fit, corr_df


def adcc_correlations(
    z: np.ndarray,
    a: float,
    b: float,
    g: float,
    qbar: np.ndarray,
    nbar: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run ADCC recursion and return correlations plus log-likelihood."""
    q = qbar.copy()
    correlations = np.empty(len(z))
    loglik = 0.0
    for t in range(len(z)):
        if t > 0:
            outer = np.outer(z[t - 1], z[t - 1])
            negative = np.minimum(z[t - 1], 0.0)
            neg_outer = np.outer(negative, negative)
            q = (1.0 - a - b - g) * qbar - g * nbar + a * outer + b * q + g * neg_outer
        r = _correlation_from_q(q)
        correlations[t] = r[0, 1]
        loglik += _normal_corr_loglik(z[t], r)
    return correlations, float(loglik)


def simulate_adcc(
    fit: AdccFit,
    means: np.ndarray,
    volatilities: np.ndarray,
    path_length: int,
    n_paths: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate bivariate paths with ADCC innovations and sampled volatilities."""
    rng = make_rng(seed)
    rows = []
    for path_id in range(n_paths):
        q = fit.qbar.copy()
        previous_z = rng.multivariate_normal(np.zeros(2), _correlation_from_q(q))
        for t in range(path_length):
            if t > 0:
                negative = np.minimum(previous_z, 0.0)
                q = (
                    (1.0 - fit.a - fit.b - fit.g) * fit.qbar
                    - fit.g * fit.nbar
                    + fit.a * np.outer(previous_z, previous_z)
                    + fit.b * q
                    + fit.g * np.outer(negative, negative)
                )
            r = _correlation_from_q(q)
            z = rng.multivariate_normal(np.zeros(2), r)
            vol = volatilities[rng.integers(0, len(volatilities))]
            x = means + vol * z
            rows.append(
                {
                    "model": "adcc",
                    "path_id": path_id,
                    "t": t,
                    "x0": x[0],
                    "x1": x[1],
                    "conditional_vol_x0": vol[0],
                    "conditional_vol_x1": vol[1],
                    "conditional_corr": r[0, 1],
                }
            )
            previous_z = z
    return pd.DataFrame(rows)
