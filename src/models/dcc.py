"""Two-dimensional DCC correlation model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import optimize

from src.utils.random import make_rng


@dataclass(frozen=True)
class DccFit:
    """Fitted DCC parameters."""

    a: float
    b: float
    qbar_00: float
    qbar_01: float
    qbar_11: float
    loglikelihood: float

    def to_dict(self) -> dict[str, float]:
        """Return serializable parameters."""
        return asdict(self)

    @property
    def qbar(self) -> np.ndarray:
        """Return Q-bar matrix."""
        return np.array([[self.qbar_00, self.qbar_01], [self.qbar_01, self.qbar_11]])


def fit_dcc(standardized_residuals: pd.DataFrame) -> tuple[DccFit, pd.DataFrame]:
    """Fit a two-dimensional DCC model to standardized residuals."""
    z = standardized_residuals.dropna().to_numpy(dtype=float)
    if z.shape[1] != 2:
        raise ValueError("DCC implementation expects exactly two columns")
    qbar = np.cov(z.T)

    def objective(theta: np.ndarray) -> float:
        a, b = theta
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        _, loglik = dcc_correlations(z, a, b, qbar)
        return -loglik

    result = optimize.minimize(
        objective,
        x0=np.array([0.03, 0.94]),
        method="L-BFGS-B",
        bounds=[(1e-6, 0.5), (1e-6, 0.999)],
    )
    a, b = result.x if result.success else np.array([0.03, 0.94])
    correlations, loglik = dcc_correlations(z, float(a), float(b), qbar)
    fit = DccFit(
        a=float(a),
        b=float(b),
        qbar_00=float(qbar[0, 0]),
        qbar_01=float(qbar[0, 1]),
        qbar_11=float(qbar[1, 1]),
        loglikelihood=float(loglik),
    )
    corr_df = pd.DataFrame({"t": np.arange(len(correlations)), "conditional_corr": correlations})
    return fit, corr_df


def dcc_correlations(
    z: np.ndarray,
    a: float,
    b: float,
    qbar: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Run DCC recursion and return correlations plus log-likelihood."""
    q = qbar.copy()
    correlations = np.empty(len(z))
    loglik = 0.0
    for t in range(len(z)):
        if t > 0:
            outer = np.outer(z[t - 1], z[t - 1])
            q = (1.0 - a - b) * qbar + a * outer + b * q
        r = _correlation_from_q(q)
        correlations[t] = r[0, 1]
        loglik += _normal_corr_loglik(z[t], r)
    return correlations, float(loglik)


def simulate_dcc(
    fit: DccFit,
    means: np.ndarray,
    volatilities: np.ndarray,
    path_length: int,
    n_paths: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate bivariate paths with DCC innovations and sampled volatilities."""
    rng = make_rng(seed)
    qbar = fit.qbar
    rows = []
    for path_id in range(n_paths):
        q = qbar.copy()
        previous_z = rng.multivariate_normal(np.zeros(2), _correlation_from_q(q))
        for t in range(path_length):
            if t > 0:
                q = (1.0 - fit.a - fit.b) * qbar + fit.a * np.outer(previous_z, previous_z) + fit.b * q
            r = _correlation_from_q(q)
            z = rng.multivariate_normal(np.zeros(2), r)
            vol = volatilities[rng.integers(0, len(volatilities))]
            x = means + vol * z
            rows.append(
                {
                    "model": "dcc",
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


def _correlation_from_q(q: np.ndarray) -> np.ndarray:
    """Convert a covariance-like matrix into a correlation matrix."""
    q = np.asarray(q, dtype=float)
    q = (q + q.T) / 2.0
    diag = np.sqrt(np.maximum(np.diag(q), 1e-12))
    r = q / np.outer(diag, diag)
    r = np.clip(r, -0.999, 0.999)
    np.fill_diagonal(r, 1.0)
    return r


def _normal_corr_loglik(z_t: np.ndarray, r: np.ndarray) -> float:
    """Gaussian correlation log-likelihood contribution."""
    sign, logdet = np.linalg.slogdet(r)
    if sign <= 0:
        return -1e12
    inv = np.linalg.inv(r)
    return float(-0.5 * (logdet + z_t @ (inv - np.eye(2)) @ z_t))
