"""Empirical regime-switching scenario generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.utils.random import make_rng


@dataclass(frozen=True)
class RegimeFit:
    """Fitted empirical regime model."""

    rolling_window: int
    n_regimes: int
    regime_names: tuple[str, ...]

    def to_dict(self) -> dict[str, int | str]:
        """Return serializable parameters."""
        row = asdict(self)
        row["regime_names"] = ",".join(self.regime_names)
        return row


def assign_vol_corr_regimes(
    data: pd.DataFrame,
    columns: list[str],
    rolling_window: int = 252,
) -> pd.Series:
    """Assign regimes from SP500 volatility and rolling correlation sign."""
    x_col, y_col = columns
    x = data[x_col]
    y = data[y_col]
    volatility = x.rolling(rolling_window, min_periods=max(20, rolling_window // 5)).std()
    corr = x.rolling(rolling_window, min_periods=max(20, rolling_window // 5)).corr(y)
    vol_threshold = volatility.quantile(0.8)

    vol_label = np.where(volatility > vol_threshold, "highvol", "lowvol")
    corr_label = np.where(corr >= 0, "poscorr", "negcorr")
    regime = pd.Series(
        [f"{v}_{c}" for v, c in zip(vol_label, corr_label)],
        index=data.index,
        name="regime",
    )
    return regime.ffill().bfill()


def fit_empirical_regime_model(
    data: pd.DataFrame,
    columns: list[str],
    rolling_window: int = 252,
) -> tuple[RegimeFit, pd.DataFrame, pd.DataFrame]:
    """Fit empirical regimes, transition probabilities, and regime samples."""
    work = data[columns].dropna().copy()
    work["regime"] = assign_vol_corr_regimes(work, columns, rolling_window)
    regimes = tuple(sorted(work["regime"].unique()))
    transitions = _transition_matrix(work["regime"], regimes)
    fit = RegimeFit(rolling_window, len(regimes), regimes)
    return fit, transitions, work


def simulate_empirical_regime(
    regime_data: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    columns: list[str],
    path_length: int,
    n_paths: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate paths by Markov-switching empirical row resampling."""
    rng = make_rng(seed)
    regimes = transition_matrix.index.to_list()
    start_probs = regime_data["regime"].value_counts(normalize=True).reindex(regimes).fillna(0.0)
    rows = []
    for path_id in range(n_paths):
        regime = rng.choice(regimes, p=start_probs.to_numpy())
        for t in range(path_length):
            bucket = regime_data[regime_data["regime"] == regime]
            if bucket.empty:
                bucket = regime_data
            sample = bucket.iloc[rng.integers(0, len(bucket))]
            row = {
                "model": "empirical_regime",
                "path_id": path_id,
                "t": t,
                "regime": regime,
            }
            for column in columns:
                row[column] = sample[column]
            rows.append(row)
            probs = transition_matrix.loc[regime].to_numpy()
            regime = rng.choice(regimes, p=probs)
    return pd.DataFrame(rows)


def _transition_matrix(regime: pd.Series, regimes: tuple[str, ...]) -> pd.DataFrame:
    """Estimate a smoothed Markov transition matrix."""
    counts = pd.DataFrame(1.0, index=regimes, columns=regimes)
    previous = regime.shift(1)
    for old, new in zip(previous.iloc[1:], regime.iloc[1:]):
        counts.loc[old, new] += 1.0
    return counts.div(counts.sum(axis=1), axis=0)
