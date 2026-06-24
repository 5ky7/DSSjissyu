"""Stylized-fact metrics for financial time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DiagnosticConfig:
    """Configuration for diagnostics."""

    acf_lags: int = 20
    leverage_lags: int = 20
    rolling_window: int = 252
    var_levels: tuple[float, ...] = (0.01, 0.05)
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)


def acf(series: pd.Series, max_lag: int) -> pd.Series:
    """Calculate autocorrelation from lag 1 to max_lag."""
    clean = pd.Series(series).dropna()
    values = {}
    for lag in range(1, max_lag + 1):
        values[f"acf_{lag}"] = clean.autocorr(lag=lag)
    return pd.Series(values)


def hill_tail_index(series: pd.Series, tail_fraction: float = 0.05) -> float:
    """Estimate a simple Hill tail index on absolute returns."""
    values = np.sort(np.abs(pd.Series(series).dropna().to_numpy()))
    if len(values) < 20:
        return np.nan
    k = max(5, int(len(values) * tail_fraction))
    tail = values[-k:]
    threshold = values[-k - 1]
    if threshold <= 0 or np.any(tail <= 0):
        return np.nan
    hill = np.mean(np.log(tail) - np.log(threshold))
    return np.nan if hill <= 0 else 1.0 / hill


def expected_shortfall(series: pd.Series, alpha: float) -> float:
    """Calculate lower-tail expected shortfall."""
    clean = pd.Series(series).dropna()
    var = clean.quantile(alpha)
    tail = clean[clean <= var]
    return float(tail.mean()) if not tail.empty else np.nan


def marginal_metrics(
    df: pd.DataFrame,
    columns: list[str],
    config: DiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Calculate marginal distribution metrics for each column."""
    config = config or DiagnosticConfig()
    rows = []
    for column in columns:
        series = pd.Series(df[column]).dropna()
        row = {
            "series": column,
            "mean": series.mean(),
            "std": series.std(ddof=1),
            "skewness": series.skew(),
            "excess_kurtosis": series.kurtosis(),
            "min": series.min(),
            "max": series.max(),
            "hill_tail_index_abs": hill_tail_index(series),
            "jarque_bera_stat": stats.jarque_bera(series).statistic,
            "jarque_bera_pvalue": stats.jarque_bera(series).pvalue,
        }
        for q in config.quantiles:
            row[f"q_{q:g}"] = series.quantile(q)
        for alpha in config.var_levels:
            row[f"var_{alpha:g}"] = series.quantile(alpha)
            row[f"es_{alpha:g}"] = expected_shortfall(series, alpha)
        rows.append(row)
    return pd.DataFrame(rows)


def acf_metrics(
    df: pd.DataFrame,
    columns: list[str],
    config: DiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Calculate ACF metrics for raw, absolute, and squared series."""
    config = config or DiagnosticConfig()
    rows = []
    for column in columns:
        series = pd.Series(df[column]).dropna()
        for transform_name, transformed in (
            ("raw", series),
            ("abs", series.abs()),
            ("squared", series**2),
        ):
            row = {"series": column, "transform": transform_name}
            row.update(acf(transformed, config.acf_lags).to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def leverage_correlation(
    series: pd.Series,
    max_lag: int,
) -> pd.Series:
    """Calculate Corr(x_t, x_{t+k}^2) for k=1..max_lag."""
    clean = pd.Series(series).dropna()
    values = {}
    for lag in range(1, max_lag + 1):
        values[f"leverage_{lag}"] = clean.corr((clean.shift(-lag) ** 2))
    return pd.Series(values)


def leverage_metrics(
    df: pd.DataFrame,
    columns: list[str],
    config: DiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Calculate leverage-effect correlation curves."""
    config = config or DiagnosticConfig()
    rows = []
    for column in columns:
        row = {"series": column}
        row.update(leverage_correlation(df[column], config.leverage_lags).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def dependence_metrics(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    config: DiagnosticConfig | None = None,
) -> pd.DataFrame:
    """Calculate two-series dependence metrics."""
    config = config or DiagnosticConfig()
    data = df[[x_col, y_col]].dropna()
    x = data[x_col]
    y = data[y_col]
    rolling_corr = x.rolling(config.rolling_window).corr(y).dropna()
    x_vol = x.rolling(config.rolling_window).std()
    high_vol_mask = x_vol > x_vol.quantile(0.8)
    down_mask = x < x.quantile(0.1)
    up_mask = x > x.quantile(0.9)

    row = {
        "x": x_col,
        "y": y_col,
        "correlation": x.corr(y),
        "rolling_corr_mean": rolling_corr.mean(),
        "rolling_corr_std": rolling_corr.std(ddof=1),
        "rolling_corr_q05": rolling_corr.quantile(0.05),
        "rolling_corr_q50": rolling_corr.quantile(0.5),
        "rolling_corr_q95": rolling_corr.quantile(0.95),
        "high_vol_correlation": x[high_vol_mask].corr(y[high_vol_mask]),
        "downside_correlation": x[down_mask].corr(y[down_mask]),
        "upside_correlation": x[up_mask].corr(y[up_mask]),
        "lower_tail_dependence_0.05": _tail_dependence(x, y, 0.05, lower=True),
        "upper_tail_dependence_0.95": _tail_dependence(x, y, 0.95, lower=False),
    }
    return pd.DataFrame([row])


def _tail_dependence(x: pd.Series, y: pd.Series, q: float, lower: bool) -> float:
    """Calculate empirical conditional tail co-occurrence."""
    if lower:
        x_tail = x <= x.quantile(q)
        y_tail = y <= y.quantile(q)
    else:
        x_tail = x >= x.quantile(q)
        y_tail = y >= y.quantile(q)
    denominator = x_tail.mean()
    if denominator == 0:
        return np.nan
    return float((x_tail & y_tail).mean() / denominator)


def stylized_fact_tables(
    df: pd.DataFrame,
    columns: list[str],
    dependence_pairs: list[tuple[str, str]] | None = None,
    config: DiagnosticConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all standard stylized-fact tables."""
    config = config or DiagnosticConfig()
    tables = {
        "marginal": marginal_metrics(df, columns, config),
        "acf": acf_metrics(df, columns, config),
        "leverage": leverage_metrics(df, columns, config),
    }
    if dependence_pairs:
        tables["dependence"] = pd.concat(
            [dependence_metrics(df, x, y, config) for x, y in dependence_pairs],
            ignore_index=True,
        )
    return tables
