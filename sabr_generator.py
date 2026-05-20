"""SABR-style simulator for two asset return series.

This module fits a practical SABR-style stochastic volatility model from two
return series and generates synthetic returns and prices.

The fitted dynamics are:

    dS_i = mu_i S_i dt + alpha_i S_i**beta_i dW_i
    d alpha_i = nu_i alpha_i dZ_i

The input data are daily returns, so calibration uses empirical moments and a
rolling-volatility proxy rather than option-implied volatility surfaces.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TRADING_DAYS = 252
EPSILON = 1e-12


@dataclasses.dataclass(frozen=True)
class AssetSabrParams:
    """SABR-style parameters for one asset."""

    name: str
    mu: float
    alpha: float
    beta: float
    nu: float
    rho: float
    s0: float = 100.0


@dataclasses.dataclass(frozen=True)
class TwoAssetSabrParams:
    """Parameters for a two-asset SABR-style simulator."""

    assets: tuple[AssetSabrParams, AssetSabrParams]
    brownian_corr: np.ndarray
    dt: float = 1.0 / TRADING_DAYS

    def to_json_dict(self) -> dict:
        """Converts the parameter object into a JSON-serializable dict."""
        return {
            "dt": self.dt,
            "assets": [dataclasses.asdict(asset) for asset in self.assets],
            "brownian_corr": self.brownian_corr.tolist(),
        }


def read_return_csv(
    path: str | Path,
    date_column: str | None,
    asset_columns: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Reads two return columns from a CSV file.

    Args:
        path: CSV file path.
        date_column: Name of the date column. If None, the first column is used.
        asset_columns: Names of the two return columns.

    Returns:
        A pair of date strings and an array with shape (n_observations, 2).

    Raises:
        ValueError: If required columns are missing or no valid rows exist.
    """
    path = Path(path)
    dates: list[str] = []
    rows: list[list[float]] = []

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header is missing: {path}")

        fieldnames = reader.fieldnames
        resolved_date_column = date_column or fieldnames[0]
        missing = [
            column
            for column in [resolved_date_column, *asset_columns]
            if column not in fieldnames
        ]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        for row in reader:
            try:
                values = [float(row[column]) for column in asset_columns]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in values):
                continue
            dates.append(row[resolved_date_column])
            rows.append(values)

    if not rows:
        raise ValueError(f"No valid return rows were found in {path}")
    return dates, np.asarray(rows, dtype=float)


def fit_two_asset_sabr(
    returns: np.ndarray,
    asset_names: Sequence[str],
    beta: float | Sequence[float] = 0.5,
    s0: float | Sequence[float] = 100.0,
    dt: float = 1.0 / TRADING_DAYS,
    vol_window: int = 21,
) -> TwoAssetSabrParams:
    """Fits two-asset SABR-style parameters from return observations.

    Args:
        returns: Return matrix with shape (n_observations, 2).
        asset_names: Names for the two assets.
        beta: SABR beta parameter. A scalar applies to both assets.
        s0: Initial generated price. A scalar applies to both assets.
        dt: Simulation time step in years.
        vol_window: Rolling window used for realized volatility proxy.

    Returns:
        Fitted two-asset parameters.
    """
    returns = _validate_returns(returns)
    betas = _as_pair(beta, "beta")
    initial_prices = _as_pair(s0, "s0")
    names = tuple(asset_names)
    if len(names) != 2:
        raise ValueError("asset_names must contain exactly two names")

    rolling_vol = _rolling_std(returns, window=vol_window)
    aligned_returns = returns[vol_window - 1 :]
    if len(aligned_returns) < 3:
        raise ValueError("Not enough observations for the requested vol_window")

    asset_params: list[AssetSabrParams] = []
    factor_columns: list[np.ndarray] = []

    for index, name in enumerate(names):
        asset_returns = aligned_returns[:, index]
        vol = np.maximum(rolling_vol[:, index], EPSILON)
        mu = float(np.mean(asset_returns) / dt)
        alpha = float(np.mean(vol) / math.sqrt(dt))
        vol_log_diff = np.diff(np.log(vol))
        nu = float(np.std(vol_log_diff, ddof=1) / math.sqrt(dt))

        standardized_return_shock = (asset_returns[1:] - np.mean(asset_returns)) / vol[1:]
        vol_shock = _safe_standardize(vol_log_diff)
        rho = _safe_corr(standardized_return_shock, vol_shock)

        asset_params.append(
            AssetSabrParams(
                name=name,
                mu=mu,
                alpha=max(alpha, EPSILON),
                beta=betas[index],
                nu=max(nu, EPSILON),
                rho=rho,
                s0=initial_prices[index],
            )
        )
        factor_columns.extend(
            [
                _safe_standardize(standardized_return_shock),
                _safe_standardize(vol_shock),
            ]
        )

    brownian_corr = _nearest_correlation(np.corrcoef(np.vstack(factor_columns)))
    return TwoAssetSabrParams(
        assets=(asset_params[0], asset_params[1]),
        brownian_corr=brownian_corr,
        dt=dt,
    )


def fit_two_asset_sabr_by_periods(
    dates: Sequence[str],
    returns: np.ndarray,
    periods: Sequence[tuple[str, str]],
    asset_names: Sequence[str],
    beta: float | Sequence[float] = 0.5,
    s0: float | Sequence[float] = 100.0,
    dt: float = 1.0 / TRADING_DAYS,
    vol_window: int = 21,
) -> dict[str, TwoAssetSabrParams]:
    """Fits SABR-style parameters for multiple date periods.

    Args:
        dates: Date strings aligned with returns. ISO-like YYYY-MM-DD strings
            are expected when using lexical date filtering.
        returns: Return matrix with shape (n_observations, 2).
        periods: Inclusive (start_date, end_date) pairs.
        asset_names: Names for the two assets.
        beta: SABR beta parameter. A scalar applies to both assets.
        s0: Initial generated price. A scalar applies to both assets.
        dt: Simulation time step in years.
        vol_window: Rolling window used for realized volatility proxy.

    Returns:
        Mapping from "start_date/end_date" to fitted parameters.
    """
    fitted: dict[str, TwoAssetSabrParams] = {}
    for start_date, end_date in periods:
        _, period_returns = filter_by_date(dates, returns, start_date, end_date)
        key = f"{start_date}/{end_date}"
        fitted[key] = fit_two_asset_sabr(
            period_returns,
            asset_names=asset_names,
            beta=beta,
            s0=s0,
            dt=dt,
            vol_window=vol_window,
        )
    return fitted


def simulate_two_asset_sabr(
    params: TwoAssetSabrParams,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Simulates returns and prices from fitted two-asset SABR-style params.

    Args:
        params: Fitted model parameters.
        n_steps: Number of time steps to generate.
        n_paths: Number of independent paths.
        seed: Optional random seed.

    Returns:
        Dict containing arrays:
            prices: shape (n_paths, n_steps + 1, 2)
            returns: shape (n_paths, n_steps, 2)
            alpha: shape (n_paths, n_steps + 1, 2)
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")

    rng = np.random.default_rng(seed)
    cholesky = np.linalg.cholesky(_nearest_correlation(params.brownian_corr))
    shocks = rng.standard_normal((n_paths, n_steps, 4)) @ cholesky.T

    prices = np.empty((n_paths, n_steps + 1, 2), dtype=float)
    returns = np.empty((n_paths, n_steps, 2), dtype=float)
    alpha = np.empty((n_paths, n_steps + 1, 2), dtype=float)

    for asset_index, asset in enumerate(params.assets):
        prices[:, 0, asset_index] = asset.s0
        alpha[:, 0, asset_index] = asset.alpha

    sqrt_dt = math.sqrt(params.dt)
    for step in range(n_steps):
        for asset_index, asset in enumerate(params.assets):
            price = np.maximum(prices[:, step, asset_index], EPSILON)
            current_alpha = np.maximum(alpha[:, step, asset_index], EPSILON)
            asset_shock = shocks[:, step, asset_index * 2]
            vol_shock = shocks[:, step, asset_index * 2 + 1]

            diffusion_scale = current_alpha * np.power(price, asset.beta - 1.0)
            period_return = asset.mu * params.dt + diffusion_scale * sqrt_dt * asset_shock
            next_price = np.maximum(price * (1.0 + period_return), EPSILON)
            next_alpha = current_alpha * np.exp(
                -0.5 * asset.nu**2 * params.dt + asset.nu * sqrt_dt * vol_shock
            )

            returns[:, step, asset_index] = period_return
            prices[:, step + 1, asset_index] = next_price
            alpha[:, step + 1, asset_index] = next_alpha
            # to do
            # これでなぜ相関を持たせられるのか．

    return {"prices": prices, "returns": returns, "alpha": alpha}


def filter_by_date(
    dates: Sequence[str],
    returns: np.ndarray,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Filters aligned dates and returns by an inclusive date range."""
    if len(dates) != len(returns):
        raise ValueError("dates and returns must have the same length")

    selected_dates: list[str] = []
    selected_returns: list[np.ndarray] = []
    for date, row in zip(dates, returns, strict=True):
        if start_date is not None and date < start_date:
            continue
        if end_date is not None and date > end_date:
            continue
        selected_dates.append(date)
        selected_returns.append(row)

    if not selected_returns:
        raise ValueError("No observations remain after date filtering")
    return selected_dates, np.asarray(selected_returns, dtype=float)


def write_simulation_csv(
    path: str | Path,
    simulated: dict[str, np.ndarray],
    asset_names: Sequence[str],
) -> None:
    """Writes simulated returns and prices to a long-format CSV file."""
    path = Path(path)
    prices = simulated["prices"]
    returns = simulated["returns"]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "path",
                "step",
                f"{asset_names[0]}_return",
                f"{asset_names[1]}_return",
                f"{asset_names[0]}_price",
                f"{asset_names[1]}_price",
            ]
        )
        for path_index in range(returns.shape[0]):
            for step in range(returns.shape[1]):
                writer.writerow(
                    [
                        path_index,
                        step + 1,
                        returns[path_index, step, 0],
                        returns[path_index, step, 1],
                        prices[path_index, step + 1, 0],
                        prices[path_index, step + 1, 1],
                    ]
                )


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 2 or returns.shape[1] != 2:
        raise ValueError("returns must have shape (n_observations, 2)")
    finite_rows = np.all(np.isfinite(returns), axis=1)
    returns = returns[finite_rows]
    if len(returns) < 30:
        raise ValueError("At least 30 valid observations are required")
    return returns


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("vol_window must be at least 2")
    if len(values) < window:
        raise ValueError("vol_window is longer than the return series")

    windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    return np.std(windows, axis=2, ddof=1)


def _as_pair(value: float | Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        values = tuple(float(item) for item in value)
        if len(values) != 2:
            raise ValueError(f"{name} must be a scalar or two values")
        return values
    scalar = float(value)
    return (scalar, scalar)


def _safe_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = np.std(values, ddof=1)
    if not math.isfinite(std) or std < EPSILON:
        return np.zeros_like(values)
    return (values - np.mean(values)) / std


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    if not math.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -0.99, 0.99))


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, 1e-8, None)
    positive = (eigenvectors * clipped) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(positive), 1e-12))
    corr = positive / np.outer(diagonal, diagonal)
    np.fill_diagonal(corr, 1.0)
    return (corr + corr.T) / 2.0


def _parse_pair(text: str | None, default: float) -> float | tuple[float, float]:
    if text is None:
        return default
    parts = [part.strip() for part in text.split(",")]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise argparse.ArgumentTypeError("Expected one scalar or two comma-separated values")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command line parser."""
    parser = argparse.ArgumentParser(
        description="Fit and simulate a two-asset SABR-style return model."
    )
    parser.add_argument("--input", default="train_sp500_us10y.csv", help="Training CSV path.")
    parser.add_argument("--date-column", default=None, help="Date column name.")
    parser.add_argument(
        "--asset-columns",
        default="sp500,DGS10",
        help="Two comma-separated return column names.",
    )
    parser.add_argument("--steps", type=int, default=TRADING_DAYS, help="Steps to simulate.")
    parser.add_argument("--paths", type=int, default=10, help="Number of paths.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--beta", default="1.0", help="Scalar or pair, e.g. 1.0 or 1.0,0.7.")
    parser.add_argument("--s0", default="100.0", help="Scalar or pair initial prices.")
    parser.add_argument("--vol-window", type=int, default=21, help="Rolling vol window.")
    parser.add_argument("--start-date", default=None, help="Inclusive training start date.")
    parser.add_argument("--end-date", default=None, help="Inclusive training end date.")
    parser.add_argument("--output", default="generated_sabr_series.csv", help="Output CSV path.")
    parser.add_argument("--params-output", default="sabr_params.json", help="Parameter JSON path.")
    return parser


def main() -> None:
    """Runs fit and simulation from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args()
    asset_columns = tuple(column.strip() for column in args.asset_columns.split(","))
    if len(asset_columns) != 2:
        raise ValueError("--asset-columns must contain exactly two names")

    dates, returns = read_return_csv(args.input, args.date_column, asset_columns)
    if args.start_date is not None or args.end_date is not None:
        _, returns = filter_by_date(dates, returns, args.start_date, args.end_date)
    params = fit_two_asset_sabr(
        returns,
        asset_names=asset_columns,
        beta=_parse_pair(args.beta, default=1.0),
        s0=_parse_pair(args.s0, default=100.0),
        vol_window=args.vol_window,
    )
    simulated = simulate_two_asset_sabr(
        params,
        n_steps=args.steps,
        n_paths=args.paths,
        seed=args.seed,
    )

    write_simulation_csv(args.output, simulated, asset_columns)
    params_path = Path(args.params_output)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        json.dumps(params.to_json_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
