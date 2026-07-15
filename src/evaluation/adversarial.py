"""Adversarial diagnostics for 5-year bivariate financial time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_COLUMNS = ("sp500", "DGS10")


@dataclass(frozen=True)
class FeatureConfig:
    """Feature extraction settings."""

    window: int = 1260
    acf_lags: tuple[int, ...] = (1, 5, 20, 63)
    rolling_windows: tuple[int, ...] = (20, 63, 252)
    quantiles: tuple[float, ...] = (0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999)
    tail_levels: tuple[float, ...] = (0.01, 0.05)
    tick_sizes: tuple[float, ...] = (0.01,)


@dataclass(frozen=True)
class WindowRecord:
    """One fixed-length bivariate window."""

    label: str
    window_id: str
    source_path: str
    start: object
    end: object
    values: np.ndarray


@dataclass
class ReferenceDistribution:
    """Robust historical feature reference used for anomaly scoring."""

    feature_names: list[str]
    center: np.ndarray
    scale: np.ndarray
    mean_z: np.ndarray
    cov_inv: np.ndarray
    hist_robust_l2: np.ndarray
    hist_mahalanobis: np.ndarray
    hist_nearest: np.ndarray
    hist_z: np.ndarray

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Return robust z-scores aligned to the historical feature names."""
        x = features.loc[:, self.feature_names].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=self.center, posinf=self.center, neginf=self.center)
        return (x - self.center) / self.scale

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Score feature rows by distance from the historical reference."""
        z = self.transform(features)
        centered = z - self.mean_z
        robust_l2 = np.sqrt(np.mean(z**2, axis=1))
        max_abs_z = np.max(np.abs(z), axis=1)
        mahalanobis = np.sqrt(
            np.maximum(np.einsum("ij,jk,ik->i", centered, self.cov_inv, centered), 0.0)
            / max(z.shape[1], 1)
        )
        nearest = np.array([_nearest_distance(row, self.hist_z) for row in z])
        return pd.DataFrame(
            {
                "robust_l2": robust_l2,
                "robust_l2_pvalue": _upper_tail_pvalue(robust_l2, self.hist_robust_l2),
                "max_abs_z": max_abs_z,
                "mahalanobis": mahalanobis,
                "mahalanobis_pvalue": _upper_tail_pvalue(
                    mahalanobis,
                    self.hist_mahalanobis,
                ),
                "nearest_feature_distance": nearest,
                "nearest_feature_pvalue": _lower_tail_pvalue(nearest, self.hist_nearest),
            }
        )


def infer_value_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Infer SP500 and DGS10-like value columns from common generated formats."""
    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and str(column).lower() not in {"t", "path_id", "scenario_id"}
        and "cumsum" not in str(column).lower()
    ]
    lower = {column: str(column).lower() for column in numeric_columns}

    sp_candidates = [
        column
        for column in numeric_columns
        if "sp500" in lower[column] or "s&p" in lower[column] or "spx" in lower[column]
    ]
    dgs_candidates = [
        column
        for column in numeric_columns
        if "dgs10" in lower[column] or "us10y" in lower[column] or "rate" in lower[column]
    ]

    if sp_candidates:
        sp_col = sp_candidates[0]
    elif numeric_columns:
        sp_col = numeric_columns[0]
    else:
        raise ValueError("No numeric value columns were found.")

    if dgs_candidates:
        dgs_col = dgs_candidates[0]
    else:
        remaining = [column for column in numeric_columns if column != sp_col]
        if not remaining:
            raise ValueError("Could not infer a DGS10-like value column.")
        dgs_col = remaining[0]
    return sp_col, dgs_col


def load_bivariate_csv(
    path: Path,
    columns: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Load a CSV and normalize target columns to ``sp500`` and ``DGS10``."""
    df = pd.read_csv(path)
    if columns is not None and all(column in df.columns for column in columns):
        sp_col, dgs_col = columns
    else:
        sp_col, dgs_col = infer_value_columns(df)

    out = df.loc[:, [sp_col, dgs_col]].apply(pd.to_numeric, errors="coerce")
    out.columns = list(DEFAULT_COLUMNS)
    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path} has no aligned numeric SP500/DGS10 values.")
    return out


def load_generated_windows(
    path: Path,
    config: FeatureConfig,
    label: str,
    columns: tuple[str, str] | None = None,
    stride: int | None = None,
) -> list[WindowRecord]:
    """Load generated windows from either a single-path or path_id CSV."""
    raw = pd.read_csv(path)
    records: list[WindowRecord] = []
    if "path_id" in raw.columns:
        for path_id, group in raw.groupby("path_id", sort=True):
            value_df = _normalize_frame(group, columns)
            if len(value_df) < config.window:
                continue
            values = value_df.iloc[: config.window].to_numpy(dtype=float)
            records.append(
                WindowRecord(
                    label=label,
                    window_id=f"{label}_{path_id}",
                    source_path=str(path),
                    start=0,
                    end=config.window - 1,
                    values=values,
                )
            )
        return records

    value_df = _normalize_frame(raw, columns)
    window_stride = stride or config.window
    return make_windows(
        value_df,
        config.window,
        window_stride,
        label=label,
        source_path=str(path),
    )


def make_windows(
    df: pd.DataFrame,
    window: int,
    stride: int,
    label: str,
    source_path: str,
    max_windows: int | None = None,
) -> list[WindowRecord]:
    """Create fixed-length window records from a bivariate frame."""
    records = []
    for ordinal, start in enumerate(range(0, len(df) - window + 1, stride)):
        if max_windows is not None and ordinal >= max_windows:
            break
        end = start + window
        values = df.iloc[start:end].to_numpy(dtype=float)
        records.append(
            WindowRecord(
                label=label,
                window_id=f"{label}_{ordinal:04d}",
                source_path=source_path,
                start=start,
                end=end - 1,
                values=values,
            )
        )
    return records


def feature_frame(records: Iterable[WindowRecord], config: FeatureConfig) -> pd.DataFrame:
    """Extract adversarial feature vectors from windows."""
    rows = []
    for record in records:
        features = extract_features(record.values, config)
        features.update(
            {
                "label": record.label,
                "window_id": record.window_id,
                "source_path": record.source_path,
                "start": record.start,
                "end": record.end,
            }
        )
        rows.append(features)
    if not rows:
        return pd.DataFrame()
    metadata = ["label", "window_id", "source_path", "start", "end"]
    frame = pd.DataFrame(rows)
    feature_columns = sorted([column for column in frame.columns if column not in metadata])
    return frame.loc[:, metadata + feature_columns]


def extract_features(values: np.ndarray, config: FeatureConfig | None = None) -> dict[str, float]:
    """Extract one feature vector from a 2D time series."""
    config = config or FeatureConfig()
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected values with shape (n, 2), got {values.shape}")
    x = values[:, 0].astype(float)
    y = values[:, 1].astype(float)
    features: dict[str, float] = {}
    features.update(_series_features(x, "sp500", config))
    features.update(_series_features(y, "DGS10", config))
    features.update(_dependence_features(x, y, config))
    return features


def fit_reference(real_features: pd.DataFrame) -> ReferenceDistribution:
    """Fit a robust reference distribution from historical real windows."""
    feature_names = _feature_columns(real_features)
    x = real_features.loc[:, feature_names].to_numpy(dtype=float)
    col_median = np.nanmedian(x, axis=0)
    x = np.nan_to_num(x, nan=col_median, posinf=col_median, neginf=col_median)
    center = np.median(x, axis=0)
    q75 = np.quantile(x, 0.75, axis=0)
    q25 = np.quantile(x, 0.25, axis=0)
    robust_scale = (q75 - q25) / 1.349
    std_scale = np.std(x, axis=0, ddof=1)
    scale = np.where(robust_scale > 1e-12, robust_scale, std_scale)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - center) / scale
    mean_z = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    diag = np.diag(np.diag(cov))
    shrink = 0.15
    cov = (1.0 - shrink) * cov + shrink * diag
    cov = cov + np.eye(cov.shape[0]) * 1e-4
    cov_inv = np.linalg.pinv(cov)
    centered = z - mean_z
    hist_mahalanobis = np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", centered, cov_inv, centered), 0.0)
        / max(z.shape[1], 1)
    )
    hist_robust_l2 = np.sqrt(np.mean(z**2, axis=1))
    hist_nearest = _leave_one_nearest(z)
    return ReferenceDistribution(
        feature_names=feature_names,
        center=center,
        scale=scale,
        mean_z=mean_z,
        cov_inv=cov_inv,
        hist_robust_l2=hist_robust_l2,
        hist_mahalanobis=hist_mahalanobis,
        hist_nearest=hist_nearest,
        hist_z=z,
    )


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return sorted numeric feature columns."""
    return _feature_columns(frame)


def distribution_distance(
    real_features: pd.DataFrame,
    fake_features: pd.DataFrame,
    reference: ReferenceDistribution,
) -> dict[str, float]:
    """Compare fake and real feature distributions in robust-z space."""
    real_z = reference.transform(real_features)
    fake_z = reference.transform(fake_features)
    mean_gap = np.sqrt(np.mean((fake_z.mean(axis=0) - real_z.mean(axis=0)) ** 2))
    std_gap = np.sqrt(np.mean((fake_z.std(axis=0) - real_z.std(axis=0)) ** 2))
    energy = energy_distance(real_z, fake_z)
    scores = reference.score(fake_features)
    return {
        "feature_mean_gap": float(mean_gap),
        "feature_std_gap": float(std_gap),
        "feature_energy_distance": float(energy),
        "median_mahalanobis": float(scores["mahalanobis"].median()),
        "median_robust_l2": float(scores["robust_l2"].median()),
        "min_mahalanobis_pvalue": float(scores["mahalanobis_pvalue"].min()),
        "outlier_rate_p05": float((scores["mahalanobis_pvalue"] < 0.05).mean()),
        "median_nearest_feature_distance": float(scores["nearest_feature_distance"].median()),
    }


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Return the multivariate energy distance between two samples."""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    xy = _pairwise_l2(x, y).mean()
    xx = _pairwise_l2(x, x).mean()
    yy = _pairwise_l2(y, y).mean()
    return float(2.0 * xy - xx - yy)


def train_feature_classifier(
    real_features: pd.DataFrame,
    fake_features: pd.DataFrame,
    seed: int = 0,
    epochs: int = 300,
) -> dict[str, float]:
    """Train a small MLP discriminator on feature vectors."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return {"classifier_auc": np.nan, "classifier_accuracy": np.nan}

    feature_names = sorted(set(_feature_columns(real_features)) & set(_feature_columns(fake_features)))
    if len(real_features) < 12 or len(fake_features) < 12 or not feature_names:
        return {"classifier_auc": np.nan, "classifier_accuracy": np.nan}

    rng = np.random.default_rng(seed)
    n = min(len(real_features), len(fake_features))
    real_idx = rng.choice(len(real_features), n, replace=False)
    fake_idx = rng.choice(len(fake_features), n, replace=False)
    x_real = real_features.iloc[real_idx].loc[:, feature_names].to_numpy(dtype=float)
    x_fake = fake_features.iloc[fake_idx].loc[:, feature_names].to_numpy(dtype=float)
    x = np.vstack([x_real, x_fake])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    x = np.nan_to_num(x, nan=np.nanmedian(x, axis=0), posinf=0.0, neginf=0.0)

    order = rng.permutation(len(y))
    x = x[order]
    y = y[order]
    split = max(4, int(0.7 * len(y)))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.10),
        nn.Linear(64, 32),
        nn.LeakyReLU(0.2),
        nn.Linear(32, 1),
    )
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train[:, None], dtype=torch.float32)
    for _ in range(epochs):
        logits = model(x_train_t)
        loss = loss_fn(logits, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32))).numpy().ravel()
    predicted = (scores >= 0.5).astype(int)
    accuracy = float((predicted == y_test).mean())
    auc = _auc_score(y_test, scores)
    return {"classifier_auc": float(auc), "classifier_accuracy": accuracy}


def exact_row_match_fraction(fake_values: np.ndarray, real_values: np.ndarray, decimals: int = 12) -> float:
    """Return the fraction of fake rows exactly appearing in real data after rounding."""
    rounded_real = np.round(real_values, decimals=decimals)
    rounded_fake = np.round(fake_values, decimals=decimals)
    real_rows = {tuple(row) for row in rounded_real}
    matches = sum(tuple(row) in real_rows for row in rounded_fake)
    return float(matches / max(len(rounded_fake), 1))


def nearest_raw_window_rms(
    fake_values: np.ndarray,
    real_values: np.ndarray,
    stride: int = 1,
) -> tuple[float, int]:
    """Return nearest standardized RMS distance to a real 5-year window."""
    if len(real_values) < len(fake_values):
        return np.nan, -1
    scale = real_values.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    fake = fake_values / scale
    best = np.inf
    best_start = -1
    for start in range(0, len(real_values) - len(fake_values) + 1, stride):
        candidate = real_values[start : start + len(fake_values)] / scale
        rms = float(np.sqrt(np.mean((fake - candidate) ** 2)))
        if rms < best:
            best = rms
            best_start = start
    return best, best_start


def _normalize_frame(
    raw: pd.DataFrame,
    columns: tuple[str, str] | None,
) -> pd.DataFrame:
    if columns is not None and all(column in raw.columns for column in columns):
        sp_col, dgs_col = columns
    else:
        sp_col, dgs_col = infer_value_columns(raw)
    out = raw.loc[:, [sp_col, dgs_col]].apply(pd.to_numeric, errors="coerce")
    out.columns = list(DEFAULT_COLUMNS)
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {"label", "window_id", "source_path", "start", "end"}
    return sorted(
        [
            column
            for column in frame.columns
            if column not in metadata and pd.api.types.is_numeric_dtype(frame[column])
        ]
    )


def _series_features(
    values: np.ndarray,
    prefix: str,
    config: FeatureConfig,
) -> dict[str, float]:
    features: dict[str, float] = {}
    x = np.asarray(values, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    centered = x - mean
    safe_std = std if std > 1e-12 else 1.0
    features[f"{prefix}_mean"] = mean
    features[f"{prefix}_std"] = std
    features[f"{prefix}_skewness"] = float(np.mean((centered / safe_std) ** 3))
    features[f"{prefix}_excess_kurtosis"] = float(np.mean((centered / safe_std) ** 4) - 3.0)
    features[f"{prefix}_min"] = float(np.min(x))
    features[f"{prefix}_max"] = float(np.max(x))
    features[f"{prefix}_positive_fraction"] = float(np.mean(x > 0.0))
    features[f"{prefix}_zero_fraction"] = float(np.mean(np.isclose(x, 0.0, atol=1e-12)))
    features[f"{prefix}_sign_switch_fraction"] = float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])))

    for quantile in config.quantiles:
        features[f"{prefix}_q_{quantile:g}"] = float(np.quantile(x, quantile))
    for alpha in config.tail_levels:
        lower = np.quantile(x, alpha)
        upper = np.quantile(x, 1.0 - alpha)
        features[f"{prefix}_es_low_{alpha:g}"] = float(x[x <= lower].mean())
        features[f"{prefix}_es_high_{alpha:g}"] = float(x[x >= upper].mean())
    for lag in config.acf_lags:
        features[f"{prefix}_acf_{lag}"] = _acf(x, lag)
        features[f"{prefix}_abs_acf_{lag}"] = _acf(np.abs(x), lag)
        features[f"{prefix}_squared_acf_{lag}"] = _acf(x**2, lag)

    cumulative = np.cumsum(x)
    peak = np.maximum.accumulate(cumulative)
    trough = np.minimum.accumulate(cumulative)
    features[f"{prefix}_cumsum_terminal"] = float(cumulative[-1])
    features[f"{prefix}_cumsum_min"] = float(cumulative.min())
    features[f"{prefix}_cumsum_max"] = float(cumulative.max())
    features[f"{prefix}_cumsum_range"] = float(cumulative.max() - cumulative.min())
    features[f"{prefix}_max_drawdown"] = float((cumulative - peak).min())
    features[f"{prefix}_max_drawup"] = float((cumulative - trough).max())

    for tick_size in config.tick_sizes:
        rounded = np.round(x / tick_size) * tick_size
        fraction = np.mean(np.isclose(x, rounded, atol=max(tick_size * 1e-8, 1e-12)))
        features[f"{prefix}_tick_{tick_size:g}_fraction"] = float(fraction)

    series = pd.Series(x)
    for window in config.rolling_windows:
        if len(series) < window:
            continue
        rolling_vol = series.rolling(window).std(ddof=0).dropna()
        if rolling_vol.empty:
            continue
        vol_values = rolling_vol.to_numpy(dtype=float)
        features[f"{prefix}_roll{window}_vol_mean"] = float(np.mean(vol_values))
        features[f"{prefix}_roll{window}_vol_std"] = float(np.std(vol_values, ddof=0))
        features[f"{prefix}_roll{window}_vol_q05"] = float(np.quantile(vol_values, 0.05))
        features[f"{prefix}_roll{window}_vol_q50"] = float(np.quantile(vol_values, 0.50))
        features[f"{prefix}_roll{window}_vol_q95"] = float(np.quantile(vol_values, 0.95))
        features[f"{prefix}_roll{window}_vol_acf1"] = _acf(vol_values, 1)
    return features


def _dependence_features(
    x: np.ndarray,
    y: np.ndarray,
    config: FeatureConfig,
) -> dict[str, float]:
    features: dict[str, float] = {}
    features["corr"] = _corr(x, y)
    features["spearman_corr"] = _corr(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())
    features["abs_corr"] = _corr(np.abs(x), np.abs(y))
    features["squared_corr"] = _corr(x**2, y**2)
    for lag in config.acf_lags:
        if len(x) > lag:
            features[f"crosscorr_sp500_to_DGS10_lag{lag}"] = _corr(x[:-lag], y[lag:])
            features[f"crosscorr_DGS10_to_sp500_lag{lag}"] = _corr(y[:-lag], x[lag:])
    for alpha in config.tail_levels:
        x_low = x <= np.quantile(x, alpha)
        x_high = x >= np.quantile(x, 1.0 - alpha)
        y_low = y <= np.quantile(y, alpha)
        y_high = y >= np.quantile(y, 1.0 - alpha)
        denom = max(float(x_low.mean()), 1e-12)
        features[f"lower_tail_dependence_{alpha:g}"] = float((x_low & y_low).mean() / denom)
        features[f"upper_tail_dependence_{alpha:g}"] = float((x_high & y_high).mean() / max(float(x_high.mean()), 1e-12))
        features[f"sp500_low_DGS10_high_tail_{alpha:g}"] = float((x_low & y_high).mean() / denom)

    x_q10 = np.quantile(x, 0.10)
    x_q90 = np.quantile(x, 0.90)
    features["downside_corr"] = _masked_corr(x, y, x <= x_q10)
    features["upside_corr"] = _masked_corr(x, y, x >= x_q90)

    for window in config.rolling_windows:
        if len(x) < window:
            continue
        x_series = pd.Series(x)
        y_series = pd.Series(y)
        rolling_corr = x_series.rolling(window).corr(y_series).dropna()
        rolling_vol = x_series.rolling(window).std(ddof=0).dropna()
        if rolling_corr.empty:
            continue
        corr_values = rolling_corr.to_numpy(dtype=float)
        features[f"roll{window}_corr_mean"] = float(np.nanmean(corr_values))
        features[f"roll{window}_corr_std"] = float(np.nanstd(corr_values))
        features[f"roll{window}_corr_q05"] = float(np.nanquantile(corr_values, 0.05))
        features[f"roll{window}_corr_q50"] = float(np.nanquantile(corr_values, 0.50))
        features[f"roll{window}_corr_q95"] = float(np.nanquantile(corr_values, 0.95))
        if not rolling_vol.empty:
            aligned_corr = rolling_corr.iloc[-len(rolling_vol) :].to_numpy(dtype=float)
            vol_values = rolling_vol.to_numpy(dtype=float)
            high_vol = vol_values >= np.quantile(vol_values, 0.80)
            low_vol = vol_values <= np.quantile(vol_values, 0.20)
            features[f"roll{window}_corr_high_sp500_vol_mean"] = float(np.nanmean(aligned_corr[high_vol]))
            features[f"roll{window}_corr_low_sp500_vol_mean"] = float(np.nanmean(aligned_corr[low_vol]))
            features[f"roll{window}_corr_vol_corr"] = _corr(aligned_corr, vol_values)
    return features


def _acf(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag:
        return 0.0
    x0 = values[:-lag]
    x1 = values[lag:]
    return _corr(x0, x1)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    x_std = x.std(ddof=0)
    y_std = y.std(ddof=0)
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x_std * y_std))


def _masked_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 3:
        return 0.0
    return _corr(x[mask], y[mask])


def _pairwise_l2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return np.sqrt(np.mean(diff**2, axis=2))


def _nearest_distance(row: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - row[None, :]) ** 2, axis=1)).min())


def _leave_one_nearest(values: np.ndarray) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        if len(values) == 1:
            out[i] = np.nan
            continue
        candidate = np.delete(values, i, axis=0)
        out[i] = _nearest_distance(values[i], candidate)
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        out[:] = 0.0
    else:
        out = np.nan_to_num(out, nan=float(np.nanmedian(finite)))
    return out


def _upper_tail_pvalue(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    return np.array([(reference >= value).mean() for value in values], dtype=float)


def _lower_tail_pvalue(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    return np.array([(reference <= value).mean() for value in values], dtype=float)


def _auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = positives.sum()
    n_neg = negatives.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[positives].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)
