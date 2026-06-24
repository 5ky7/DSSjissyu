"""Bootstrap scenario generators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.random import make_rng


def iid_bootstrap(
    data: pd.DataFrame,
    columns: list[str],
    path_length: int,
    n_paths: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate paths by independently resampling historical rows."""
    rng = make_rng(seed)
    values = data[columns].dropna().to_numpy()
    paths = []
    for path_id in range(n_paths):
        indices = rng.integers(0, len(values), size=path_length)
        paths.append(_path_frame(values[indices], columns, path_id, "iid_bootstrap"))
    return pd.concat(paths, ignore_index=True)


def moving_block_bootstrap(
    data: pd.DataFrame,
    columns: list[str],
    path_length: int,
    n_paths: int,
    block_length: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate paths with moving non-circular blocks."""
    rng = make_rng(seed)
    values = data[columns].dropna().to_numpy()
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if block_length > len(values):
        raise ValueError("block_length cannot exceed data length")

    max_start = len(values) - block_length
    paths = []
    for path_id in range(n_paths):
        samples = []
        while sum(len(block) for block in samples) < path_length:
            start = rng.integers(0, max_start + 1)
            samples.append(values[start : start + block_length])
        sampled = np.vstack(samples)[:path_length]
        paths.append(
            _path_frame(sampled, columns, path_id, f"moving_block_{block_length}")
        )
    return pd.concat(paths, ignore_index=True)


def stationary_bootstrap(
    data: pd.DataFrame,
    columns: list[str],
    path_length: int,
    n_paths: int,
    block_length: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate paths using Politis-Romano stationary bootstrap."""
    rng = make_rng(seed)
    values = data[columns].dropna().to_numpy()
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    continuation_probability = 1.0 - (1.0 / block_length)
    paths = []
    for path_id in range(n_paths):
        sampled = np.empty((path_length, len(columns)))
        index = rng.integers(0, len(values))
        for t in range(path_length):
            if t > 0 and rng.random() > continuation_probability:
                index = rng.integers(0, len(values))
            sampled[t] = values[index]
            index = (index + 1) % len(values)
        paths.append(
            _path_frame(sampled, columns, path_id, f"stationary_{block_length}")
        )
    return pd.concat(paths, ignore_index=True)


def circular_block_bootstrap(
    data: pd.DataFrame,
    columns: list[str],
    path_length: int,
    n_paths: int,
    block_length: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate paths with circular blocks that wrap around the sample end."""
    rng = make_rng(seed)
    values = data[columns].dropna().to_numpy()
    if block_length <= 0:
        raise ValueError("block_length must be positive")

    paths = []
    offsets = np.arange(block_length)
    for path_id in range(n_paths):
        samples = []
        while sum(len(block) for block in samples) < path_length:
            start = rng.integers(0, len(values))
            indices = (start + offsets) % len(values)
            samples.append(values[indices])
        sampled = np.vstack(samples)[:path_length]
        paths.append(
            _path_frame(sampled, columns, path_id, f"circular_block_{block_length}")
        )
    return pd.concat(paths, ignore_index=True)


def _path_frame(
    values: np.ndarray,
    columns: list[str],
    path_id: int,
    model: str,
) -> pd.DataFrame:
    """Create a long-format path DataFrame."""
    frame = pd.DataFrame(values, columns=columns)
    frame.insert(0, "t", np.arange(len(frame)))
    frame.insert(0, "path_id", path_id)
    frame.insert(0, "model", model)
    return frame
