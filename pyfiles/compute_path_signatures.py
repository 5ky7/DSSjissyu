"""Compute path signature features for masked mixed datasets.

The script treats each mask pair, ``(mask*_sp500, mask*_DGS10)``, as one
two-dimensional path and computes its truncated path signature up to level 5.
Results are written as one feature row per dataset and mask.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUTS = (
    ("mixed_brown", Path("data/mixed_data_1st/mixed_brown_masked.csv")),
    ("mixed_sabr", Path("data/mixed_data_1st/mixed_sabr_masked.csv")),
)
DEFAULT_OUTPUT = Path("data/processed/path_signatures_level5.csv")
DEFAULT_DEPTH = 5
MASK_IDS = range(1, 6)
CHANNELS = ("sp500", "DGS10")
INPUT_ALIASES = {
    "mix_sabr_mask.csv": Path("data/mixed_data_1st/mixed_sabr_masked.csv"),
    "mixed_sabr_mask.csv": Path("data/mixed_data_1st/mixed_sabr_masked.csv"),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute path signature features for masked mixed datasets.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help=f"Truncation depth of path signatures. Default: {DEFAULT_DEPTH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--include-time",
        action="store_true",
        help="Add a normalized time channel before computing signatures.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        metavar="NAME=PATH",
        help=(
            "Optional dataset definitions. Example: "
            "brown=data/mixed_data_1st/mixed_brown_masked.csv"
        ),
    )
    return parser.parse_args()


def dataset_specs(input_args: list[str] | None) -> list[tuple[str, Path]]:
    """Return dataset names and input paths."""
    if input_args is None:
        return list(DEFAULT_INPUTS)

    specs = []
    for item in input_args:
        if "=" not in item:
            raise ValueError(f"Input must be NAME=PATH format: {item}")
        name, path = item.split("=", maxsplit=1)
        if not name:
            raise ValueError(f"Dataset name is empty: {item}")
        specs.append((name, Path(path)))
    return specs


def resolve_input_path(path: Path) -> Path:
    """Resolve known filename aliases and validate that the input exists."""
    if path.exists():
        return path

    alias_path = INPUT_ALIASES.get(path.name)
    if alias_path is not None and alias_path.exists():
        return alias_path

    raise FileNotFoundError(f"Input CSV does not exist: {path}")


def signature_feature_names(channels: tuple[str, ...], depth: int) -> list[str]:
    """Return feature names ordered by signature level and tensor word."""
    names = []
    for level in range(1, depth + 1):
        for word in product(channels, repeat=level):
            names.append("sig_" + "_".join(word))
    return names


def segment_signature(increment: np.ndarray, depth: int) -> list[np.ndarray]:
    """Compute the truncated signature of one linear segment."""
    levels = []
    current = np.array([1.0])
    factorial = 1.0

    for level in range(1, depth + 1):
        current = np.kron(current, increment)
        factorial *= level
        levels.append(current / factorial)

    return levels


def chen_product(
    left: list[np.ndarray],
    right: list[np.ndarray],
    depth: int,
) -> list[np.ndarray]:
    """Combine two truncated signatures using Chen's identity."""
    combined = []

    for level in range(1, depth + 1):
        level_value = left[level - 1] + right[level - 1]
        for split in range(1, level):
            level_value = level_value + np.kron(
                left[split - 1],
                right[level - split - 1],
            )
        combined.append(level_value)

    return combined


def path_signature(path: np.ndarray, depth: int) -> np.ndarray:
    """Compute the truncated signature of a piecewise linear path."""
    if path.ndim != 2:
        raise ValueError("Path must be a two-dimensional array.")
    if len(path) < 2:
        raise ValueError("Path must have at least two observations.")

    signature = [
        np.zeros(path.shape[1] ** level, dtype=float)
        for level in range(1, depth + 1)
    ]

    for increment in np.diff(path, axis=0):
        signature = chen_product(signature, segment_signature(increment, depth), depth)

    return np.concatenate(signature)


def mask_path(
    df: pd.DataFrame,
    mask_id: int,
    include_time: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the path array for one mask."""
    columns = [f"mask{mask_id}_{channel}" for channel in CHANNELS]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for mask{mask_id}: {missing}")

    path = df[columns].to_numpy(dtype=float)
    if not np.isfinite(path).all():
        raise ValueError(f"Non-finite values found in mask{mask_id}: {columns}")

    channels = CHANNELS

    if include_time:
        time = np.linspace(0.0, 1.0, len(path), dtype=float).reshape(-1, 1)
        path = np.column_stack([time, path])
        channels = ("time",) + CHANNELS

    return path, channels


def compute_dataset_signatures(
    name: str,
    path: Path,
    depth: int,
    include_time: bool,
) -> pd.DataFrame:
    """Compute signatures for all masks in one dataset."""
    input_path = resolve_input_path(path)
    df = pd.read_csv(input_path)
    rows = []

    for mask_id in MASK_IDS:
        mask_values, channels = mask_path(df, mask_id, include_time)
        feature_names = signature_feature_names(channels, depth)
        signature = path_signature(mask_values, depth)

        row = {
            "dataset": name,
            "source_file": str(input_path),
            "mask_id": mask_id,
            "n_observations": len(mask_values),
            "depth": depth,
            "include_time": include_time,
        }
        row.update(dict(zip(feature_names, signature)))
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    """Run path signature calculation and save the feature table."""
    args = parse_args()
    if args.depth < 1:
        raise ValueError("--depth must be at least 1.")

    outputs = [
        compute_dataset_signatures(name, path, args.depth, args.include_time)
        for name, path in dataset_specs(args.inputs)
    ]
    result = pd.concat(outputs, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} rows and {len(result.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
