"""Generate bootstrap scenario paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bootstrap import (
    circular_block_bootstrap,
    iid_bootstrap,
    moving_block_bootstrap,
    stationary_bootstrap,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/daily_features.csv")
    parser.add_argument("--output", default="data/generated/bootstrap/generated_paths.csv")
    parser.add_argument("--version", choices=["A", "B"], default="B")
    parser.add_argument("--path-length", type=int, default=1260)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run bootstrap generation."""
    args = parse_args()
    df = pd.read_csv(args.input)
    columns = (
        ["sp500_return", "dgs10_diff"]
        if args.version == "A"
        else ["sp500_return", "bond_return_proxy"]
    )
    generated = pd.concat(
        [
            iid_bootstrap(df, columns, args.path_length, args.n_paths, args.seed),
            moving_block_bootstrap(
                df,
                columns,
                args.path_length,
                args.n_paths,
                args.block_length,
                args.seed + 1,
            ),
            stationary_bootstrap(
                df,
                columns,
                args.path_length,
                args.n_paths,
                args.block_length,
                args.seed + 2,
            ),
            circular_block_bootstrap(
                df,
                columns,
                args.path_length,
                args.n_paths,
                args.block_length,
                args.seed + 3,
            ),
        ],
        ignore_index=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_csv(output, index=False)
    print(f"Wrote {output} ({len(generated)} rows)")


if __name__ == "__main__":
    main()
