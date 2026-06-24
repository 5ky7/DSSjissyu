"""Evaluate generated paths against real data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.compare import comparison_table


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="data/processed/daily_features.csv")
    parser.add_argument(
        "--generated",
        nargs="+",
        default=[
            "data/generated/bootstrap/generated_paths.csv",
            "data/generated/dcc/generated_paths.csv",
            "data/generated/adcc/generated_paths.csv",
            "data/generated/copula/generated_paths.csv",
            "data/generated/regime/generated_paths.csv",
        ],
    )
    parser.add_argument("--output", default="reports/tables/model_comparison.csv")
    parser.add_argument("--version", choices=["A", "B"], default="B")
    return parser.parse_args()


def main() -> None:
    """Run generated-path evaluation."""
    args = parse_args()
    real = pd.read_csv(args.real)
    x_col = "sp500_return"
    y_col = "dgs10_diff" if args.version == "A" else "bond_return_proxy"
    generated_frames = []
    for path in args.generated:
        generated_path = Path(path)
        if generated_path.exists():
            generated_frames.append(pd.read_csv(generated_path))
    if not generated_frames:
        raise FileNotFoundError("No generated path files were found")
    generated = pd.concat(generated_frames, ignore_index=True)
    table = comparison_table(real, generated, x_col, y_col)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
