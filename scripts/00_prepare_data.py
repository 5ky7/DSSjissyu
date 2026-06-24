"""Prepare daily SP500/DGS10 feature data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import PreprocessConfig, preprocess_csv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_sp500_us10y.csv")
    parser.add_argument("--output", default="data/processed/daily_features.csv")
    parser.add_argument("--duration-years", type=float, default=8.0)
    parser.add_argument("--business-days", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run preprocessing."""
    args = parse_args()
    config = PreprocessConfig(
        duration_years=args.duration_years,
        reindex_business_days=args.business_days,
    )
    features = preprocess_csv(args.input, args.output, config)
    print(f"Wrote {args.output} ({len(features)} rows)")


if __name__ == "__main__":
    main()
