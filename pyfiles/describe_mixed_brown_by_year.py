"""Print yearly describe() summaries for the masked Brownian dataset."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("mixed_data_1st/mixed_brown_masked.csv")
POINTS_PER_YEAR = 252
MASK_IDS = range(1, 6)
SUFFIXES = ("sp500", "DGS10")


def load_data(path: Path) -> pd.DataFrame:
    """Load the masked Brownian dataset."""
    return pd.read_csv(path)


def mask_columns(suffix: str) -> list[str]:
    """Return mask columns for the given suffix in mask order."""
    return [f"mask{mask_id}_{suffix}" for mask_id in MASK_IDS]


def print_yearly_describe(
    df: pd.DataFrame,
    points_per_year: int,
    suffix: str,
) -> None:
    """Print pandas describe() output for each yearly chunk of one suffix."""
    total_rows = len(df)
    columns = mask_columns(suffix)

    print("#" * 80)
    print(suffix)
    print("#" * 80)
    print()

    for start in range(0, total_rows, points_per_year):
        end = min(start + points_per_year, total_rows)
        year_index = start // points_per_year + 1
        yearly_df = df.iloc[start:end][columns]

        print("=" * 80)
        print(f"Year {year_index}: rows {start} to {end - 1} ({len(yearly_df)} points)")
        print("=" * 80)
        print(yearly_df.describe().to_string())
        print()


def main() -> None:
    """Run yearly describe() summaries."""
    df = load_data(DATA_PATH)

    for suffix in SUFFIXES:
        print_yearly_describe(df, POINTS_PER_YEAR, suffix)


if __name__ == "__main__":
    main()
