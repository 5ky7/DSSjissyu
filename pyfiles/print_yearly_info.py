"""Print pandas info for each 252-row block of the masked Brownian dataset."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("mixed_data_1st/mixed_brown_masked.csv")
ROWS_PER_YEAR = 252


def load_data(path: Path) -> pd.DataFrame:
    """Load the masked Brownian dataset."""
    return pd.read_csv(path)


def print_yearly_info(df: pd.DataFrame, rows_per_year: int) -> None:
    """Print DataFrame.info() for each yearly block."""
    for start in range(0, len(df), rows_per_year):
        end = min(start + rows_per_year, len(df))
        year_index = start // rows_per_year + 1
        yearly_df = df.iloc[start:end]

        print("=" * 80)
        print(f"Year block {year_index}: rows {start} to {end - 1} ({len(yearly_df)} rows)")
        print("=" * 80)
        yearly_df.info()
        print()


def main() -> None:
    """Run yearly info output."""
    df = load_data(DATA_PATH)
    print_yearly_info(df, ROWS_PER_YEAR)


if __name__ == "__main__":
    main()
