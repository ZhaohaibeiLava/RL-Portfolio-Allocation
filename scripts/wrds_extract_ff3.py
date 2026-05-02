#!/usr/bin/env python3
"""Extract monthly Fama-French 3-factor data from WRDS.

This is the script form of ``03_wrds_extract_ff3.ipynb``. It preserves the
notebook's WRDS table, SQL fields, monthly date handling, and output filename.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import wrds
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("wrds_extract_ff3")

DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTDIR = "data/raw"
DEFAULT_FF_TABLE = "ff.factors_monthly"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract monthly Fama-French 3-factor data from WRDS."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--ff-table", default=DEFAULT_FF_TABLE)
    parser.add_argument("--wrds-username", default=None)
    parser.add_argument("--wrds-password", default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_date(value: str, arg_name: str) -> pd.Timestamp:
    """Parse a command-line date."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{arg_name} must be a valid date, got {value!r}.")
    return pd.Timestamp(parsed).normalize()


def connect_wrds(wrds_username: str | None, wrds_password: str | None) -> wrds.Connection:
    """Connect to WRDS using the same optional credential logic as the notebook."""
    LOGGER.info("Connecting to WRDS.")
    if wrds_username is not None and wrds_password is not None:
        return wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    if wrds_username is not None:
        return wrds.Connection(wrds_username=wrds_username)
    return wrds.Connection()


def fetch_ff3(
    db: wrds.Connection,
    ff_table: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Run the Fama-French SQL from the original notebook."""
    ff_sql = f"""
        SELECT
            dateff::date AS dateff,
            mktrf,
            smb,
            hml,
            rf
        FROM {ff_table}
        WHERE dateff::date BETWEEN CAST(%(start_date)s AS date) AND CAST(%(end_date)s AS date)
        ORDER BY dateff
    """

    ff = db.raw_sql(
        ff_sql,
        params={
            "start_date": start_date.date(),
            "end_date": end_date.date(),
        },
    )

    if ff.empty:
        raise RuntimeError(
            f"FF factors query returned zero rows from {ff_table!r}. "
            "The table may be unavailable for this WRDS account."
        )
    return ff


def clean_ff3(ff: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's factor cleaning."""
    result = ff.copy()
    result["dateff"] = pd.to_datetime(result["dateff"], errors="coerce")
    result = result.dropna(subset=["dateff"]).copy()
    result["month_end"] = result["dateff"] + MonthEnd(0)

    for col in ["mktrf", "smb", "hml", "rf"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    if result.empty:
        raise RuntimeError("FF factor rows were returned, but none had valid dates.")

    return result


def run(args: argparse.Namespace) -> None:
    """Run the extraction."""
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if start_date > end_date:
        raise ValueError("--start-date must be on or before --end-date.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    db = connect_wrds(args.wrds_username, args.wrds_password)
    try:
        ff = fetch_ff3(
            db=db,
            ff_table=args.ff_table,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        db.close()

    ff = clean_ff3(ff)

    ff_outfile = outdir / "ff3_monthly.parquet"
    ff.to_parquet(ff_outfile, engine="pyarrow", index=False)

    LOGGER.info("saved: %s", ff_outfile)
    LOGGER.info("rows: %s", f"{len(ff):,}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    try:
        run(args)
    except Exception:
        LOGGER.exception("WRDS FF3 extraction failed.")
        raise


if __name__ == "__main__":
    main()
