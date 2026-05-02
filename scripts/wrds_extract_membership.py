#!/usr/bin/env python3
"""Extract monthly S&P 500 membership from WRDS/CRSP.

This is the script form of ``01_wrds_extract_membership.ipynb``. The extraction
logic is intentionally kept the same: query ``crsp.dsp500list``, expand each
membership interval to month-end observations, and save the monthly membership
parquet used by downstream CRSP panel extraction.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import wrds
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("wrds_extract_membership")

DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTDIR = "data/raw"
DEFAULT_MEMBERSHIP_TABLE = "crsp.dsp500list"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract monthly S&P 500 membership from WRDS/CRSP."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--membership-table", default=DEFAULT_MEMBERSHIP_TABLE)
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


def iter_month_ends(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DatetimeIndex:
    """Return month ends in the closed interval used by the original notebook."""
    first_month_end = start_dt + MonthEnd(0)
    if first_month_end > end_dt:
        return pd.DatetimeIndex([])
    return pd.date_range(first_month_end, end_dt, freq=MonthEnd())


def fetch_membership(
    db: wrds.Connection,
    membership_table: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Run the membership SQL from the original notebook."""
    membership_sql = f"""
        SELECT DISTINCT
            permno::integer AS permno,
            GREATEST("start"::date, CAST(%(start_date)s AS date)) AS start_date,
            LEAST(
                COALESCE(ending::date, CAST(%(end_date)s AS date)),
                CAST(%(end_date)s AS date)
            ) AS ending_date
        FROM {membership_table}
        WHERE permno IS NOT NULL
          AND "start"::date <= CAST(%(end_date)s AS date)
          AND COALESCE(ending::date, CAST(%(end_date)s AS date)) >= CAST(%(start_date)s AS date)
        ORDER BY permno, start_date, ending_date
    """

    membership = db.raw_sql(
        membership_sql,
        params={
            "start_date": start_date.date(),
            "end_date": end_date.date(),
        },
    )

    if membership.empty:
        raise RuntimeError(
            f"Membership query returned zero rows from {membership_table!r}. "
            "The table may be unavailable for this WRDS account."
        )
    return membership


def clean_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """Apply the same type cleaning as the original notebook."""
    result = membership.copy()
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce").astype("Int64")
    result["start_date"] = pd.to_datetime(result["start_date"], errors="coerce")
    result["ending_date"] = pd.to_datetime(result["ending_date"], errors="coerce")
    result = result.dropna(subset=["permno", "start_date", "ending_date"]).copy()
    result["permno"] = result["permno"].astype(int)

    if result.empty:
        raise RuntimeError(
            "Membership rows were returned, but none had valid permno/start/end values."
        )
    return result


def expand_monthly_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """Expand CRSP membership ranges to one row per member month end."""
    pieces = []
    for row in membership.itertuples(index=False):
        month_ends = iter_month_ends(row.start_date, row.ending_date)
        if len(month_ends) == 0:
            continue
        pieces.append(pd.DataFrame({"permno": int(row.permno), "month_end": month_ends}))

    if not pieces:
        raise RuntimeError("No valid month-end membership observations could be constructed.")

    return (
        pd.concat(pieces, ignore_index=True)
        .drop_duplicates(subset=["permno", "month_end"])
        .sort_values(["month_end", "permno"])
        .reset_index(drop=True)
    )


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
        membership = fetch_membership(
            db=db,
            membership_table=args.membership_table,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        db.close()

    membership = clean_membership(membership)
    membership_monthly = expand_monthly_membership(membership)

    membership_outfile = outdir / "sp500_membership_monthly.parquet"
    membership_monthly.to_parquet(membership_outfile, engine="pyarrow", index=False)

    LOGGER.info("saved: %s", membership_outfile)
    LOGGER.info("rows: %s", f"{len(membership_monthly):,}")
    LOGGER.info("permnos: %s", f"{membership_monthly['permno'].nunique():,}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    try:
        run(args)
    except Exception:
        LOGGER.exception("WRDS membership extraction failed.")
        raise


if __name__ == "__main__":
    main()
