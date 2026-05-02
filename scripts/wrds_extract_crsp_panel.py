#!/usr/bin/env python3
"""Extract a CRSP monthly panel for S&P 500 member months.

This is the script form of ``02_wrds_extract_crsp_panel.ipynb``. It preserves
the notebook's query, cleaning, adjusted-return calculation, membership join,
and output filename.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import wrds
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("wrds_extract_crsp_panel")

DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTDIR = "data/raw"
DEFAULT_CRSP_TABLE = "crsp.msf"
DEFAULT_CRSP_EVENT_TABLE = "crsp.mseall"
DEFAULT_CRSP_NAMES_TABLE = "crsp.msenames"
DEFAULT_MEMBERSHIP_FILE = "sp500_membership_monthly.parquet"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract CRSP monthly stock data for S&P 500 member months."
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--crsp-table", default=DEFAULT_CRSP_TABLE)
    parser.add_argument("--crsp-event-table", default=DEFAULT_CRSP_EVENT_TABLE)
    parser.add_argument("--crsp-names-table", default=DEFAULT_CRSP_NAMES_TABLE)
    parser.add_argument("--membership-file", default=None)
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


def load_membership(path: Path) -> tuple[pd.DataFrame, list[int]]:
    """Load the monthly membership parquet and extract PERMNOs."""
    if not path.exists():
        raise FileNotFoundError(f"Missing membership parquet: {path}")

    membership_monthly = pd.read_parquet(path)
    membership_monthly["month_end"] = pd.to_datetime(membership_monthly["month_end"])
    permnos = sorted(membership_monthly["permno"].dropna().astype(int).unique().tolist())

    if not permnos:
        raise RuntimeError("The membership parquet exists, but it contains no PERMNOs.")

    return membership_monthly, permnos


def fetch_crsp(
    db: wrds.Connection,
    permnos: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    crsp_table: str,
    crsp_event_table: str,
    crsp_names_table: str,
) -> pd.DataFrame:
    """Run the CRSP SQL from the original notebook."""
    crsp_sql = f"""
        SELECT
            msf.permno::integer AS permno,
            msf."date"::date AS date,
            msf.ret,
            msf.retx,
            mse.dlret,
            msf.prc,
            msf.shrout,
            msf.vol,
            names.exchcd,
            names.shrcd
        FROM {crsp_table} AS msf
        LEFT JOIN {crsp_event_table} AS mse
          ON msf.permno = mse.permno
         AND msf."date" = mse."date"
        LEFT JOIN {crsp_names_table} AS names
          ON msf.permno = names.permno
         AND names.namedt <= msf."date"
         AND msf."date" <= COALESCE(names.nameendt, CAST('9999-12-31' AS date))
        WHERE msf.permno = ANY(CAST(%(permnos)s AS integer[]))
          AND msf."date"::date BETWEEN CAST(%(start_date)s AS date) AND CAST(%(end_date)s AS date)
        ORDER BY msf.permno, msf."date"
    """

    crsp = db.raw_sql(
        crsp_sql,
        params={
            "permnos": permnos,
            "start_date": start_date.date(),
            "end_date": end_date.date(),
        },
    )

    if crsp.empty:
        raise RuntimeError(
            f"CRSP query returned zero rows from {crsp_table!r}. "
            "The table may be unavailable for this WRDS account."
        )
    return crsp


def clean_and_join_crsp(crsp: pd.DataFrame, membership_monthly: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's CRSP cleaning and membership join."""
    result = crsp.copy()
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce").astype("Int64")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["permno", "date"]).copy()

    if result.empty:
        raise RuntimeError("CRSP rows were returned, but none had valid permno/date values.")

    result["permno"] = result["permno"].astype(int)
    result["month_end"] = result["date"] + MonthEnd(0)

    numeric_cols = ["ret", "retx", "dlret", "prc", "shrout", "vol", "exchcd", "shrcd"]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["me"] = result["prc"].abs() * result["shrout"]
    both_missing = result["ret"].isna() & result["dlret"].isna()
    result["retadj"] = (1.0 + result["ret"].fillna(0.0)) * (
        1.0 + result["dlret"].fillna(0.0)
    ) - 1.0
    result.loc[both_missing, "retadj"] = np.nan

    member_keys = membership_monthly[["permno", "month_end"]].drop_duplicates()
    crsp_panel = (
        result.merge(
            member_keys, on=["permno", "month_end"], how="inner", validate="many_to_one"
        )
        .sort_values(["month_end", "permno"])
        .reset_index(drop=True)
    )

    if crsp_panel.empty:
        raise RuntimeError("The CRSP/member-month join produced zero rows.")

    return crsp_panel


def run(args: argparse.Namespace) -> None:
    """Run the extraction."""
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if start_date > end_date:
        raise ValueError("--start-date must be on or before --end-date.")

    outdir = Path(args.outdir)
    membership_file = (
        Path(args.membership_file)
        if args.membership_file is not None
        else outdir / DEFAULT_MEMBERSHIP_FILE
    )

    membership_monthly, permnos = load_membership(membership_file)
    LOGGER.info("Loaded %s member permnos from %s.", f"{len(permnos):,}", membership_file)

    db = connect_wrds(args.wrds_username, args.wrds_password)
    try:
        crsp = fetch_crsp(
            db=db,
            permnos=permnos,
            start_date=start_date,
            end_date=end_date,
            crsp_table=args.crsp_table,
            crsp_event_table=args.crsp_event_table,
            crsp_names_table=args.crsp_names_table,
        )
    finally:
        db.close()

    crsp_panel = clean_and_join_crsp(crsp, membership_monthly)

    crsp_outfile = outdir / "crsp_monthly_sp500_panel.parquet"
    crsp_panel.to_parquet(crsp_outfile, engine="pyarrow", index=False)

    LOGGER.info("saved: %s", crsp_outfile)
    LOGGER.info("rows: %s", f"{len(crsp_panel):,}")
    LOGGER.info("permnos: %s", f"{crsp_panel['permno'].nunique():,}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    try:
        run(args)
    except Exception:
        LOGGER.exception("WRDS CRSP panel extraction failed.")
        raise


if __name__ == "__main__":
    main()
