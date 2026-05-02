#!/usr/bin/env python3
"""Fetch a clean monthly SPY benchmark series from WRDS/CRSP.

The script identifies an ETF/security by ticker from CRSP names data, selects
the most plausible continuous CRSP permno over the requested date range, pulls
monthly returns from ``crsp.msf``, and writes a cleaned benchmark parquet plus
human-readable metadata.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import wrds
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("fetch_spy_from_wrds")

DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTDIR = "data/benchmark"
DEFAULT_TICKER = "SPY"
PARQUET_COMPRESSION = "snappy"

NAME_TABLE_CANDIDATES = [
    {
        "library": "crsp",
        "table": "msenames",
        "start_col": "namedt",
        "end_col": "nameendt",
    },
    {
        "library": "crsp",
        "table": "stocknames",
        "start_col": "namedt",
        "end_col": "nameenddt",
    },
    {
        "library": "crsp",
        "table": "stocknames",
        "start_col": "namedt",
        "end_col": "nameendt",
    },
]

MSF_OPTIONAL_COLUMNS = ["exchcd", "shrcd"]
MSF_REQUIRED_COLUMNS = ["permno", "date", "ret", "retx", "prc", "shrout", "vol"]
NUMERIC_COLUMNS = ["ret", "retx", "prc", "shrout", "vol", "exchcd", "shrcd"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch monthly SPY benchmark data from WRDS/CRSP."
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Start date, inclusive. Default: {DEFAULT_START_DATE}.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"End date, inclusive. Default: {DEFAULT_END_DATE}.",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help=f"Output directory. Default: {DEFAULT_OUTDIR}.",
    )
    parser.add_argument(
        "--wrds-username",
        default=None,
        help="Optional WRDS username. If omitted, WRDS uses its default login flow.",
    )
    parser.add_argument(
        "--wrds-password",
        default=None,
        help="Optional WRDS password. If omitted, WRDS uses its default login flow.",
    )
    parser.add_argument(
        "--ticker",
        default=DEFAULT_TICKER,
        help=f"Ticker to fetch. Default: {DEFAULT_TICKER}.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity. Default: INFO.",
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
    """Parse a CLI date argument with a clear error."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{arg_name} must be a valid date, got {value!r}.")
    return pd.Timestamp(parsed).normalize()


def ensure_pyarrow_available() -> None:
    """Fail early if pyarrow is unavailable for parquet output."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "The pyarrow package is required for parquet output in this project."
        ) from exc
    LOGGER.info("Using pyarrow %s for parquet output.", pa.__version__)


def connect_wrds(wrds_username: str | None, wrds_password: str | None) -> wrds.Connection:
    """Open a WRDS connection using the same pattern as the project notebooks."""
    LOGGER.info("Connecting to WRDS.")
    if wrds_username is not None and wrds_password is not None:
        return wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    if wrds_username is not None:
        return wrds.Connection(wrds_username=wrds_username)
    return wrds.Connection()


def get_table_columns(db: wrds.Connection, library: str, table: str) -> set[str]:
    """Return lower-case column names for a WRDS table, or an empty set if absent."""
    try:
        columns = db.raw_sql(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %(library)s
              AND table_name = %(table)s
            ORDER BY ordinal_position
            """,
            params={"library": library, "table": table},
        )
    except Exception as exc:
        LOGGER.debug("Could not inspect %s.%s columns: %s", library, table, exc)
        return set()

    if columns.empty or "column_name" not in columns.columns:
        return set()
    return {str(col).lower() for col in columns["column_name"].dropna()}


def choose_name_source(db: wrds.Connection) -> dict[str, str]:
    """Choose an available CRSP name/security source table."""
    required = {"permno", "ticker"}
    for candidate in NAME_TABLE_CANDIDATES:
        columns = get_table_columns(db, candidate["library"], candidate["table"])
        if not columns:
            continue
        needed = required | {candidate["start_col"], candidate["end_col"]}
        if needed.issubset(columns):
            selected = dict(candidate)
            selected["columns"] = columns
            LOGGER.info(
                "Using CRSP names table %s.%s.",
                selected["library"],
                selected["table"],
            )
            return selected

    raise RuntimeError(
        "Could not find a usable CRSP names table. Tried crsp.msenames and "
        "crsp.stocknames with standard date-validity columns."
    )


def identifier_select_list(columns: set[str], start_col: str, end_col: str) -> str:
    """Build a SELECT list containing standard and optional name-table fields."""
    optional_cols = ["comnam", "ncusip", "cusip", "exchcd", "shrcd", "shrcls"]
    pieces = [
        "permno::integer AS permno",
        "UPPER(TRIM(ticker)) AS ticker",
        f"{start_col}::date AS namedt",
        f"{end_col}::date AS nameendt",
    ]
    for col in optional_cols:
        if col in columns:
            pieces.append(f"{col} AS {col}")
    return ",\n            ".join(pieces)


def identify_security(
    db: wrds.Connection,
    ticker: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[int, pd.DataFrame, list[str], str]:
    """Identify the most plausible CRSP permno for a ticker."""
    source = choose_name_source(db)
    table_name = f"{source['library']}.{source['table']}"
    select_list = identifier_select_list(
        source["columns"], source["start_col"], source["end_col"]
    )

    names_sql = f"""
        SELECT DISTINCT
            {select_list}
        FROM {table_name}
        WHERE UPPER(TRIM(ticker)) = UPPER(TRIM(%(ticker)s))
          AND {source["start_col"]}::date <= CAST(%(end_date)s AS date)
          AND COALESCE({source["end_col"]}::date, DATE '9999-12-31')
              >= CAST(%(start_date)s AS date)
        ORDER BY permno, namedt, nameendt
    """
    names = db.raw_sql(
        names_sql,
        params={
            "ticker": ticker,
            "start_date": start_date.date(),
            "end_date": end_date.date(),
        },
    )
    if names.empty:
        raise RuntimeError(
            f"No CRSP name/security rows found for ticker {ticker!r} overlapping "
            f"{start_date.date()} to {end_date.date()} in {table_name}."
        )

    LOGGER.info("Found %d CRSP name rows for ticker %s.", len(names), ticker.upper())
    permnos = sorted(names["permno"].dropna().astype(int).unique().tolist())
    stats = get_candidate_history_stats(db, permnos, start_date, end_date)
    candidates = names.merge(stats, on="permno", how="left")
    candidates["request_start"] = start_date.date()
    candidates["request_end"] = end_date.date()
    candidates["name_source"] = table_name

    aggregations = {
        "ticker": ("ticker", "first"),
        "first_namedt": ("namedt", "min"),
        "last_nameendt": ("nameendt", "max"),
        "first_msf_date": ("first_msf_date", "min"),
        "last_msf_date": ("last_msf_date", "max"),
        "row_count": ("row_count", "max"),
    }
    if "exchcd" in candidates.columns:
        aggregations["exchcd"] = ("exchcd", "first")
    if "shrcd" in candidates.columns:
        aggregations["shrcd"] = ("shrcd", "first")

    collapsed = candidates.groupby("permno", as_index=False).agg(**aggregations).copy()
    collapsed["row_count"] = collapsed["row_count"].fillna(0).astype(int)
    collapsed["first_msf_date"] = pd.to_datetime(collapsed["first_msf_date"], errors="coerce")
    collapsed["last_msf_date"] = pd.to_datetime(collapsed["last_msf_date"], errors="coerce")
    collapsed["starts_by_request"] = collapsed["first_msf_date"].le(start_date + MonthEnd(0))
    collapsed["ends_by_request"] = collapsed["last_msf_date"].ge(end_date - MonthEnd(0))
    collapsed["score"] = (
        collapsed["row_count"]
        + collapsed["starts_by_request"].astype(int) * 10_000
        + collapsed["ends_by_request"].astype(int) * 10_000
    )
    collapsed = collapsed.sort_values(
        ["score", "row_count", "last_msf_date", "first_msf_date"],
        ascending=[False, False, False, True],
    )
    selected_permno = int(collapsed.iloc[0]["permno"])

    ambiguity_notes: list[str] = []
    if len(permnos) > 1:
        ambiguity_notes.append(
            "Multiple matching permnos found: "
            + ", ".join(str(permno) for permno in permnos)
            + f". Selected {selected_permno} using monthly-history coverage."
        )
        LOGGER.warning("%s", ambiguity_notes[-1])
        LOGGER.warning("Candidate summary:\n%s", collapsed.to_string(index=False))
    else:
        ambiguity_notes.append("No ambiguity: exactly one matching permno found.")

    return selected_permno, candidates, ambiguity_notes, table_name


def get_candidate_history_stats(
    db: wrds.Connection,
    permnos: list[int],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Summarize CRSP monthly history for candidate permnos."""
    stats_sql = """
        SELECT
            permno::integer AS permno,
            MIN(date)::date AS first_msf_date,
            MAX(date)::date AS last_msf_date,
            COUNT(*)::integer AS row_count
        FROM crsp.msf
        WHERE permno = ANY(%(permnos)s)
          AND date BETWEEN CAST(%(start_date)s AS date) AND CAST(%(end_date)s AS date)
        GROUP BY permno
        ORDER BY permno
    """
    return db.raw_sql(
        stats_sql,
        params={
            "permnos": permnos,
            "start_date": start_date.date(),
            "end_date": end_date.date(),
        },
    )


def msf_select_columns(db: wrds.Connection) -> list[str]:
    """Choose available CRSP monthly stock file columns."""
    columns = get_table_columns(db, "crsp", "msf")
    required = set(MSF_REQUIRED_COLUMNS)
    if not required.issubset(columns):
        missing = sorted(required.difference(columns))
        raise RuntimeError(f"crsp.msf is missing required columns: {', '.join(missing)}")
    return MSF_REQUIRED_COLUMNS + [col for col in MSF_OPTIONAL_COLUMNS if col in columns]


def fetch_monthly_msf(
    db: wrds.Connection,
    permno: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Pull monthly CRSP rows for the selected permno."""
    selected_cols = msf_select_columns(db)
    select_list = ",\n            ".join(
        ["permno::integer AS permno", "date::date AS date"]
        + [col for col in selected_cols if col not in {"permno", "date"}]
    )
    msf_sql = f"""
        SELECT
            {select_list}
        FROM crsp.msf
        WHERE permno = %(permno)s
          AND date BETWEEN CAST(%(start_date)s AS date) AND CAST(%(end_date)s AS date)
        ORDER BY date
    """
    LOGGER.info("Fetching crsp.msf rows for permno %s.", permno)
    msf = db.raw_sql(
        msf_sql,
        params={
            "permno": int(permno),
            "start_date": start_date.date(),
            "end_date": end_date.date(),
        },
    )
    if msf.empty:
        raise RuntimeError(
            f"crsp.msf returned zero rows for permno {permno} from "
            f"{start_date.date()} to {end_date.date()}."
        )
    LOGGER.info("Fetched %d monthly rows.", len(msf))
    return msf


def clean_monthly_data(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Clean CRSP monthly data for benchmark use."""
    result = raw.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    invalid_dates = int(result["date"].isna().sum())
    if invalid_dates:
        raise ValueError(f"CRSP monthly data contains {invalid_dates:,} invalid dates.")

    result["month_end"] = result["date"] + MonthEnd(0)
    result["ticker"] = ticker.upper()
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce").astype("int64")

    for col in NUMERIC_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    if {"prc", "shrout"}.issubset(result.columns):
        result["me"] = np.abs(result["prc"]) * result["shrout"]

    output_cols = [
        "ticker",
        "permno",
        "date",
        "month_end",
        "ret",
        "retx",
        "prc",
        "shrout",
        "vol",
        "me",
        "exchcd",
        "shrcd",
    ]
    output_cols = [col for col in output_cols if col in result.columns]
    result = result[output_cols].sort_values("month_end").reset_index(drop=True)

    duplicate_months = int(result["month_end"].duplicated().sum())
    if duplicate_months:
        raise ValueError(
            f"Cleaned benchmark has {duplicate_months:,} duplicate month_end values."
        )
    return result


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact missingness summary for metadata."""
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": df.isna().mean().mul(100.0),
        }
    )
    return summary.reset_index(names="column")


def write_metadata(
    path: Path,
    ticker: str,
    selected_permno: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cleaned: pd.DataFrame,
    candidates: pd.DataFrame,
    ambiguity_notes: Iterable[str],
    name_source: str,
) -> None:
    """Write human-readable extraction metadata."""
    missingness = missingness_summary(cleaned)
    candidate_cols = [
        col
        for col in [
            "permno",
            "ticker",
            "namedt",
            "nameendt",
            "comnam",
            "ncusip",
            "cusip",
            "exchcd",
            "shrcd",
            "first_msf_date",
            "last_msf_date",
            "row_count",
        ]
        if col in candidates.columns
    ]

    lines = [
        f"{ticker.upper()} Monthly Benchmark Metadata",
        "================================",
        f"Ticker: {ticker.upper()}",
        f"Selected permno: {selected_permno}",
        f"Requested date range: {start_date.date()} to {end_date.date()}",
        f"Observed date range: {cleaned['month_end'].min().date()} to {cleaned['month_end'].max().date()}",
        f"Row count: {len(cleaned):,}",
        f"CRSP name source: {name_source}",
        "",
        "Ambiguity / security matching notes:",
        *[f"- {note}" for note in ambiguity_notes],
        "",
        "Candidate rows:",
        candidates[candidate_cols].to_string(index=False),
        "",
        "Missingness summary:",
        missingness.to_string(index=False, formatters={"missing_pct": "{:.2f}".format}),
        "",
        "Return definition:",
        "- ret is kept as the main monthly benchmark total return proxy from CRSP.",
        "- retx is retained as the ex-dividend return where available.",
        "- me is computed as abs(prc) * shrout when both inputs are available.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote metadata to %s.", path)


def run(args: argparse.Namespace) -> None:
    """Run the extraction."""
    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if start_date > end_date:
        raise ValueError("--start-date must be on or before --end-date.")

    ticker = args.ticker.strip().upper()
    if not ticker:
        raise ValueError("--ticker must not be empty.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ensure_pyarrow_available()

    db = connect_wrds(args.wrds_username, args.wrds_password)
    try:
        selected_permno, candidates, ambiguity_notes, name_source = identify_security(
            db=db,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        raw = fetch_monthly_msf(
            db=db,
            permno=selected_permno,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        try:
            db.close()
        except Exception:
            LOGGER.debug("WRDS connection close failed.", exc_info=True)

    cleaned = clean_monthly_data(raw, ticker)

    raw_path = outdir / f"{ticker.lower()}_monthly_raw.parquet"
    cleaned_path = outdir / f"{ticker.lower()}_monthly.parquet"
    metadata_path = outdir / f"{ticker.lower()}_metadata.txt"

    raw.to_parquet(raw_path, index=False, compression=PARQUET_COMPRESSION)
    cleaned.to_parquet(cleaned_path, index=False, compression=PARQUET_COMPRESSION)
    LOGGER.info("Wrote raw monthly data to %s.", raw_path)
    LOGGER.info("Wrote clean monthly benchmark to %s.", cleaned_path)

    write_metadata(
        path=metadata_path,
        ticker=ticker,
        selected_permno=selected_permno,
        start_date=start_date,
        end_date=end_date,
        cleaned=cleaned,
        candidates=candidates,
        ambiguity_notes=ambiguity_notes,
        name_source=name_source,
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    try:
        run(args)
    except Exception:
        LOGGER.exception("SPY benchmark extraction failed.")
        raise


if __name__ == "__main__":
    main()
