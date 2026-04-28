#!/usr/bin/env python3
"""Build a CRSP-only monthly stock panel for S&P 500 allocation research.

The design is deliberately simple for version 1:
- Compute CRSP-only features and next-month targets on the full monthly CRSP
  history for all ever-in-sample permnos.
- Only after feature and target construction, keep rows where the stock is an
  S&P 500 member at month t.
- Do not require S&P 500 membership at t+1.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("build_basic_monthly_stock_panel")

PREFERRED_CRSP_FILE = "crsp_monthly_ever_sp500.parquet"
FALLBACK_CRSP_FILE = "crsp_monthly_sp500_panel.parquet"
DEFAULT_MEMBERSHIP_FILE = "sp500_membership_monthly.parquet"
DEFAULT_FF_FILE = "ff3_monthly.parquet"
PARQUET_COMPRESSION = "snappy"

MEMBERSHIP_REQUIRED_COLUMNS = ["permno", "month_end"]
FF_REQUIRED_COLUMNS = ["month_end", "mktrf", "smb", "hml", "rf"]
CRSP_REQUIRED_COLUMNS = [
    "permno",
    "date",
    "ret",
    "retx",
    "dlret",
    "prc",
    "shrout",
    "vol",
    "exchcd",
    "shrcd",
    "me",
    "retadj",
]
CRSP_NUMERIC_COLUMNS = [
    "permno",
    "ret",
    "retx",
    "dlret",
    "prc",
    "shrout",
    "vol",
    "exchcd",
    "shrcd",
    "me",
    "retadj",
]
CORE_MISSINGNESS_COLUMNS = [
    "target_ret_1m",
    "target_rf_1m",
    "target_excess_1m",
    "log_me",
    "rev_1m",
    "log_prc",
    "mom_12_2",
    "vol_12m",
    "retadj",
    "rf",
    "mktrf",
    "smb",
    "hml",
]
MODEL_REQUIRED_COLUMNS = [
    "target_ret_1m",
    "target_excess_1m",
    "log_me",
    "rev_1m",
    "mom_12_2",
    "vol_12m",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a CRSP-only basic monthly stock panel with S&P 500 membership "
            "applied after feature and target engineering."
        )
    )
    parser.add_argument("--indir", required=True, help="Input parquet folder.")
    parser.add_argument("--outdir", required=True, help="Output folder.")
    parser.add_argument(
        "--crsp-file",
        default=None,
        help=(
            "Optional CRSP parquet filename override. If omitted, the script uses "
            f"{PREFERRED_CRSP_FILE} when present, else {FALLBACK_CRSP_FILE}."
        ),
    )
    parser.add_argument(
        "--membership-file",
        default=DEFAULT_MEMBERSHIP_FILE,
        help=f"Membership parquet filename. Default: {DEFAULT_MEMBERSHIP_FILE}.",
    )
    parser.add_argument(
        "--ff-file",
        default=DEFAULT_FF_FILE,
        help=f"Fama-French monthly factor parquet filename. Default: {DEFAULT_FF_FILE}.",
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


def require_columns(df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> None:
    """Fail clearly if required columns are missing."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def read_parquet_file(path: Path, dataset_name: str) -> pd.DataFrame:
    """Read a parquet file with useful error context."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {path}")
    LOGGER.info("Loading %s from %s.", dataset_name, path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {dataset_name} parquet file {path}: {exc}") from exc


def choose_crsp_input(indir: Path, crsp_file: str | None) -> tuple[Path, str, bool]:
    """Choose the CRSP input file according to the requested priority order."""
    if crsp_file:
        path = indir / crsp_file
        if not path.exists():
            raise FileNotFoundError(f"--crsp-file was provided but does not exist: {path}")
        return path, crsp_file, False

    preferred_path = indir / PREFERRED_CRSP_FILE
    if preferred_path.exists():
        return preferred_path, PREFERRED_CRSP_FILE, False

    fallback_path = indir / FALLBACK_CRSP_FILE
    if not fallback_path.exists():
        raise FileNotFoundError(
            "No CRSP input found. Expected either "
            f"{preferred_path} or fallback {fallback_path}."
        )

    LOGGER.warning(
        "Preferred CRSP input %s was not found; using fallback %s. "
        "Next-month targets may be incomplete for stocks that exit the S&P 500 after month t.",
        preferred_path,
        fallback_path,
    )
    return fallback_path, FALLBACK_CRSP_FILE, True


def ensure_pyarrow_available() -> None:
    """Fail early if pyarrow is unavailable for parquet output."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "The pyarrow package is required for parquet output. Install it with "
            "`pip install pyarrow` before running this script."
        ) from exc
    LOGGER.info("Using pyarrow %s for parquet output.", pa.__version__)


def normalize_month_end(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Parse or create month_end as the calendar month end timestamp."""
    result = df.copy()
    if "month_end" not in result.columns:
        if "date" not in result.columns:
            raise ValueError(
                f"{dataset_name} must contain month_end or date to construct month_end."
            )
        LOGGER.info("%s has no month_end column; constructing it from date.", dataset_name)
        result["month_end"] = pd.to_datetime(result["date"], errors="coerce") + MonthEnd(0)
    else:
        result["month_end"] = pd.to_datetime(result["month_end"], errors="coerce") + MonthEnd(0)

    invalid = int(result["month_end"].isna().sum())
    if invalid:
        raise ValueError(f"{dataset_name} has {invalid:,} invalid month_end values.")
    return result


def clean_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """Create a unique month-t S&P 500 membership indicator."""
    require_columns(membership, MEMBERSHIP_REQUIRED_COLUMNS, "membership")
    result = normalize_month_end(membership, "membership")
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce")
    result = result.dropna(subset=["permno", "month_end"]).copy()
    result["permno"] = result["permno"].astype("int64")
    result = result[["permno", "month_end"]].drop_duplicates()
    result["in_sp500_t"] = 1
    return result.sort_values(["permno", "month_end"]).reset_index(drop=True)


def clean_ff_factors(ff: pd.DataFrame) -> pd.DataFrame:
    """Clean Fama-French monthly factors and keep the requested columns."""
    require_columns(ff, FF_REQUIRED_COLUMNS, "ff3")
    result = normalize_month_end(ff, "ff3")
    for col in ["mktrf", "smb", "hml", "rf"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    duplicate_months = int(result.duplicated(["month_end"]).sum())
    if duplicate_months:
        raise ValueError(f"ff3 has {duplicate_months:,} duplicate month_end rows.")

    return (
        result[["month_end", "mktrf", "smb", "hml", "rf"]]
        .sort_values("month_end")
        .reset_index(drop=True)
    )


def clean_crsp(crsp: pd.DataFrame) -> pd.DataFrame:
    """Clean CRSP monthly rows before any membership filtering."""
    require_columns(crsp, CRSP_REQUIRED_COLUMNS, "crsp")
    result = normalize_month_end(crsp, "crsp")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")

    for col in CRSP_NUMERIC_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    result = result.dropna(subset=["permno", "month_end"]).copy()
    result["permno"] = result["permno"].astype("int64")

    exact_duplicates = int(result.duplicated(keep="first").sum())
    if exact_duplicates:
        LOGGER.warning(
            "CRSP has %s exact duplicate rows; dropping exact duplicates before panel construction.",
            f"{exact_duplicates:,}",
        )
        result = result.drop_duplicates(keep="first").copy()

    duplicate_key_mask = result.duplicated(["permno", "month_end"], keep=False)
    if duplicate_key_mask.any():
        duplicate_key_count = int(
            result.loc[duplicate_key_mask, ["permno", "month_end"]]
            .drop_duplicates()
            .shape[0]
        )
        examples = (
            result.loc[duplicate_key_mask, ["permno", "month_end", "date", "retadj", "prc", "me"]]
            .sort_values(["permno", "month_end", "date"])
            .head(10)
            .to_string(index=False)
        )
        raise ValueError(
            "CRSP still has conflicting duplicate (permno, month_end) keys after "
            f"dropping exact duplicates. Duplicate key count: {duplicate_key_count:,}. "
            "Inspect and resolve these rows before building a one-row-per-stock-month panel.\n"
            f"Examples:\n{examples}"
        )

    result = result.sort_values(["permno", "month_end"]).reset_index(drop=True)
    result["abs_prc"] = result["prc"].abs()

    valid_price = result["abs_prc"].gt(0).fillna(False)
    result["log_prc"] = np.nan
    result.loc[valid_price, "log_prc"] = np.log(
        result.loc[valid_price, "abs_prc"].astype("float64")
    )

    valid_me = result["me"].gt(0).fillna(False)
    result["log_me"] = np.nan
    result.loc[valid_me, "log_me"] = np.log(
        result.loc[valid_me, "me"].astype("float64")
    )
    return result


def merge_factors(crsp: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    """Merge Fama-French factors onto CRSP rows by month_end."""
    merged = crsp.merge(ff, on="month_end", how="left", validate="many_to_one")
    missing_factor_rows = int(merged["rf"].isna().sum())
    if missing_factor_rows:
        LOGGER.warning(
            "%s CRSP rows have no matching Fama-French rf value.",
            f"{missing_factor_rows:,}",
        )
    return merged.sort_values(["permno", "month_end"]).reset_index(drop=True)


def add_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Add next-month return, risk-free rate, and excess-return targets."""
    result = panel.copy()
    grouped = result.groupby("permno", sort=False)
    next_month = grouped["month_end"].shift(-1)
    expected_next_month = result["month_end"] + MonthEnd(1)
    consecutive_next_month = next_month.eq(expected_next_month)

    result["target_ret_1m"] = grouped["retadj"].shift(-1)
    result["target_rf_1m"] = grouped["rf"].shift(-1)
    result.loc[~consecutive_next_month, ["target_ret_1m", "target_rf_1m"]] = np.nan
    result["target_excess_1m"] = result["target_ret_1m"] - result["target_rf_1m"]
    return result


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add simple CRSP-only month-t features."""
    result = panel.copy()
    grouped = result.groupby("permno", sort=False)

    result["rev_1m"] = result["retadj"]

    lagged_ret = grouped["retadj"].shift(1)
    one_plus_lagged_ret = 1.0 + lagged_ret
    result["mom_12_2"] = (
        one_plus_lagged_ret.groupby(result["permno"], sort=False)
        .rolling(window=11, min_periods=11)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )

    result["vol_12m"] = (
        grouped["retadj"]
        .rolling(window=12, min_periods=12)
        .std()
        .reset_index(level=0, drop=True)
    )
    return result


def apply_membership_filter(panel: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Merge month-t membership and keep only in-sample member-month rows."""
    merged = panel.merge(
        membership,
        on=["permno", "month_end"],
        how="left",
        validate="many_to_one",
    )
    merged["in_sp500_t"] = merged["in_sp500_t"].fillna(0).astype("int8")
    return (
        merged.loc[merged["in_sp500_t"].eq(1)]
        .sort_values(["month_end", "permno"])
        .reset_index(drop=True)
    )


def apply_basic_sample_filters(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply simple v1 sample filters after month-t membership filtering."""
    result = panel.dropna(subset=["permno", "month_end"]).copy()

    if "shrcd" in result.columns:
        before = len(result)
        result = result.loc[result["shrcd"].isin([10, 11])].copy()
        LOGGER.info("Common-share filter shrcd in {10, 11}: kept %s of %s rows.", len(result), before)

    if "exchcd" in result.columns:
        before = len(result)
        result = result.loc[result["exchcd"].isin([1, 2, 3])].copy()
        LOGGER.info("Exchange filter exchcd in {1, 2, 3}: kept %s of %s rows.", len(result), before)

    return result.sort_values(["month_end", "permno"]).reset_index(drop=True)


def build_model_panel(full_panel: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with complete model-required targets and features."""
    require_columns(full_panel, MODEL_REQUIRED_COLUMNS, "full_panel")
    return (
        full_panel.dropna(subset=MODEL_REQUIRED_COLUMNS)
        .sort_values(["month_end", "permno"])
        .reset_index(drop=True)
    )


def missingness_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute missing counts and rates for core columns."""
    available_columns = [col for col in columns if col in df.columns]
    total_rows = len(df)
    rows = []
    for col in available_columns:
        missing_count = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing_count,
                "total_rows": total_rows,
                "missing_rate": missing_count / total_rows if total_rows else np.nan,
            }
        )
    return pd.DataFrame(rows)


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row and unique-permno counts by month."""
    if df.empty:
        return pd.DataFrame(columns=["month_end", "row_count", "unique_permnos"])
    counts = (
        df.groupby("month_end")
        .agg(row_count=("permno", "size"), unique_permnos=("permno", "nunique"))
        .reset_index()
        .sort_values("month_end")
    )
    return counts


def format_date(value: object) -> str:
    """Format a pandas timestamp-like value for summary output."""
    if value is None or pd.isna(value):
        return "NA"
    return pd.Timestamp(value).date().isoformat()


def write_summary(
    outdir: Path,
    full_panel: pd.DataFrame,
    model_panel: pd.DataFrame,
    missingness: pd.DataFrame,
    counts: pd.DataFrame,
    crsp_filename: str,
    used_fallback: bool,
) -> None:
    """Write text, missingness, and monthly-count summary files."""
    date_min = full_panel["month_end"].min() if not full_panel.empty else None
    date_max = full_panel["month_end"].max() if not full_panel.empty else None

    lines = [
        "Basic monthly stock panel summary",
        "=================================",
        f"CRSP input file: {crsp_filename}",
        f"Used fallback CRSP input: {used_fallback}",
        f"Date range: {format_date(date_min)} to {format_date(date_max)}",
        f"Unique permnos, full panel: {full_panel['permno'].nunique():,}",
        f"Rows, full panel: {len(full_panel):,}",
        f"Rows, model panel: {len(model_panel):,}",
        "",
        "Core-column missingness rates, full panel:",
    ]
    for row in missingness.itertuples(index=False):
        lines.append(
            f"- {row.column}: {row.missing_count:,} / {row.total_rows:,} "
            f"({row.missing_rate:.2%})"
        )

    if counts.empty:
        lines.extend(["", "Monthly row counts: none"])
    else:
        lines.extend(
            [
                "",
                "Monthly row-count summary, full panel:",
                counts["row_count"].describe().to_string(),
                "",
                "Monthly unique-permno summary, full panel:",
                counts["unique_permnos"].describe().to_string(),
            ]
        )

    (outdir / "panel_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    missingness.to_csv(outdir / "panel_missingness.csv", index=False)
    counts.to_csv(outdir / "panel_monthly_counts.csv", index=False)


def save_outputs(
    outdir: Path,
    full_panel: pd.DataFrame,
    model_panel: pd.DataFrame,
    crsp_filename: str,
    used_fallback: bool,
) -> None:
    """Save parquet panel outputs and summary files."""
    outdir.mkdir(parents=True, exist_ok=True)
    full_path = outdir / "monthly_stock_panel_basic_full.parquet"
    model_path = outdir / "monthly_stock_panel_basic_model.parquet"

    LOGGER.info("Saving full panel to %s.", full_path)
    full_panel.to_parquet(full_path, index=False, engine="pyarrow", compression=PARQUET_COMPRESSION)

    LOGGER.info("Saving model panel to %s.", model_path)
    model_panel.to_parquet(model_path, index=False, engine="pyarrow", compression=PARQUET_COMPRESSION)

    missingness = missingness_summary(full_panel, CORE_MISSINGNESS_COLUMNS)
    counts = monthly_counts(full_panel)
    write_summary(outdir, full_panel, model_panel, missingness, counts, crsp_filename, used_fallback)


def build_panel(
    membership: pd.DataFrame,
    crsp: pd.DataFrame,
    ff: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build full and model-ready panels from raw inputs."""
    membership_clean = clean_membership(membership)
    crsp_clean = clean_crsp(crsp)
    ff_clean = clean_ff_factors(ff)

    panel = merge_factors(crsp_clean, ff_clean)
    panel = add_targets(panel)
    panel = add_features(panel)
    panel = apply_membership_filter(panel, membership_clean)
    full_panel = apply_basic_sample_filters(panel)
    model_panel = build_model_panel(full_panel)
    return full_panel, model_panel


def main() -> None:
    """Run the panel build from command-line arguments."""
    args = parse_args()
    setup_logging(args.log_level)
    ensure_pyarrow_available()

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not indir.exists():
        raise FileNotFoundError(f"Input directory not found: {indir}")

    crsp_path, crsp_filename, used_fallback = choose_crsp_input(indir, args.crsp_file)
    membership_path = indir / args.membership_file
    ff_path = indir / args.ff_file

    membership = read_parquet_file(membership_path, "membership")
    crsp = read_parquet_file(crsp_path, "crsp")
    ff = read_parquet_file(ff_path, "ff3")

    full_panel, model_panel = build_panel(membership, crsp, ff)
    save_outputs(outdir, full_panel, model_panel, crsp_filename, used_fallback)

    LOGGER.info(
        "Done. Full panel rows: %s; model panel rows: %s; output directory: %s.",
        f"{len(full_panel):,}",
        f"{len(model_panel):,}",
        outdir,
    )


if __name__ == "__main__":
    main()
