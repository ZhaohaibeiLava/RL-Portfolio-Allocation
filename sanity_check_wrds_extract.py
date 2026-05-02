#!/usr/bin/env python
"""
Sanity checks for WRDS extract outputs.

This script validates the raw data produced by 01_run_wrds_extracts.ipynb:
  - S&P 500 membership monthly panel
  - CRSP monthly S&P 500 stock panel
  - Fama-French monthly factors
  - SPY monthly benchmark data

It writes compact CSV/TXT/PNG QA artifacts under data/sanity_checks by default.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)

CRSP_MISSINGNESS_COLUMNS = [
    "ret",
    "retx",
    "dlret",
    "prc",
    "shrout",
    "vol",
    "me",
    "retadj",
]
FF3_MISSINGNESS_COLUMNS = ["mktrf", "smb", "hml", "rf"]
SPY_MISSINGNESS_COLUMNS = ["ret", "retx", "prc", "shrout", "vol", "me"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sanity checks for WRDS extract parquet outputs."
    )
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing WRDS raw extract parquet files.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default="data/benchmark",
        help="Directory containing benchmark extract parquet files.",
    )
    parser.add_argument(
        "--outdir",
        default="data/sanity_checks",
        help="Output directory for sanity-check artifacts.",
    )
    parser.add_argument(
        "--membership-file",
        default="sp500_membership_monthly.parquet",
        help="S&P 500 membership parquet filename or path.",
    )
    parser.add_argument(
        "--crsp-file",
        default="crsp_monthly_sp500_panel.parquet",
        help="CRSP monthly panel parquet filename or path.",
    )
    parser.add_argument(
        "--ff3-file",
        default="ff3_monthly.parquet",
        help="Fama-French factor parquet filename or path.",
    )
    parser.add_argument(
        "--spy-file",
        default="spy_monthly.parquet",
        help="SPY monthly benchmark parquet filename or path.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "portfolio_rl_mpl_cache")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def resolve_path(project_root: Path, base_dir: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if len(path.parts) > 1:
        return project_root / path
    return base_dir / path


def read_parquet_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    LOGGER.info("Loading %s", path)
    return pd.read_parquet(path)


def require_columns(
    df: pd.DataFrame, columns: Iterable[str], dataset_name: str
) -> None:
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def normalize_month_end(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    require_columns(df, ["month_end"], dataset_name)
    result = df.copy()
    result["month_end"] = pd.to_datetime(result["month_end"], errors="coerce")
    if result["month_end"].isna().any():
        bad_rows = int(result["month_end"].isna().sum())
        raise ValueError(f"{dataset_name} has {bad_rows:,} invalid month_end values.")
    result["month_end"] = result["month_end"].dt.to_period("M").dt.to_timestamp("M")
    return result


def format_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return pd.Timestamp(value).date().isoformat()


def date_min_max(df: pd.DataFrame) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if df.empty:
        return None, None
    return df["month_end"].min(), df["month_end"].max()


def count_by_month(
    df: pd.DataFrame,
    count_column_name: str,
    *,
    unique_permnos: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            {
                "month_end": pd.Series(dtype="datetime64[ns]"),
                count_column_name: pd.Series(dtype="int64"),
            }
        )
    if unique_permnos:
        require_columns(df, ["permno"], count_column_name)
        counts = df.groupby("month_end")["permno"].nunique()
    else:
        counts = df.groupby("month_end").size()
    return (
        counts.rename(count_column_name)
        .reset_index()
        .sort_values("month_end")
        .reset_index(drop=True)
    )


def count_summary_stats(counts: pd.Series) -> pd.Series:
    return counts.describe(percentiles=[0.25, 0.5, 0.75])


def format_count_stats(stats: pd.Series) -> str:
    if stats.empty:
        return "count=0"
    return (
        f"count={stats['count']:.0f}, mean={stats['mean']:.2f}, "
        f"std={stats['std']:.2f}, min={stats['min']:.2f}, "
        f"p25={stats['25%']:.2f}, p50={stats['50%']:.2f}, "
        f"p75={stats['75%']:.2f}, max={stats['max']:.2f}"
    )


def missingness_summary(
    df: pd.DataFrame, columns: list[str], dataset_name: str
) -> pd.DataFrame:
    require_columns(df, columns, dataset_name)
    total_rows = len(df)
    rows = []
    for col in columns:
        missing_count = int(df[col].isna().sum())
        missing_rate = missing_count / total_rows if total_rows else 0.0
        rows.append(
            {
                "dataset": dataset_name,
                "column": col,
                "missing_count": missing_count,
                "total_rows": total_rows,
                "missing_rate": missing_rate,
            }
        )
    return pd.DataFrame(rows)


def duplicate_key_summary(df: pd.DataFrame, keys: list[str]) -> tuple[bool, int, int]:
    require_columns(df, keys, "duplicate key check")
    duplicate_mask = df.duplicated(keys, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_keys = int(df.loc[duplicate_mask, keys].drop_duplicates().shape[0])
    return duplicate_rows > 0, duplicate_rows, duplicate_keys


def compare_monthly_counts(
    membership_counts: pd.DataFrame, crsp_counts: pd.DataFrame
) -> pd.DataFrame:
    comparison = crsp_counts.merge(membership_counts, on="month_end", how="outer")
    comparison = comparison.sort_values("month_end").reset_index(drop=True)
    comparison["crsp_count"] = comparison["crsp_count"].fillna(0).astype(int)
    comparison["membership_count"] = (
        comparison["membership_count"].fillna(0).astype(int)
    )
    comparison["crsp_minus_membership"] = (
        comparison["crsp_count"] - comparison["membership_count"]
    )
    return comparison


def factor_coverage_summary(
    crsp: pd.DataFrame, ff3: pd.DataFrame
) -> tuple[int, list[pd.Timestamp]]:
    crsp_months = crsp[["month_end"]].dropna().drop_duplicates()
    ff_months = ff3[["month_end"]].dropna().drop_duplicates()
    factor_months = ff_months.assign(has_factor_data=True)
    coverage = crsp_months.merge(factor_months, on="month_end", how="left")
    missing_factor_months = sorted(
        coverage.loc[coverage["has_factor_data"].isna(), "month_end"]
    )
    matched_months = int(coverage["has_factor_data"].fillna(False).sum())
    return matched_months, missing_factor_months


def build_dataset_summary(
    membership: pd.DataFrame, crsp: pd.DataFrame, ff3: pd.DataFrame, spy: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for dataset_name, df, unique_col in [
        ("membership", membership, "permno"),
        ("crsp", crsp, "permno"),
        ("ff3", ff3, None),
        ("spy", spy, "permno" if "permno" in spy.columns else None),
    ]:
        month_start, month_end = date_min_max(df)
        rows.append(
            {
                "dataset": dataset_name,
                "rows": len(df),
                "unique_permnos": df[unique_col].nunique() if unique_col else pd.NA,
                "month_start": month_start,
                "month_end": month_end,
            }
        )
    return pd.DataFrame(rows)


def build_coverage_summary_text(
    membership: pd.DataFrame,
    crsp: pd.DataFrame,
    ff3: pd.DataFrame,
    spy: pd.DataFrame,
    membership_counts: pd.DataFrame,
    count_comparison: pd.DataFrame,
    crsp_has_duplicates: bool,
    crsp_duplicate_rows: int,
    crsp_duplicate_keys: int,
    ff3_has_duplicate_months: bool,
    ff3_duplicate_rows: int,
    ff3_duplicate_months: int,
    spy_has_duplicate_months: bool,
    spy_duplicate_rows: int,
    spy_duplicate_months: int,
    matched_factor_months: int,
    missing_factor_months: list[pd.Timestamp],
) -> str:
    membership_min, membership_max = date_min_max(membership)
    crsp_min, crsp_max = date_min_max(crsp)
    ff3_min, ff3_max = date_min_max(ff3)
    spy_min, spy_max = date_min_max(spy)
    membership_stats = count_summary_stats(membership_counts["membership_count"])
    crsp_stats = count_summary_stats(count_comparison["crsp_count"])
    diff_stats = count_summary_stats(count_comparison["crsp_minus_membership"])

    lines = [
        "WRDS extract sanity check",
        "",
        "Membership",
        f"- month_end range: {format_date(membership_min)} to {format_date(membership_max)}",
        f"- rows: {len(membership):,}",
        f"- unique permnos: {membership['permno'].nunique():,}",
        f"- monthly constituent count stats: {format_count_stats(membership_stats)}",
        "",
        "CRSP panel",
        f"- month_end range: {format_date(crsp_min)} to {format_date(crsp_max)}",
        f"- rows: {len(crsp):,}",
        f"- unique permnos: {crsp['permno'].nunique():,}",
        f"- duplicate (permno, month_end) keys: {'YES' if crsp_has_duplicates else 'NO'}",
        f"- duplicate rows involved: {crsp_duplicate_rows:,}",
        f"- duplicate key count: {crsp_duplicate_keys:,}",
        f"- monthly CRSP count stats: {format_count_stats(crsp_stats)}",
        f"- CRSP minus membership count stats: {format_count_stats(diff_stats)}",
        "",
        "Fama-French factors",
        f"- month_end range: {format_date(ff3_min)} to {format_date(ff3_max)}",
        f"- rows: {len(ff3):,}",
        f"- unique month_end: {'NO' if ff3_has_duplicate_months else 'YES'}",
        f"- duplicate month rows involved: {ff3_duplicate_rows:,}",
        f"- duplicate month count: {ff3_duplicate_months:,}",
        "",
        "SPY benchmark",
        f"- month_end range: {format_date(spy_min)} to {format_date(spy_max)}",
        f"- rows: {len(spy):,}",
        f"- unique month_end: {'NO' if spy_has_duplicate_months else 'YES'}",
        f"- duplicate month rows involved: {spy_duplicate_rows:,}",
        f"- duplicate month count: {spy_duplicate_months:,}",
        "",
        "CRSP and FF3 merge coverage",
        f"- matched CRSP months with factor rows: {matched_factor_months:,}",
        f"- any CRSP month missing factor data: {'YES' if missing_factor_months else 'NO'}",
    ]

    if missing_factor_months:
        missing_str = ", ".join(format_date(month) for month in missing_factor_months)
        lines.append(f"- CRSP months missing factors: {missing_str}")

    return "\n".join(lines) + "\n"


def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("Skipping plot because matplotlib is not installed: %s", output_path)
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x_col], df[y_col], linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Month end")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved plot: %s", output_path)


def run_sanity_checks(
    membership_file: Path,
    crsp_file: Path,
    ff3_file: Path,
    spy_file: Path,
    outdir: Path,
    *,
    make_plots: bool = True,
) -> dict[str, pd.DataFrame | str]:
    outdir.mkdir(parents=True, exist_ok=True)

    membership = normalize_month_end(read_parquet_file(membership_file), "membership")
    crsp = normalize_month_end(read_parquet_file(crsp_file), "crsp")
    ff3 = normalize_month_end(read_parquet_file(ff3_file), "ff3")
    spy = normalize_month_end(read_parquet_file(spy_file), "spy")

    require_columns(membership, ["permno", "month_end"], "membership")
    require_columns(crsp, ["permno", "month_end"], "crsp")
    require_columns(ff3, ["month_end"], "ff3")
    require_columns(spy, ["month_end", "ret"], "spy")

    dataset_summary = build_dataset_summary(membership, crsp, ff3, spy)

    membership_counts = count_by_month(membership, "membership_count")
    crsp_counts = count_by_month(crsp, "crsp_count")
    count_comparison = compare_monthly_counts(membership_counts, crsp_counts)

    missingness_parts = [
        missingness_summary(crsp, CRSP_MISSINGNESS_COLUMNS, "crsp"),
        missingness_summary(ff3, FF3_MISSINGNESS_COLUMNS, "ff3"),
    ]
    spy_missing_cols = [col for col in SPY_MISSINGNESS_COLUMNS if col in spy.columns]
    if spy_missing_cols:
        missingness_parts.append(missingness_summary(spy, spy_missing_cols, "spy"))
    missingness = pd.concat(missingness_parts, ignore_index=True)

    crsp_has_duplicates, crsp_duplicate_rows, crsp_duplicate_keys = (
        duplicate_key_summary(crsp, ["permno", "month_end"])
    )
    ff3_has_duplicate_months, ff3_duplicate_rows, ff3_duplicate_months = (
        duplicate_key_summary(ff3, ["month_end"])
    )
    spy_has_duplicate_months, spy_duplicate_rows, spy_duplicate_months = (
        duplicate_key_summary(spy, ["month_end"])
    )

    duplicate_summary = pd.DataFrame(
        [
            {
                "dataset": "crsp",
                "key": "permno, month_end",
                "has_duplicates": crsp_has_duplicates,
                "duplicate_rows": crsp_duplicate_rows,
                "duplicate_keys": crsp_duplicate_keys,
            },
            {
                "dataset": "ff3",
                "key": "month_end",
                "has_duplicates": ff3_has_duplicate_months,
                "duplicate_rows": ff3_duplicate_rows,
                "duplicate_keys": ff3_duplicate_months,
            },
            {
                "dataset": "spy",
                "key": "month_end",
                "has_duplicates": spy_has_duplicate_months,
                "duplicate_rows": spy_duplicate_rows,
                "duplicate_keys": spy_duplicate_months,
            },
        ]
    )

    matched_factor_months, missing_factor_months = factor_coverage_summary(crsp, ff3)
    factor_coverage = pd.DataFrame(
        [
            {
                "crsp_distinct_months": crsp["month_end"].nunique(),
                "ff3_distinct_months": ff3["month_end"].nunique(),
                "matched_crsp_months_with_ff3": matched_factor_months,
                "missing_crsp_months_with_ff3": len(missing_factor_months),
            }
        ]
    )

    max_abs_count_gap = (
        int(count_comparison["crsp_minus_membership"].abs().max())
        if not count_comparison.empty
        else 0
    )
    final_checks = pd.DataFrame(
        [
            {
                "check": "CRSP duplicate (permno, month_end) keys",
                "value": crsp_duplicate_keys,
            },
            {"check": "FF3 duplicate month_end keys", "value": ff3_duplicate_months},
            {"check": "SPY duplicate month_end keys", "value": spy_duplicate_months},
            {
                "check": "CRSP months missing FF3 factor data",
                "value": len(missing_factor_months),
            },
            {
                "check": "Max absolute CRSP-minus-membership count gap",
                "value": max_abs_count_gap,
            },
        ]
    )

    coverage_summary_text = build_coverage_summary_text(
        membership=membership,
        crsp=crsp,
        ff3=ff3,
        spy=spy,
        membership_counts=membership_counts,
        count_comparison=count_comparison,
        crsp_has_duplicates=crsp_has_duplicates,
        crsp_duplicate_rows=crsp_duplicate_rows,
        crsp_duplicate_keys=crsp_duplicate_keys,
        ff3_has_duplicate_months=ff3_has_duplicate_months,
        ff3_duplicate_rows=ff3_duplicate_rows,
        ff3_duplicate_months=ff3_duplicate_months,
        spy_has_duplicate_months=spy_has_duplicate_months,
        spy_duplicate_rows=spy_duplicate_rows,
        spy_duplicate_months=spy_duplicate_months,
        matched_factor_months=matched_factor_months,
        missing_factor_months=missing_factor_months,
    )

    dataset_summary.to_csv(outdir / "extract_dataset_summary.csv", index=False)
    membership_counts.to_csv(outdir / "membership_monthly_counts.csv", index=False)
    count_comparison.to_csv(outdir / "crsp_monthly_counts.csv", index=False)
    missingness.to_csv(outdir / "missingness_summary.csv", index=False)
    duplicate_summary.to_csv(outdir / "duplicate_key_summary.csv", index=False)
    factor_coverage.to_csv(outdir / "factor_coverage_summary.csv", index=False)
    final_checks.to_csv(outdir / "final_checks.csv", index=False)
    (outdir / "coverage_summary.txt").write_text(
        coverage_summary_text, encoding="utf-8"
    )

    LOGGER.info("Saved sanity-check tables and report to: %s", outdir)

    if make_plots:
        save_line_plot(
            membership_counts,
            "month_end",
            "membership_count",
            "Monthly S&P 500 Constituent Count",
            "Constituent count",
            outdir / "membership_monthly_count.png",
        )
        save_line_plot(
            count_comparison,
            "month_end",
            "crsp_count",
            "Monthly CRSP S&P 500 Panel Count",
            "CRSP stock count",
            outdir / "crsp_monthly_count.png",
        )
        save_line_plot(
            count_comparison,
            "month_end",
            "crsp_minus_membership",
            "CRSP Count Minus Membership Count",
            "CRSP count - membership count",
            outdir / "crsp_minus_membership_count.png",
        )

    return {
        "dataset_summary": dataset_summary,
        "membership_counts": membership_counts,
        "count_comparison": count_comparison,
        "missingness": missingness,
        "duplicate_summary": duplicate_summary,
        "factor_coverage": factor_coverage,
        "final_checks": final_checks,
        "coverage_summary_text": coverage_summary_text,
    }


def main() -> None:
    setup_logging()
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    raw_dir = resolve_path(project_root, project_root, args.raw_dir)
    benchmark_dir = resolve_path(project_root, project_root, args.benchmark_dir)
    outdir = resolve_path(project_root, project_root, args.outdir)

    membership_file = resolve_path(project_root, raw_dir, args.membership_file)
    crsp_file = resolve_path(project_root, raw_dir, args.crsp_file)
    ff3_file = resolve_path(project_root, raw_dir, args.ff3_file)
    spy_file = resolve_path(project_root, benchmark_dir, args.spy_file)

    results = run_sanity_checks(
        membership_file=membership_file,
        crsp_file=crsp_file,
        ff3_file=ff3_file,
        spy_file=spy_file,
        outdir=outdir,
        make_plots=not args.skip_plots,
    )

    print(results["coverage_summary_text"])
    print("Final checks:")
    print(results["final_checks"].to_string(index=False))
    print(f"Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
