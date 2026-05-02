#!/usr/bin/env python3
"""Generate prediction-layer diagnostics over a test period."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("run_signal_diagnostics_v1")


def parse_month(value: str | None) -> pd.Timestamp | None:
    """Parse YYYY-MM into a month-end timestamp."""
    if value is None:
        return None
    try:
        return pd.Timestamp(value) + MonthEnd(0)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Expected month formatted as YYYY-MM, got {value!r}.") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate monthly signal diagnostics.")
    parser.add_argument("--pred-file", default="data/prediction/fm_oos_predictions.parquet")
    parser.add_argument("--outdir", default="data/diagnostics/signal")
    parser.add_argument("--test-start", type=parse_month, default=parse_month("2020-01"))
    parser.add_argument("--test-end", type=parse_month, default=parse_month("2024-11"))
    parser.add_argument("--target-col", default="realized_target")
    parser.add_argument("--min-cross-section", type=int, default=50)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def normalize_month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize month_end to calendar month-end."""
    result = df.copy()
    if "month_end" not in result.columns:
        raise ValueError("Prediction file must contain month_end.")
    result["month_end"] = pd.to_datetime(result["month_end"], errors="coerce") + MonthEnd(0)
    bad = int(result["month_end"].isna().sum())
    if bad:
        raise ValueError(f"Prediction file has {bad:,} invalid month_end values.")
    return result


def validate_columns(df: pd.DataFrame, target_col: str) -> None:
    """Validate required prediction columns."""
    missing = sorted({"month_end", "mu_hat", target_col}.difference(df.columns))
    if missing:
        available = ", ".join(df.columns)
        raise ValueError(
            f"Prediction file is missing required columns: {', '.join(missing)}. "
            f"Available columns: {available}"
        )


def load_predictions(path: Path, target_col: str) -> pd.DataFrame:
    """Load and clean prediction data."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_parquet(path)
    validate_columns(df, target_col)
    df = normalize_month_end(df)
    df["mu_hat"] = pd.to_numeric(df["mu_hat"], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["month_end", "mu_hat", target_col]).copy()
    LOGGER.info("Kept %s of %s rows after cleaning.", f"{len(df):,}", f"{before:,}")
    return df.sort_values(["month_end", "mu_hat"]).reset_index(drop=True)


def top_bottom_spread(group: pd.DataFrame, target_col: str) -> tuple[float, float, float]:
    """Compute equal-weight top-decile minus bottom-decile realized return."""
    sorted_group = group.sort_values("mu_hat", kind="mergesort")
    n = len(sorted_group)
    decile_n = max(int(np.floor(n * 0.10)), 1)
    bottom = sorted_group.iloc[:decile_n][target_col].astype(float)
    top = sorted_group.iloc[-decile_n:][target_col].astype(float)
    top_return = float(top.mean())
    bottom_return = float(bottom.mean())
    return top_return, bottom_return, top_return - bottom_return


def compute_monthly_diagnostics(
    predictions: pd.DataFrame,
    target_col: str,
    test_start: pd.Timestamp | None,
    test_end: pd.Timestamp | None,
    min_cross_section: int,
) -> pd.DataFrame:
    """Compute monthly signal diagnostics."""
    df = predictions.copy()
    if test_start is not None:
        df = df.loc[df["month_end"] >= test_start]
    if test_end is not None:
        df = df.loc[df["month_end"] <= test_end]
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    for month_end, group in df.groupby("month_end", sort=True):
        group = group[["mu_hat", target_col]].dropna()
        if len(group) < min_cross_section:
            LOGGER.warning("Skipping %s: only %s observations.", month_end.date(), len(group))
            continue
        ic = float(group["mu_hat"].corr(group[target_col], method="pearson"))
        rank_ic = float(group["mu_hat"].corr(group[target_col], method="spearman"))
        top_return, bottom_return, spread = top_bottom_spread(group, target_col)
        mu = group["mu_hat"].astype(float)
        rows.append(
            {
                "month_end": month_end,
                "n_obs": int(len(group)),
                "ic": ic,
                "rank_ic": rank_ic,
                "top_decile_return": top_return,
                "bottom_decile_return": bottom_return,
                "top_bottom_spread": spread,
                "mu_mean": float(mu.mean()),
                "mu_std": float(mu.std(ddof=1)),
                "mu_p10": float(mu.quantile(0.10)),
                "mu_p50": float(mu.quantile(0.50)),
                "mu_p90": float(mu.quantile(0.90)),
                "mu_p90_minus_p10": float(mu.quantile(0.90) - mu.quantile(0.10)),
            }
        )
    monthly = pd.DataFrame(rows)
    if monthly.empty:
        raise RuntimeError("No monthly signal diagnostics were produced for the requested period.")
    return monthly.sort_values("month_end").reset_index(drop=True)


def summarize(monthly: pd.DataFrame) -> dict[str, float | int]:
    """Compute scalar signal diagnostics."""
    spread = monthly["top_bottom_spread"].dropna()
    spread_vol = float(spread.std(ddof=1) * np.sqrt(12.0)) if len(spread) > 1 else np.nan
    spread_sharpe = float(spread.mean() / spread.std(ddof=1) * np.sqrt(12.0)) if len(spread) > 1 and spread.std(ddof=1) > 0 else np.nan
    return {
        "n_months": int(len(monthly)),
        "mean_ic": float(monthly["ic"].mean()),
        "mean_rank_ic": float(monthly["rank_ic"].mean()),
        "mean_top_bottom_spread": float(spread.mean()) if len(spread) else np.nan,
        "annualized_top_bottom_spread": float(spread.mean() * 12.0) if len(spread) else np.nan,
        "annualized_spread_vol": spread_vol,
        "spread_sharpe": spread_sharpe,
        "average_mu_std": float(monthly["mu_std"].mean()),
        "average_p90_minus_p10": float(monthly["mu_p90_minus_p10"].mean()),
        "average_n_obs": float(monthly["n_obs"].mean()),
    }


def write_summary_text(path: Path, summary: dict[str, float | int]) -> None:
    """Write a readable summary."""
    lines = [
        "Signal Diagnostic Summary",
        "",
        f"n_months: {summary['n_months']}",
        f"Mean IC: {summary['mean_ic']:.6f}",
        f"Mean rank IC: {summary['mean_rank_ic']:.6f}",
        f"Mean top-bottom spread: {summary['mean_top_bottom_spread']:.6f}",
        f"Annualized top-bottom spread: {summary['annualized_top_bottom_spread']:.6f}",
        f"Annualized spread vol: {summary['annualized_spread_vol']:.6f}",
        f"Spread Sharpe: {summary['spread_sharpe']:.6f}",
        f"Average mu_std: {summary['average_mu_std']:.6f}",
        f"Average p90_minus_p10: {summary['average_p90_minus_p10']:.6f}",
        f"Average n_obs: {summary['average_n_obs']:.2f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_series(monthly: pd.DataFrame, y_cols: list[str], title: str, ylabel: str, path: Path) -> None:
    """Save a simple line plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(monthly["month_end"])
    for col in y_cols:
        ax.plot(x, monthly[col], linewidth=1.4, label=col)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(y_cols) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run prediction-layer diagnostics."""
    if args.min_cross_section <= 0:
        raise ValueError("--min-cross-section must be positive.")
    if args.test_start is not None and args.test_end is not None and args.test_start > args.test_end:
        raise ValueError("--test-start must be less than or equal to --test-end.")
    pred_file = Path(args.pred_file).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(pred_file, args.target_col)
    monthly = compute_monthly_diagnostics(
        predictions,
        args.target_col,
        args.test_start,
        args.test_end,
        args.min_cross_section,
    )
    summary = summarize(monthly)

    monthly.to_csv(outdir / "signal_monthly_diagnostics.csv", index=False)
    pd.DataFrame([summary]).to_csv(outdir / "signal_summary.csv", index=False)
    write_summary_text(outdir / "signal_summary.txt", summary)
    plot_series(monthly, ["ic"], "Monthly Pearson IC", "IC", outdir / "signal_ic_timeseries.png")
    plot_series(monthly, ["rank_ic"], "Monthly Spearman Rank IC", "Rank IC", outdir / "signal_rank_ic_timeseries.png")
    plot_series(monthly, ["top_bottom_spread"], "Monthly Top-Decile Minus Bottom-Decile Spread", "Spread return", outdir / "signal_top_decile_spread.png")
    plot_series(monthly, ["mu_p10", "mu_p50", "mu_p90"], "Monthly mu_hat Dispersion", "mu_hat", outdir / "signal_mu_dispersion.png")

    LOGGER.info("Wrote signal diagnostics to %s.", outdir)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
