#!/usr/bin/env python3
"""Generate prediction-layer diagnostics by train, validation, and test split."""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("run_signal_diagnostics_v1")
SPLIT_ORDER = ("train", "validation", "test")
SUMMARY_SPLIT_ORDER = ("test", "validation", "train")
TEST_RELATIVE_METRICS = (
    "mean_ic",
    "mean_rank_ic",
    "mean_top_bottom_spread",
    "annualized_top_bottom_spread",
    "spread_sharpe",
    "average_mu_std",
    "average_p90_minus_p10",
)


@dataclass(frozen=True)
class SplitWindow:
    """Calendar month-end window for one diagnostic split."""

    name: str
    start: pd.Timestamp
    end: pd.Timestamp


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
    parser = argparse.ArgumentParser(description="Generate split-aware monthly signal diagnostics.")
    parser.add_argument("--pred-file", default="data/prediction/fm_oos_predictions.parquet")
    parser.add_argument("--outdir", default="data/diagnostics/signal")
    parser.add_argument("--train-start", type=parse_month, default=parse_month("2006-01"))
    parser.add_argument("--train-end", type=parse_month, default=parse_month("2016-12"))
    parser.add_argument("--val-start", type=parse_month, default=parse_month("2017-01"))
    parser.add_argument("--val-end", type=parse_month, default=parse_month("2019-12"))
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


def validate_split_windows(splits: list[SplitWindow]) -> None:
    """Validate split windows are ordered and non-overlapping."""
    for split in splits:
        if split.start > split.end:
            raise ValueError(f"{split.name} start must be less than or equal to end.")
    for previous, current in zip(splits[:-1], splits[1:]):
        if previous.end >= current.start:
            raise ValueError(
                f"Split windows overlap or are out of order: {previous.name} ends "
                f"{previous.end.date()}, {current.name} starts {current.start.date()}."
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


def assign_split(df: pd.DataFrame, splits: list[SplitWindow]) -> pd.DataFrame:
    """Assign train/validation/test labels based on month_end."""
    result = df.copy()
    result["split"] = pd.NA
    for split in splits:
        mask = (result["month_end"] >= split.start) & (result["month_end"] <= split.end)
        result.loc[mask, "split"] = split.name
    return result.dropna(subset=["split"]).copy()


def compute_monthly_diagnostics(
    predictions: pd.DataFrame,
    target_col: str,
    splits: list[SplitWindow],
    min_cross_section: int,
) -> pd.DataFrame:
    """Compute monthly signal diagnostics with split labels."""
    df = assign_split(predictions, splits)
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for (split_name, month_end), group in df.groupby(["split", "month_end"], sort=True):
        group = group[["mu_hat", target_col]].dropna()
        if len(group) < min_cross_section:
            LOGGER.warning(
                "Skipping %s %s: only %s observations.",
                split_name,
                month_end.date(),
                len(group),
            )
            continue
        ic = float(group["mu_hat"].corr(group[target_col], method="pearson"))
        rank_ic = float(group["mu_hat"].corr(group[target_col], method="spearman"))
        top_return, bottom_return, spread = top_bottom_spread(group, target_col)
        mu = group["mu_hat"].astype(float)
        rows.append(
            {
                "split": str(split_name),
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
        raise RuntimeError("No monthly signal diagnostics were produced for the requested split windows.")
    monthly["split"] = pd.Categorical(monthly["split"], categories=SPLIT_ORDER, ordered=True)
    return monthly.sort_values(["split", "month_end"]).reset_index(drop=True)


def summarize_one_split(monthly: pd.DataFrame, split_name: str, split_window: SplitWindow) -> dict[str, float | int | str]:
    """Compute scalar signal diagnostics for one split."""
    spread = monthly["top_bottom_spread"].dropna()
    spread_vol = float(spread.std(ddof=1) * np.sqrt(12.0)) if len(spread) > 1 else np.nan
    spread_sharpe = (
        float(spread.mean() / spread.std(ddof=1) * np.sqrt(12.0))
        if len(spread) > 1 and spread.std(ddof=1) > 0
        else np.nan
    )
    return {
        "split": split_name,
        "start_month": split_window.start.date().isoformat(),
        "end_month": split_window.end.date().isoformat(),
        "n_months": int(len(monthly)),
        "mean_ic": float(monthly["ic"].mean()) if len(monthly) else np.nan,
        "mean_rank_ic": float(monthly["rank_ic"].mean()) if len(monthly) else np.nan,
        "mean_top_bottom_spread": float(spread.mean()) if len(spread) else np.nan,
        "annualized_top_bottom_spread": float(spread.mean() * 12.0) if len(spread) else np.nan,
        "annualized_spread_vol": spread_vol,
        "spread_sharpe": spread_sharpe,
        "average_mu_std": float(monthly["mu_std"].mean()) if len(monthly) else np.nan,
        "average_p90_minus_p10": float(monthly["mu_p90_minus_p10"].mean()) if len(monthly) else np.nan,
        "average_n_obs": float(monthly["n_obs"].mean()) if len(monthly) else np.nan,
    }


def summarize_by_split(monthly: pd.DataFrame, splits: list[SplitWindow]) -> pd.DataFrame:
    """Compute scalar diagnostics separately for train/validation/test."""
    rows = []
    for split in splits:
        split_monthly = monthly.loc[monthly["split"].astype(str) == split.name].copy()
        rows.append(summarize_one_split(split_monthly, split.name, split))
    summary = pd.DataFrame(rows)
    test_row = summary.loc[summary["split"] == "test"]
    if not test_row.empty:
        test_values = test_row.iloc[0]
        for metric in TEST_RELATIVE_METRICS:
            summary[f"{metric}_minus_test"] = summary[metric] - test_values[metric]
    else:
        for metric in TEST_RELATIVE_METRICS:
            summary[f"{metric}_minus_test"] = np.nan
    summary["split"] = pd.Categorical(summary["split"], categories=SUMMARY_SPLIT_ORDER, ordered=True)
    return summary.sort_values("split").reset_index(drop=True)


def write_summary_text(path: Path, summary: pd.DataFrame) -> None:
    """Write a readable split summary."""
    lines = ["Signal Diagnostic Split Summary", ""]
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"{row.split.upper()} ({row.start_month} to {row.end_month})",
                f"n_months: {row.n_months}",
                f"Mean IC: {row.mean_ic:.6f}",
                f"Mean rank IC: {row.mean_rank_ic:.6f}",
                f"Mean top-bottom spread: {row.mean_top_bottom_spread:.6f}",
                f"Annualized top-bottom spread: {row.annualized_top_bottom_spread:.6f}",
                f"Annualized spread vol: {row.annualized_spread_vol:.6f}",
                f"Spread Sharpe: {row.spread_sharpe:.6f}",
                f"Average mu_std: {row.average_mu_std:.6f}",
                f"Average p90_minus_p10: {row.average_p90_minus_p10:.6f}",
                f"Average n_obs: {row.average_n_obs:.2f}",
                f"Mean IC minus test: {row.mean_ic_minus_test:.6f}",
                f"Mean rank IC minus test: {row.mean_rank_ic_minus_test:.6f}",
                f"Mean top-bottom spread minus test: {row.mean_top_bottom_spread_minus_test:.6f}",
                f"Average mu_std minus test: {row.average_mu_std_minus_test:.6f}",
                f"Average p90_minus_p10 minus test: {row.average_p90_minus_p10_minus_test:.6f}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_by_split(monthly: pd.DataFrame, y_cols: list[str], title: str, ylabel: str, path: Path) -> None:
    """Save a split-colored time-series plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    for split_name in SPLIT_ORDER:
        subset = monthly.loc[monthly["split"].astype(str) == split_name].copy()
        if subset.empty:
            continue
        x = pd.to_datetime(subset["month_end"])
        for col in y_cols:
            label = split_name if len(y_cols) == 1 else f"{split_name}: {col}"
            ax.plot(x, subset[col], linewidth=1.4, label=label, color=colors.get(split_name))
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run prediction-layer diagnostics by split."""
    if args.min_cross_section <= 0:
        raise ValueError("--min-cross-section must be positive.")
    splits = [
        SplitWindow("train", args.train_start, args.train_end),
        SplitWindow("validation", args.val_start, args.val_end),
        SplitWindow("test", args.test_start, args.test_end),
    ]
    validate_split_windows(splits)

    pred_file = Path(args.pred_file).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(pred_file, args.target_col)
    monthly = compute_monthly_diagnostics(predictions, args.target_col, splits, args.min_cross_section)
    summary = summarize_by_split(monthly, splits)

    monthly.to_csv(outdir / "signal_monthly_diagnostics.csv", index=False)
    summary.to_csv(outdir / "signal_split_summary.csv", index=False)
    write_summary_text(outdir / "signal_split_summary.txt", summary)
    plot_by_split(monthly, ["ic"], "Monthly Pearson IC by Split", "IC", outdir / "signal_ic_timeseries_by_split.png")
    plot_by_split(monthly, ["rank_ic"], "Monthly Spearman Rank IC by Split", "Rank IC", outdir / "signal_rank_ic_timeseries_by_split.png")
    plot_by_split(monthly, ["top_bottom_spread"], "Monthly Top-Decile Minus Bottom-Decile Spread by Split", "Spread return", outdir / "signal_top_bottom_spread_by_split.png")
    plot_by_split(monthly, ["mu_p10", "mu_p50", "mu_p90"], "Monthly mu_hat Dispersion by Split", "mu_hat", outdir / "signal_mu_dispersion_by_split.png")

    LOGGER.info("Wrote split-aware signal diagnostics to %s.", outdir)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
