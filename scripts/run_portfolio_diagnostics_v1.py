#!/usr/bin/env python3
"""Generate portfolio-level diagnostics for one strategy."""

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


LOGGER = logging.getLogger("run_portfolio_diagnostics_v1")


def parse_month(value: str | None) -> pd.Timestamp | None:
    """Parse an optional YYYY-MM string into a month-end timestamp."""
    if value is None:
        return None
    try:
        return pd.Timestamp(value) + MonthEnd(0)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Expected month formatted as YYYY-MM, got {value!r}.") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate portfolio diagnostics for one strategy.")
    parser.add_argument(
        "--strategy-name",
        required=True,
        choices=("static_allocator", "rl_overlay_sac", "static_fixed_param", "static_fixed_param_fair"),
    )
    parser.add_argument("--backtest-file", required=True)
    parser.add_argument("--weights-file", required=True)
    parser.add_argument("--outdir", default="data/diagnostics/portfolio")
    parser.add_argument("--weight-threshold", type=float, default=1e-6)
    parser.add_argument("--start-month", type=parse_month, default=None)
    parser.add_argument("--end-month", type=parse_month, default=None)
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


def normalize_month_end(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Normalize month_end to calendar month-end."""
    result = df.copy()
    if "month_end" not in result.columns:
        raise ValueError(f"{dataset_name} must contain month_end.")
    result["month_end"] = pd.to_datetime(result["month_end"], errors="coerce") + MonthEnd(0)
    bad = int(result["month_end"].isna().sum())
    if bad:
        raise ValueError(f"{dataset_name} has {bad:,} invalid month_end values.")
    return result


def require_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> None:
    """Validate required columns."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")


def load_backtest(path: Path) -> pd.DataFrame:
    """Load and validate a strategy backtest file."""
    if not path.exists():
        raise FileNotFoundError(f"Backtest file not found: {path}")
    df = pd.read_csv(path)
    df = normalize_month_end(df, "backtest file")
    require_columns(df, ["month_end", "gross_return", "net_return"], "backtest file")
    for col in ["gross_return", "net_return", "cost", "turnover"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "cost" not in df or df["cost"].isna().all():
        df["cost"] = df["gross_return"] - df["net_return"]
    return df.sort_values("month_end").reset_index(drop=True)


def load_weights(path: Path) -> pd.DataFrame:
    """Load and validate a strategy weights file."""
    if not path.exists():
        raise FileNotFoundError(f"Weights file not found: {path}")
    df = pd.read_parquet(path)
    df = normalize_month_end(df, "weights file")
    require_columns(df, ["month_end", "permno", "weight"], "weights file")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["month_end", "permno", "weight"]).copy()
    return df.sort_values(["month_end", "permno"]).reset_index(drop=True)


def concentration_by_month(weights: pd.DataFrame, weight_threshold: float) -> pd.DataFrame:
    """Compute holdings and concentration diagnostics by month."""
    rows: list[dict[str, float | pd.Timestamp]] = []
    for month_end, group in weights.groupby("month_end", sort=True):
        w = group["weight"].astype(float).clip(lower=0.0).to_numpy()
        positive = w[w > weight_threshold]
        hhi = float(np.square(w).sum()) if len(w) else np.nan
        sorted_w = np.sort(w)[::-1]
        rows.append(
            {
                "month_end": month_end,
                "n_holdings": int(len(positive)),
                "hhi": hhi,
                "effective_n": float(1.0 / hhi) if hhi > 0 else np.nan,
                "max_weight": float(sorted_w[0]) if len(sorted_w) else np.nan,
                "top10_weight_share": float(sorted_w[:10].sum()) if len(sorted_w) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def filter_months(df: pd.DataFrame, start_month: pd.Timestamp | None, end_month: pd.Timestamp | None) -> pd.DataFrame:
    """Apply optional month range filters."""
    result = df.copy()
    if start_month is not None:
        result = result.loc[result["month_end"] >= start_month]
    if end_month is not None:
        result = result.loc[result["month_end"] <= end_month]
    return result.reset_index(drop=True)


def add_nav_and_drawdown(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative NAV and drawdown diagnostics from net returns."""
    result = monthly.copy()
    result["cost_drag"] = result["gross_return"] - result["net_return"]
    result["cumulative_nav"] = (1.0 + result["net_return"].fillna(0.0)).cumprod()
    running_peak = result["cumulative_nav"].cummax().clip(lower=1.0)
    result["drawdown"] = result["cumulative_nav"] / running_peak - 1.0
    return result


def find_top_drawdown_periods(monthly: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Identify distinct drawdown episodes with peak, trough, and recovery dates."""
    if monthly.empty:
        return pd.DataFrame(columns=["rank", "peak_month", "trough_month", "recovery_month", "drawdown"])

    ordered = monthly.sort_values("month_end").reset_index(drop=True)
    episodes: list[dict[str, object]] = []
    in_drawdown = False
    peak_month: pd.Timestamp | None = None
    trough_month: pd.Timestamp | None = None
    trough_depth = 0.0

    for row in ordered.itertuples(index=False):
        month = row.month_end
        depth = float(row.drawdown)
        if depth < 0 and not in_drawdown:
            in_drawdown = True
            prior = ordered.loc[ordered["month_end"] <= month]
            peak_month = prior.loc[prior["cumulative_nav"].idxmax(), "month_end"]
            trough_month = month
            trough_depth = depth
        elif depth < 0 and in_drawdown and depth < trough_depth:
            trough_month = month
            trough_depth = depth
        elif depth >= 0 and in_drawdown:
            episodes.append(
                {
                    "peak_month": peak_month,
                    "trough_month": trough_month,
                    "recovery_month": month,
                    "drawdown": trough_depth,
                }
            )
            in_drawdown = False

    if in_drawdown:
        episodes.append(
            {
                "peak_month": peak_month,
                "trough_month": trough_month,
                "recovery_month": pd.NaT,
                "drawdown": trough_depth,
            }
        )

    result = pd.DataFrame(episodes)
    if result.empty:
        return pd.DataFrame(columns=["rank", "peak_month", "trough_month", "recovery_month", "drawdown"])
    result = result.sort_values("drawdown").head(top_n).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def summarize(monthly: pd.DataFrame, strategy_name: str, top_drawdowns: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute scalar portfolio diagnostics."""
    net = monthly["net_return"].dropna()
    gross = monthly["gross_return"].dropna()
    return {
        "strategy": strategy_name,
        "n_months": int(len(monthly)),
        "mean_gross_return": float(gross.mean()) if len(gross) else np.nan,
        "mean_net_return": float(net.mean()) if len(net) else np.nan,
        "mean_cost_drag": float(monthly["cost_drag"].mean()),
        "mean_turnover": float(monthly["turnover"].mean()),
        "mean_n_holdings": float(monthly["n_holdings"].mean()),
        "mean_hhi": float(monthly["hhi"].mean()),
        "mean_effective_n": float(monthly["effective_n"].mean()),
        "mean_max_weight": float(monthly["max_weight"].mean()),
        "mean_top10_weight_share": float(monthly["top10_weight_share"].mean()),
        "max_drawdown": float(monthly["drawdown"].min()),
        "ending_nav": float(monthly["cumulative_nav"].iloc[-1]) if len(monthly) else np.nan,
        "deepest_drawdown_peak": str(top_drawdowns["peak_month"].iloc[0].date()) if not top_drawdowns.empty else "n/a",
        "deepest_drawdown_trough": str(top_drawdowns["trough_month"].iloc[0].date()) if not top_drawdowns.empty else "n/a",
        "deepest_drawdown_recovery": str(top_drawdowns["recovery_month"].iloc[0].date()) if not top_drawdowns.empty and pd.notna(top_drawdowns["recovery_month"].iloc[0]) else "unrecovered",
    }


def write_summary_text(path: Path, summary: dict[str, float | int | str], top_drawdowns: pd.DataFrame) -> None:
    """Write a readable summary including top drawdown periods."""
    lines = [
        f"{summary['strategy']} Portfolio Diagnostic Summary",
        "",
        f"n_months: {summary['n_months']}",
        f"Mean gross return: {summary['mean_gross_return']:.6f}",
        f"Mean net return: {summary['mean_net_return']:.6f}",
        f"Mean cost drag: {summary['mean_cost_drag']:.6f}",
        f"Mean turnover: {summary['mean_turnover']:.6f}",
        f"Mean n_holdings: {summary['mean_n_holdings']:.2f}",
        f"Mean effective_n: {summary['mean_effective_n']:.2f}",
        f"Mean max weight: {summary['mean_max_weight']:.6f}",
        f"Mean top-10 weight share: {summary['mean_top10_weight_share']:.6f}",
        f"Max drawdown: {summary['max_drawdown']:.6f}",
        f"Ending NAV: {summary['ending_nav']:.6f}",
        "",
        "Top drawdown troughs:",
    ]
    if top_drawdowns.empty:
        lines.append("n/a")
    else:
        for row in top_drawdowns.itertuples(index=False):
            recovery = row.recovery_month.date() if pd.notna(row.recovery_month) else "unrecovered"
            lines.append(
                f"{row.rank}. peak={row.peak_month.date()} trough={row.trough_month.date()} recovery={recovery} drawdown={row.drawdown:.6f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_series(monthly: pd.DataFrame, columns: list[str], title: str, ylabel: str, path: Path) -> None:
    """Save a simple line plot."""
    if monthly.empty:
        LOGGER.warning("Skipping plot %s because diagnostics are empty.", path.name)
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(monthly["month_end"])
    for col in columns:
        if col in monthly.columns:
            ax.plot(x, monthly[col], linewidth=1.4, label=col)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(columns) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run diagnostics and write deterministic artifacts."""
    if args.weight_threshold < 0:
        raise ValueError("--weight-threshold must be nonnegative.")
    if args.start_month is not None and args.end_month is not None and args.start_month > args.end_month:
        raise ValueError("--start-month must be less than or equal to --end-month.")
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    backtest = filter_months(load_backtest(Path(args.backtest_file).expanduser()), args.start_month, args.end_month)
    weights = filter_months(load_weights(Path(args.weights_file).expanduser()), args.start_month, args.end_month)
    if backtest.empty:
        raise RuntimeError("No backtest rows remain after applying the requested month filter.")
    if weights.empty:
        raise RuntimeError("No weight rows remain after applying the requested month filter.")
    concentration = concentration_by_month(weights, args.weight_threshold)
    monthly = backtest.merge(concentration, on="month_end", how="left")
    monthly = add_nav_and_drawdown(monthly)
    monthly = monthly[
        [
            "month_end",
            "gross_return",
            "net_return",
            "cost",
            "cost_drag",
            "turnover",
            "n_holdings",
            "hhi",
            "effective_n",
            "max_weight",
            "top10_weight_share",
            "cumulative_nav",
            "drawdown",
        ]
    ]
    top_drawdowns = find_top_drawdown_periods(monthly)
    summary = summarize(monthly, args.strategy_name, top_drawdowns)

    prefix = args.strategy_name
    monthly.to_csv(outdir / f"{prefix}_portfolio_diag_monthly.csv", index=False)
    pd.DataFrame([summary]).to_csv(outdir / f"{prefix}_portfolio_diag_summary.csv", index=False)
    write_summary_text(outdir / f"{prefix}_portfolio_diag_summary.txt", summary, top_drawdowns)

    plot_series(monthly, ["gross_return", "net_return"], f"{prefix}: Gross vs Net Return", "Monthly return", outdir / f"{prefix}_gross_vs_net.png")
    plot_series(monthly, ["cost_drag"], f"{prefix}: Cost Drag", "Gross minus net return", outdir / f"{prefix}_cost_drag.png")
    plot_series(monthly, ["n_holdings"], f"{prefix}: Number of Holdings", "Holdings", outdir / f"{prefix}_n_holdings.png")
    plot_series(monthly, ["effective_n"], f"{prefix}: Effective Number of Names", "1 / HHI", outdir / f"{prefix}_effective_names.png")
    plot_series(monthly, ["cumulative_nav"], f"{prefix}: Cumulative NAV", "NAV", outdir / f"{prefix}_cumret.png")
    plot_series(monthly, ["drawdown"], f"{prefix}: Drawdown", "Drawdown", outdir / f"{prefix}_drawdown.png")

    LOGGER.info("Wrote portfolio diagnostics for %s to %s.", args.strategy_name, outdir)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
