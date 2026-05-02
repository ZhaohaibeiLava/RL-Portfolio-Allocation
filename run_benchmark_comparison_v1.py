#!/usr/bin/env python
"""
Unified benchmark comparison for monthly portfolio-allocation strategies.

The script compares:
  - Equal-weight stock-level baseline
  - Static allocator baseline
  - RL overlay SAC
  - SPY buy-and-hold benchmark

All strategies are restricted to a common monthly test window and evaluated with
one reporting framework.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
TRADING_MONTHS_PER_YEAR = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified benchmark comparison for monthly allocation outputs."
    )
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument(
        "--returns-file",
        default="data/panel/monthly_stock_panel_basic_full.parquet",
        help="Monthly stock-level return panel.",
    )
    parser.add_argument(
        "--spy-file",
        default="data/benchmark/spy_monthly.parquet",
        help="SPY monthly benchmark parquet file.",
    )
    parser.add_argument(
        "--static-backtest-file",
        default="data/allocator/static_allocator_backtest.csv",
        help="Static allocator backtest CSV.",
    )
    parser.add_argument(
        "--rl-backtest-file",
        default="data/rl_overlay_sac/test_backtest.csv",
        help="RL overlay SAC backtest CSV.",
    )
    parser.add_argument(
        "--outdir",
        default="data/benchmark_eval",
        help="Output directory for benchmark evaluation artifacts.",
    )
    parser.add_argument("--test-start", default="2020-01", help="First test month.")
    parser.add_argument("--test-end", default="2025-12", help="Last test month.")
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Proportional transaction cost in basis points for equal-weight rebalancing.",
    )
    parser.add_argument(
        "--spy-entry-cost-bps",
        type=float,
        default=0.0,
        help="Optional one-time initial entry cost for SPY, in basis points.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(project_root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root / path


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} path is not a file: {path}")


def parse_month_end(value: str) -> pd.Timestamp:
    try:
        return pd.Period(value, freq="M").to_timestamp(how="end").normalize()
    except Exception as exc:
        raise ValueError(f"Could not parse month value '{value}'. Use YYYY-MM.") from exc


def to_month_end(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    if dates.isna().any():
        bad_count = int(dates.isna().sum())
        raise ValueError(f"Date column contains {bad_count} unparsable values.")
    return dates.dt.to_period("M").dt.to_timestamp("M")


def coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().all():
        raise ValueError(f"Column '{column_name}' has no numeric observations.")
    return out.astype(float)


def pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def load_strategy_backtest(path: Path, strategy_name: str) -> pd.DataFrame:
    require_file(path, strategy_name)
    LOGGER.info("Loading %s backtest: %s", strategy_name, path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{strategy_name} backtest is empty: {path}")

    date_col = pick_column(df.columns, ["month_end", "date", "month"])
    if date_col is None:
        raise ValueError(
            f"{strategy_name} backtest must contain one of: month_end, date, month."
        )

    return_col = pick_column(
        df.columns,
        ["net_return", "return_net", "portfolio_return_net", "ret_net"],
    )
    if return_col is None:
        gross_col = pick_column(df.columns, ["gross_return", "return", "ret"])
        cost_col = pick_column(df.columns, ["cost", "tcost", "transaction_cost"])
        if gross_col is not None and cost_col is not None:
            LOGGER.info(
                "%s has no net-return column; using %s - %s.",
                strategy_name,
                gross_col,
                cost_col,
            )
            returns = coerce_numeric(df[gross_col], gross_col) - coerce_numeric(
                df[cost_col], cost_col
            )
        elif gross_col is not None:
            LOGGER.warning(
                "%s has no net-return column; using %s as returns.",
                strategy_name,
                gross_col,
            )
            returns = coerce_numeric(df[gross_col], gross_col)
        else:
            raise ValueError(
                f"{strategy_name} backtest must contain a net_return column, or gross/return data."
            )
    else:
        returns = coerce_numeric(df[return_col], return_col)

    out = pd.DataFrame(
        {
            "month_end": to_month_end(df[date_col]),
            "return": returns,
        }
    )
    turnover_col = pick_column(df.columns, ["turnover", "monthly_turnover"])
    if turnover_col is not None:
        out["turnover"] = coerce_numeric(df[turnover_col], turnover_col)
    else:
        out["turnover"] = np.nan

    out = (
        out.dropna(subset=["month_end", "return"])
        .sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError(f"{strategy_name} has no valid monthly returns after parsing.")
    return out


def load_spy_returns(path: Path) -> pd.DataFrame:
    require_file(path, "SPY benchmark")
    LOGGER.info("Loading SPY benchmark: %s", path)
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"SPY benchmark file is empty: {path}")
    if "ret" not in df.columns:
        raise ValueError(f"SPY benchmark file must contain a 'ret' column: {path}")

    date_col = pick_column(df.columns, ["month_end", "date", "month"])
    if date_col is None:
        raise ValueError("SPY benchmark must contain one of: month_end, date, month.")

    out = pd.DataFrame(
        {
            "month_end": to_month_end(df[date_col]),
            "return": coerce_numeric(df["ret"], "ret"),
            "turnover": np.nan,
        }
    )
    out = (
        out.dropna(subset=["month_end", "return"])
        .sort_values("month_end")
        .drop_duplicates("month_end", keep="last")
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError("SPY benchmark has no valid monthly returns after parsing.")
    return out


def build_equal_weight_strategy(
    returns_file: Path,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    cost_bps: float,
) -> pd.DataFrame:
    require_file(returns_file, "Stock-level return panel")
    LOGGER.info("Building equal-weight baseline from: %s", returns_file)
    df = pd.read_parquet(returns_file)
    required = {"permno", "month_end", "target_ret_1m"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Stock panel is missing required columns for equal-weight baseline: {missing}"
        )

    panel = df[["permno", "month_end", "target_ret_1m"]].copy()
    if "in_sp500_t" in df.columns:
        panel["in_sp500_t"] = df["in_sp500_t"]
        panel = panel[panel["in_sp500_t"].eq(1)].drop(columns=["in_sp500_t"])
        LOGGER.info("Using in_sp500_t == 1 as the investable-universe filter.")
    else:
        LOGGER.warning(
            "Column in_sp500_t not found; using all panel rows as the investable universe."
        )

    panel["month_end"] = to_month_end(panel["month_end"])
    panel["target_ret_1m"] = coerce_numeric(panel["target_ret_1m"], "target_ret_1m")
    panel = panel.dropna(subset=["permno", "month_end", "target_ret_1m"])
    panel = panel.sort_values(["month_end", "permno"])

    # Need one formation month before the first realized test month.
    formation_start = (test_start - pd.offsets.MonthEnd(1)).normalize()
    formation_end = (test_end - pd.offsets.MonthEnd(1)).normalize()
    panel = panel[
        panel["month_end"].between(formation_start, formation_end, inclusive="both")
    ]
    if panel.empty:
        raise ValueError("No panel rows available to build equal-weight baseline.")

    cost_rate = cost_bps / 10_000.0
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    prev_post_return_weights: pd.Series | None = None

    for formation_month, group in panel.groupby("month_end", sort=True):
        returns = group.set_index("permno")["target_ret_1m"].astype(float)
        returns = returns[returns.notna()]
        n_assets = int(returns.shape[0])
        if n_assets == 0:
            continue

        target_weights = pd.Series(1.0 / n_assets, index=returns.index)
        if prev_post_return_weights is None:
            turnover = 1.0
        else:
            aligned_index = target_weights.index.union(prev_post_return_weights.index)
            turnover = float(
                (
                    target_weights.reindex(aligned_index, fill_value=0.0)
                    - prev_post_return_weights.reindex(aligned_index, fill_value=0.0)
                )
                .abs()
                .sum()
            )

        gross_return = float((target_weights * returns).sum())
        cost = cost_rate * turnover
        net_return = gross_return - cost
        realized_month = (formation_month + pd.offsets.MonthEnd(1)).normalize()

        ending_values = target_weights * (1.0 + returns)
        ending_total = float(ending_values.sum())
        if ending_total <= 0 or not np.isfinite(ending_total):
            raise ValueError(
                f"Equal-weight ending portfolio value is invalid for {formation_month.date()}."
            )
        prev_post_return_weights = ending_values / ending_total

        rows.append(
            {
                "month_end": realized_month,
                "return": net_return,
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "n_assets": n_assets,
            }
        )

    out = pd.DataFrame(rows)
    out = out[out["month_end"].between(test_start, test_end, inclusive="both")]
    if out.empty:
        raise ValueError("Equal-weight baseline has no observations in the test window.")
    return out.reset_index(drop=True)


def restrict_to_test_window(
    df: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> pd.DataFrame:
    return df[df["month_end"].between(test_start, test_end, inclusive="both")].copy()


def assemble_common_returns(
    strategies: dict[str, pd.DataFrame],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    spy_entry_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = {
        name: restrict_to_test_window(df, test_start, test_end)
        for name, df in strategies.items()
    }
    for name, df in filtered.items():
        if df.empty:
            raise ValueError(f"{name} has no observations in requested test window.")

    common_months = sorted(
        set.intersection(*(set(df["month_end"]) for df in filtered.values()))
    )
    if not common_months:
        raise ValueError("No common months across all benchmark strategies.")

    LOGGER.info(
        "Common test window has %d months: %s to %s.",
        len(common_months),
        common_months[0].strftime("%Y-%m"),
        common_months[-1].strftime("%Y-%m"),
    )

    returns = pd.DataFrame({"month_end": common_months})
    turnover = pd.DataFrame({"month_end": common_months})
    for name, df in filtered.items():
        indexed = df.set_index("month_end").reindex(common_months)
        returns[name] = indexed["return"].astype(float).to_numpy()
        turnover[name] = indexed["turnover"].astype(float).to_numpy()

    if spy_entry_cost_bps:
        entry_cost = spy_entry_cost_bps / 10_000.0
        returns.loc[0, "spy_buy_hold"] = returns.loc[0, "spy_buy_hold"] - entry_cost
        LOGGER.info("Applied one-time SPY entry cost of %.4f.", entry_cost)

    return returns, turnover


def max_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.astype(float)).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())


def compute_summary(returns: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    rows = []
    strategy_cols = [col for col in returns.columns if col != "month_end"]
    n_months = len(returns)
    for strategy in strategy_cols:
        r = returns[strategy].astype(float)
        cumulative_return = float((1.0 + r).prod() - 1.0)
        ann_return = float((1.0 + cumulative_return) ** (TRADING_MONTHS_PER_YEAR / n_months) - 1.0)
        ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_MONTHS_PER_YEAR)) if n_months > 1 else np.nan
        sharpe = ann_return / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol != 0 else np.nan
        t = turnover[strategy].astype(float) if strategy in turnover.columns else pd.Series(dtype=float)
        mean_turnover = float(t.mean()) if t.notna().any() else np.nan
        rows.append(
            {
                "strategy": strategy,
                "annualized_return": ann_return,
                "annualized_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown(r),
                "cumulative_return": cumulative_return,
                "mean_monthly_turnover": mean_turnover,
                "win_rate": float((r > 0).mean()),
                "n_months": n_months,
            }
        )
    return pd.DataFrame(rows)


def save_monthly_returns(
    returns: pd.DataFrame, turnover: pd.DataFrame, out_path: Path
) -> None:
    out = returns.copy()
    for col in turnover.columns:
        if col == "month_end":
            continue
        out[f"{col}_turnover"] = turnover[col]
    out["month_end"] = out["month_end"].dt.strftime("%Y-%m-%d")
    out.to_csv(out_path, index=False)
    LOGGER.info("Saved monthly benchmark returns: %s", out_path)


def write_summary_text(summary: pd.DataFrame, out_path: Path) -> None:
    display = summary.copy()
    pct_cols = [
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "cumulative_return",
        "mean_monthly_turnover",
        "win_rate",
    ]
    for col in pct_cols:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    display["sharpe_ratio"] = display["sharpe_ratio"].map(
        lambda x: "" if pd.isna(x) else f"{x:.3f}"
    )
    text = [
        "Benchmark Comparison Summary",
        "=" * 28,
        "",
        display.to_string(index=False),
        "",
    ]
    out_path.write_text("\n".join(text), encoding="utf-8")
    LOGGER.info("Saved benchmark summary text: %s", out_path)


def plot_cumulative_nav(returns: pd.DataFrame, out_path: Path) -> None:
    nav = returns.set_index("month_end").drop(columns=[], errors="ignore")
    nav = (1.0 + nav).cumprod()
    fig, ax = plt.subplots(figsize=(11, 6))
    for col in nav.columns:
        ax.plot(nav.index, nav[col], label=col)
    ax.set_title("Cumulative NAV Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved cumulative NAV plot: %s", out_path)


def plot_monthly_returns(returns: pd.DataFrame, out_path: Path) -> None:
    plot_df = returns.set_index("month_end")
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in plot_df.columns:
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title("Monthly Return Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved monthly return comparison plot: %s", out_path)


def plot_turnover(turnover: pd.DataFrame, out_path: Path) -> None:
    stock_level_cols = [
        col
        for col in ["equal_weight", "static_allocator", "rl_overlay_sac"]
        if col in turnover.columns and turnover[col].notna().any()
    ]
    if not stock_level_cols:
        LOGGER.warning("No stock-level turnover series available; skipping turnover plot.")
        return

    plot_df = turnover.set_index("month_end")[stock_level_cols]
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in plot_df.columns:
        ax.plot(plot_df.index, plot_df[col], label=col, linewidth=1.4)
    ax.set_title("Monthly Turnover Comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel("Turnover")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved turnover comparison plot: %s", out_path)


def save_plots(returns: pd.DataFrame, turnover: pd.DataFrame, outdir: Path) -> None:
    plot_cumulative_nav(returns, outdir / "cumulative_nav_comparison.png")
    plot_monthly_returns(returns, outdir / "monthly_return_comparison.png")
    plot_turnover(turnover, outdir / "turnover_comparison.png")


def main() -> None:
    setup_logging()
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    returns_file = resolve_path(project_root, args.returns_file)
    spy_file = resolve_path(project_root, args.spy_file)
    static_backtest_file = resolve_path(project_root, args.static_backtest_file)
    rl_backtest_file = resolve_path(project_root, args.rl_backtest_file)
    outdir = resolve_path(project_root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    test_start = parse_month_end(args.test_start)
    test_end = parse_month_end(args.test_end)
    if test_start > test_end:
        raise ValueError("--test-start must be earlier than or equal to --test-end.")

    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Requested test window: %s to %s", args.test_start, args.test_end)

    equal_weight = build_equal_weight_strategy(
        returns_file=returns_file,
        test_start=test_start,
        test_end=test_end,
        cost_bps=args.cost_bps,
    )
    static_allocator = load_strategy_backtest(
        static_backtest_file, strategy_name="static_allocator"
    )
    rl_overlay_sac = load_strategy_backtest(
        rl_backtest_file, strategy_name="rl_overlay_sac"
    )
    spy_buy_hold = load_spy_returns(spy_file)

    strategies = {
        "equal_weight": equal_weight,
        "static_allocator": static_allocator,
        "rl_overlay_sac": rl_overlay_sac,
        "spy_buy_hold": spy_buy_hold,
    }
    returns, turnover = assemble_common_returns(
        strategies=strategies,
        test_start=test_start,
        test_end=test_end,
        spy_entry_cost_bps=args.spy_entry_cost_bps,
    )
    summary = compute_summary(returns, turnover)

    monthly_returns_path = outdir / "benchmark_monthly_returns.csv"
    summary_csv_path = outdir / "benchmark_summary.csv"
    summary_txt_path = outdir / "benchmark_summary.txt"

    save_monthly_returns(returns, turnover, monthly_returns_path)
    summary.to_csv(summary_csv_path, index=False)
    LOGGER.info("Saved benchmark summary CSV: %s", summary_csv_path)
    write_summary_text(summary, summary_txt_path)
    save_plots(returns, turnover, outdir)

    LOGGER.info("Benchmark comparison complete.")


if __name__ == "__main__":
    main()
