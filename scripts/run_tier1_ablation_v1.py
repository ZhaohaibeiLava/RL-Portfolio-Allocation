#!/usr/bin/env python3
"""Run Tier-1 ablations for the portfolio-allocation RL overlay pipeline.

This script is intentionally focused on a small set of high-signal diagnostic
ablations. It reuses the same data alignment, optimizer, timing convention, and
transaction-cost accounting as the static fixed-parameter benchmark.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from run_static_fixed_parameter_benchmark import (
    PARQUET_COMPRESSION,
    SUCCESS_STATUSES,
    align_month_inputs,
    align_previous_weights,
    annualized_return,
    build_month_sequence,
    clean_predictions,
    clean_returns,
    compute_turnover,
    discover_covariance_files,
    evaluate_next_return,
    load_covariance_npz,
    max_drawdown,
    mean_action_parameters,
    read_action_history,
    read_parquet_file,
    read_risk_metadata,
    resolve_path,
    select_fixed_parameters,
    solve_allocator,
)


LOGGER = logging.getLogger("run_tier1_ablation_v1")
ABLATION_TYPES = (
    "fair_fixed_param",
    "lambda_only",
    "tau_only",
    "state_ablation",
    "cost_robustness",
)
FIXED_PARAM_SOURCES = ("train_mean", "validation_mean", "manual")
SOLVER_CHOICES = ("CLARABEL", "ECOS", "SCS", "OSQP")


def parse_month(value: str | None) -> pd.Timestamp | None:
    """Parse YYYY-MM into a calendar month-end timestamp."""
    if value is None:
        return None
    try:
        return pd.Timestamp(value) + MonthEnd(0)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Expected month formatted as YYYY-MM, got {value!r}.") from exc


def parse_bool(value: str | bool) -> bool:
    """Parse a CLI boolean."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def parse_cost_grid(value: str) -> list[float]:
    """Parse comma-separated cost bps values."""
    try:
        costs = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --cost-grid {value!r}.") from exc
    if not costs:
        raise argparse.ArgumentTypeError("--cost-grid must contain at least one value.")
    if any(cost < 0 for cost in costs):
        raise argparse.ArgumentTypeError("--cost-grid values must be nonnegative.")
    return costs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run focused Tier-1 RL overlay ablations.")
    parser.add_argument("--ablation-type", required=True, choices=ABLATION_TYPES)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--outdir", default="data/ablation_v1")
    parser.add_argument("--pred-file", default="data/prediction/fm_oos_predictions.parquet")
    parser.add_argument("--risk-dir", default="data/risk/risk_cov_npz")
    parser.add_argument("--risk-meta-file", default="data/risk/risk_monthly_metadata.csv")
    parser.add_argument("--returns-file", default="data/panel/monthly_stock_panel_basic_full.parquet")
    parser.add_argument("--rl-backtest-file", default="data/rl_overlay_sac/test_backtest.csv")
    parser.add_argument("--rl-weights-file", default="data/rl_overlay_sac/test_weights.parquet")
    parser.add_argument("--rl-action-history-file", default="data/rl_overlay_sac/test_action_history.csv")
    parser.add_argument("--fair-fixed-backtest-file", default="data/diagnostics/static_fixed_param_fair/static_fixed_fair_backtest.csv")
    parser.add_argument("--fair-fixed-weights-file", default="data/diagnostics/static_fixed_param_fair/static_fixed_fair_weights.parquet")
    parser.add_argument("--fair-fixed-summary-file", default="data/diagnostics/static_fixed_param_fair/static_fixed_fair_summary.csv")
    parser.add_argument("--train-action-history-file", default="data/rl_overlay_sac/train_action_history.csv")
    parser.add_argument("--validation-action-history-file", default="data/rl_overlay_sac/validation_action_history.csv")
    parser.add_argument("--fixed-param-source", default="validation_mean", choices=FIXED_PARAM_SOURCES)
    parser.add_argument("--fixed-lambda", type=float, default=None)
    parser.add_argument("--fixed-tau", type=float, default=None)
    parser.add_argument("--state-variant", default="full_summary_state")
    parser.add_argument("--state-backtest-file", default=None)
    parser.add_argument("--state-action-history-file", default=None)
    parser.add_argument("--allow-missing-state-variant", type=parse_bool, default=True)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--cost-grid", type=parse_cost_grid, default=parse_cost_grid("0,10,20,50"))
    parser.add_argument("--solver", default="CLARABEL", choices=SOLVER_CHOICES)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--test-start", type=parse_month, default=parse_month("2020-01"))
    parser.add_argument("--test-end", type=parse_month, default=parse_month("2024-11"))
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


def require_columns(df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> None:
    """Validate required columns."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")


def normalize_month_end(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Normalize a month_end column."""
    result = df.copy()
    if "month_end" not in result.columns:
        raise ValueError(f"{dataset_name} must contain month_end.")
    result["month_end"] = pd.to_datetime(result["month_end"], errors="coerce") + MonthEnd(0)
    bad = int(result["month_end"].isna().sum())
    if bad:
        raise ValueError(f"{dataset_name} has {bad:,} invalid month_end values.")
    return result


def load_backtest(path: Path, dataset_name: str) -> pd.DataFrame:
    """Load a backtest CSV."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, ["month_end", "net_return"], dataset_name)
    df = normalize_month_end(df, dataset_name)
    for col in ["gross_return", "net_return", "cost", "turnover", "lambda_t", "tau_t", "fixed_lambda", "fixed_tau"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("month_end").reset_index(drop=True)


def load_weights(path: Path) -> pd.DataFrame:
    """Load weights if available."""
    if not path.exists():
        raise FileNotFoundError(f"weights file not found: {path}")
    df = pd.read_parquet(path)
    require_columns(df, ["month_end", "permno", "weight"], "weights file")
    df = normalize_month_end(df, "weights file")
    return df.sort_values(["month_end", "permno"]).reset_index(drop=True)


def filter_test_period(df: pd.DataFrame, test_start: pd.Timestamp | None, test_end: pd.Timestamp | None) -> pd.DataFrame:
    """Keep rows in the requested test window."""
    result = df.copy()
    if test_start is not None:
        result = result.loc[result["month_end"] >= test_start]
    if test_end is not None:
        result = result.loc[result["month_end"] <= test_end]
    return result.reset_index(drop=True)


def recompute_nav(backtest: pd.DataFrame) -> pd.DataFrame:
    """Recompute cumulative NAV from net_return."""
    result = backtest.copy()
    result["net_return"] = pd.to_numeric(result["net_return"], errors="coerce")
    result["cumulative_nav"] = (1.0 + result["net_return"].fillna(0.0)).cumprod()
    return result


def compute_summary(backtest: pd.DataFrame, run_name: str, ablation_type: str, role: str, cost_bps: float | None) -> dict[str, object]:
    """Compute scalar metrics for an ablation run."""
    if "solver_status" in backtest.columns:
        successful = backtest.loc[backtest["solver_status"].isin(SUCCESS_STATUSES)].copy()
    else:
        successful = backtest.copy()
    successful = successful.dropna(subset=["net_return"]).copy()
    if "cumulative_nav" not in successful.columns or successful["cumulative_nav"].isna().all():
        successful = recompute_nav(successful)
    net = successful["net_return"].astype(float) if not successful.empty else pd.Series(dtype=float)
    gross = successful["gross_return"].astype(float) if "gross_return" in successful.columns and not successful.empty else net
    net_vol = float(net.std(ddof=1) * np.sqrt(12.0)) if len(net) > 1 else np.nan
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(12.0)) if len(net) > 1 and net.std(ddof=1) > 0 else np.nan
    lambda_series = successful["lambda_t"] if "lambda_t" in successful.columns else successful.get("fixed_lambda", pd.Series(dtype=float))
    tau_series = successful["tau_t"] if "tau_t" in successful.columns else successful.get("fixed_tau", pd.Series(dtype=float))
    return {
        "run_name": run_name,
        "ablation_type": ablation_type,
        "role": role,
        "status": "ok" if not successful.empty else "no_valid_months",
        "cost_bps": np.nan if cost_bps is None else float(cost_bps),
        "start_month": successful["month_end"].min().date().isoformat() if not successful.empty else "",
        "end_month": successful["month_end"].max().date().isoformat() if not successful.empty else "",
        "n_months": int(len(successful)),
        "annualized_return": annualized_return(net),
        "annualized_gross_return": annualized_return(gross),
        "annualized_volatility": net_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(successful["cumulative_nav"]) if not successful.empty else np.nan,
        "cumulative_return": float(successful["cumulative_nav"].iloc[-1] - 1.0) if not successful.empty else np.nan,
        "mean_monthly_turnover": float(successful["turnover"].mean()) if "turnover" in successful.columns and not successful.empty else np.nan,
        "mean_cost": float(successful["cost"].mean()) if "cost" in successful.columns and not successful.empty else np.nan,
        "mean_lambda": float(pd.to_numeric(lambda_series, errors="coerce").mean()) if len(lambda_series) else np.nan,
        "std_lambda": float(pd.to_numeric(lambda_series, errors="coerce").std(ddof=1)) if len(lambda_series) > 1 else np.nan,
        "mean_tau": float(pd.to_numeric(tau_series, errors="coerce").mean()) if len(tau_series) else np.nan,
        "std_tau": float(pd.to_numeric(tau_series, errors="coerce").std(ddof=1)) if len(tau_series) > 1 else np.nan,
    }


def write_summary_text(path: Path, summary: dict[str, object]) -> None:
    """Write a readable run summary."""
    lines = [
        f"{summary['run_name']} Tier-1 Ablation Summary",
        "",
        f"Ablation type: {summary['ablation_type']}",
        f"Role: {summary['role']}",
        f"Status: {summary['status']}",
        f"Cost bps: {summary['cost_bps']}",
        f"Date range: {summary['start_month']} to {summary['end_month']}",
        f"n_months: {summary['n_months']}",
        f"Annualized return: {summary['annualized_return']:.6f}",
        f"Annualized volatility: {summary['annualized_volatility']:.6f}",
        f"Sharpe ratio: {summary['sharpe_ratio']:.6f}",
        f"Max drawdown: {summary['max_drawdown']:.6f}",
        f"Cumulative return: {summary['cumulative_return']:.6f}",
        f"Mean monthly turnover: {summary['mean_monthly_turnover']:.6f}",
        f"Mean cost: {summary['mean_cost']:.6f}",
        f"Mean lambda: {summary['mean_lambda']:.6f}",
        f"Std lambda: {summary['std_lambda']:.6f}",
        f"Mean tau: {summary['mean_tau']:.6f}",
        f"Std tau: {summary['std_tau']:.6f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_run_outputs(
    outdir: Path,
    ablation_type: str,
    run_name: str,
    role: str,
    backtest: pd.DataFrame,
    weights: pd.DataFrame | None,
    cost_bps: float | None,
) -> pd.DataFrame:
    """Save monthly, weight, summary, and text outputs for one run."""
    run_dir = outdir / ablation_type / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    backtest = backtest.sort_values("month_end").reset_index(drop=True)
    backtest.to_csv(run_dir / f"{run_name}_backtest.csv", index=False)
    if weights is not None and not weights.empty:
        weights.to_parquet(run_dir / f"{run_name}_weights.parquet", index=False, compression=PARQUET_COMPRESSION)
    summary = compute_summary(backtest, run_name, ablation_type, role, cost_bps)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(run_dir / f"{run_name}_summary.csv", index=False)
    write_summary_text(run_dir / f"{run_name}_summary.txt", summary)
    return summary_df


def save_missing_run(outdir: Path, ablation_type: str, run_name: str, role: str, reason: str) -> pd.DataFrame:
    """Save a deterministic missing-artifact summary row."""
    run_dir = outdir / ablation_type / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    monthly = pd.DataFrame(
        columns=["month_end", "lambda_t", "tau_t", "turnover", "gross_return", "cost", "net_return", "cumulative_nav"]
    )
    monthly.to_csv(run_dir / f"{run_name}_backtest.csv", index=False)
    summary = {
        "run_name": run_name,
        "ablation_type": ablation_type,
        "role": role,
        "status": f"missing_artifact: {reason}",
        "cost_bps": np.nan,
        "start_month": "",
        "end_month": "",
        "n_months": 0,
        "annualized_return": np.nan,
        "annualized_gross_return": np.nan,
        "annualized_volatility": np.nan,
        "sharpe_ratio": np.nan,
        "max_drawdown": np.nan,
        "cumulative_return": np.nan,
        "mean_monthly_turnover": np.nan,
        "mean_cost": np.nan,
        "mean_lambda": np.nan,
        "std_lambda": np.nan,
        "mean_tau": np.nan,
        "std_tau": np.nan,
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(run_dir / f"{run_name}_summary.csv", index=False)
    write_summary_text(run_dir / f"{run_name}_summary.txt", summary)
    return summary_df


def load_common_inputs(args: argparse.Namespace, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[pd.Timestamp, Path], list[pd.Timestamp]]:
    """Load prediction, return, risk inputs and build the test month sequence."""
    predictions = clean_predictions(read_parquet_file(resolve_path(project_root, args.pred_file), "prediction file"))
    returns = clean_returns(read_parquet_file(resolve_path(project_root, args.returns_file), "return panel"))
    risk_meta_file = resolve_path(project_root, args.risk_meta_file)
    risk_metadata = read_risk_metadata(risk_meta_file)
    if not risk_metadata.empty:
        LOGGER.info("Loaded risk metadata with %s rows.", f"{len(risk_metadata):,}")
    cov_files = discover_covariance_files(resolve_path(project_root, args.risk_dir))
    months = build_month_sequence(predictions, cov_files, args.test_start, args.test_end)
    if not months:
        raise RuntimeError("No overlapping prediction/risk months found for requested test window.")
    return predictions, returns, cov_files, months


def make_optimizer_args(args: argparse.Namespace, cost_bps: float) -> SimpleNamespace:
    """Create the small args object expected by shared optimizer helpers."""
    return SimpleNamespace(solver=args.solver, max_weight=args.max_weight, cost_bps=float(cost_bps))


def build_action_schedule(
    actions: pd.DataFrame,
    months: list[pd.Timestamp],
    fixed_lambda: float | None = None,
    fixed_tau: float | None = None,
) -> pd.DataFrame:
    """Build month-indexed lambda/tau schedule for dynamic replay."""
    actions = normalize_month_end(actions, "action history")
    require_columns(actions, ["month_end", "lambda_t", "tau_t"], "action history")
    schedule = actions[["month_end", "lambda_t", "tau_t"]].copy()
    schedule["lambda_t"] = pd.to_numeric(schedule["lambda_t"], errors="coerce")
    schedule["tau_t"] = pd.to_numeric(schedule["tau_t"], errors="coerce")
    schedule = schedule.dropna(subset=["lambda_t", "tau_t"]).drop_duplicates("month_end", keep="last")
    schedule = schedule.loc[schedule["month_end"].isin(months)].copy()
    if fixed_lambda is not None:
        schedule["lambda_t"] = float(fixed_lambda)
    if fixed_tau is not None:
        schedule["tau_t"] = float(fixed_tau)
    if schedule.empty:
        raise RuntimeError("No action rows overlap the requested test months.")
    return schedule.sort_values("month_end").reset_index(drop=True)


def run_action_path_backtest(
    months: list[pd.Timestamp],
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    cov_files: dict[pd.Timestamp, Path],
    action_schedule: pd.DataFrame,
    args: argparse.Namespace,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a replay backtest using a supplied month-by-month lambda/tau path."""
    pred_by_month = {month: df for month, df in predictions.groupby("month_end", sort=False)}
    returns_by_month = {month: df for month, df in returns.groupby("month_end", sort=False)}
    action_by_month = action_schedule.set_index("month_end")[["lambda_t", "tau_t"]]
    optimizer_args = make_optimizer_args(args, cost_bps)
    previous_weights: pd.Series | None = None
    nav = 1.0
    weight_rows: list[pd.DataFrame] = []
    backtest_rows: list[dict[str, object]] = []

    for month_end in months:
        if month_end not in action_by_month.index:
            LOGGER.warning("Skipping %s: no action for this month.", month_end.date())
            continue
        action = action_by_month.loc[month_end]
        lambda_t = float(action["lambda_t"])
        tau_t = float(action["tau_t"])
        next_month = month_end + MonthEnd(1)
        pred_month = pred_by_month.get(month_end)
        returns_next = returns_by_month.get(next_month)
        if pred_month is None or pred_month.empty or returns_next is None or returns_next["retadj"].notna().sum() == 0:
            LOGGER.warning("Skipping %s: missing predictions or next returns.", month_end.date())
            continue
        try:
            cov_data = load_covariance_npz(cov_files[month_end], month_end)
            permnos, mu, cov, vols = align_month_inputs(month_end, pred_month, cov_data)
        except Exception as exc:
            LOGGER.warning("Skipping %s: failed to align inputs: %s", month_end.date(), exc)
            continue
        n_assets = len(permnos)
        if n_assets < 2:
            LOGGER.warning("Skipping %s: only %s assets remain.", month_end.date(), n_assets)
            continue
        if optimizer_args.max_weight is not None and optimizer_args.max_weight * n_assets < 1.0 - 1e-10:
            LOGGER.warning("Skipping %s: max-weight infeasible.", month_end.date())
            continue

        result = solve_allocator(
            mu,
            cov,
            align_previous_weights(permnos, previous_weights),
            lambda_t,
            tau_t,
            optimizer_args.solver,
            optimizer_args.max_weight,
        )
        if result.weights is None:
            backtest_rows.append(
                {
                    "month_end": month_end,
                    "lambda_t": lambda_t,
                    "tau_t": tau_t,
                    "n_assets": n_assets,
                    "solver_status": result.status,
                    "objective_value": result.objective_value,
                    "turnover": np.nan,
                    "gross_return": np.nan,
                    "cost": np.nan,
                    "net_return": np.nan,
                    "cumulative_nav": nav,
                }
            )
            continue

        new_weights = pd.Series(result.weights, index=pd.Index(permnos, name="permno"))
        prior = previous_weights
        if prior is None or prior.empty:
            prior = pd.Series(np.repeat(1.0 / n_assets, n_assets), index=pd.Index(permnos, name="permno"))
        turnover = compute_turnover(new_weights, prior)
        gross_return = evaluate_next_return(month_end, new_weights, returns_next)
        if gross_return is None:
            LOGGER.warning("Skipping %s: no usable realized returns.", month_end.date())
            continue
        cost = (cost_bps / 10000.0) * turnover
        net_return = gross_return - cost
        nav *= 1.0 + net_return
        weight_rows.append(
            pd.DataFrame(
                {
                    "month_end": month_end,
                    "permno": permnos.astype("int64"),
                    "weight": result.weights,
                    "lambda_t": lambda_t,
                    "tau_t": tau_t,
                    "mu_hat": mu,
                    "vol_hat": vols if vols is not None else np.nan,
                    "in_optimizer_universe": True,
                }
            )
        )
        backtest_rows.append(
            {
                "month_end": month_end,
                "lambda_t": lambda_t,
                "tau_t": tau_t,
                "n_assets": n_assets,
                "solver_status": result.status,
                "objective_value": result.objective_value,
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "cumulative_nav": nav,
            }
        )
        previous_weights = new_weights

    weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    backtest = pd.DataFrame(backtest_rows)
    if not weights.empty:
        weights = weights.sort_values(["month_end", "permno"]).reset_index(drop=True)
    if not backtest.empty:
        backtest = backtest.sort_values("month_end").reset_index(drop=True)
    return weights, backtest


def select_fixed_pair(args: argparse.Namespace, project_root: Path) -> tuple[float, float]:
    """Select ex-ante fixed lambda/tau using train/validation/manual source."""
    fixed_lambda, fixed_tau, _ = select_fixed_parameters(
        args.fixed_param_source,
        resolve_path(project_root, args.train_action_history_file),
        resolve_path(project_root, args.validation_action_history_file),
        args.fixed_lambda,
        args.fixed_tau,
        args.test_start,
    )
    return fixed_lambda, fixed_tau


def save_rl_baseline(args: argparse.Namespace, project_root: Path, outdir: Path, ablation_type: str) -> pd.DataFrame:
    """Save the existing RL test backtest as a comparable ablation row."""
    backtest = filter_test_period(load_backtest(resolve_path(project_root, args.rl_backtest_file), "RL backtest"), args.test_start, args.test_end)
    weights_path = resolve_path(project_root, args.rl_weights_file)
    weights = load_weights(weights_path) if weights_path.exists() else None
    if weights is not None:
        weights = filter_test_period(weights, args.test_start, args.test_end)
    return save_run_outputs(outdir, ablation_type, "rl_overlay_sac", "baseline_rl_overlay", backtest, weights, args.cost_bps)


def run_fair_fixed_param(args: argparse.Namespace, project_root: Path, outdir: Path) -> list[pd.DataFrame]:
    """Compare existing RL overlay against fair fixed-parameter benchmark."""
    rows = [save_rl_baseline(args, project_root, outdir, "fair_fixed_param")]
    fair_backtest_path = resolve_path(project_root, args.fair_fixed_backtest_file)
    fair_weights_path = resolve_path(project_root, args.fair_fixed_weights_file)
    if fair_backtest_path.exists():
        backtest = filter_test_period(load_backtest(fair_backtest_path, "fair fixed backtest"), args.test_start, args.test_end)
        weights = load_weights(fair_weights_path) if fair_weights_path.exists() else None
        if weights is not None:
            weights = filter_test_period(weights, args.test_start, args.test_end)
        rows.append(save_run_outputs(outdir, "fair_fixed_param", "static_fixed_param_fair", "fair_fixed_ex_ante", backtest, weights, args.cost_bps))
        return rows

    LOGGER.info("Fair fixed benchmark output not found; computing it inside ablation runner.")
    try:
        fixed_lambda, fixed_tau = select_fixed_pair(args, project_root)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Cannot compute fair fixed benchmark: %s", exc)
        rows.append(
            save_missing_run(
                outdir,
                "fair_fixed_param",
                "static_fixed_param_fair",
                "fair_fixed_ex_ante",
                str(exc),
            )
        )
        return rows
    predictions, returns, cov_files, months = load_common_inputs(args, project_root)
    schedule = pd.DataFrame({"month_end": months, "lambda_t": fixed_lambda, "tau_t": fixed_tau})
    weights, backtest = run_action_path_backtest(months, predictions, returns, cov_files, schedule, args, args.cost_bps)
    rows.append(save_run_outputs(outdir, "fair_fixed_param", "static_fixed_param_fair", "fair_fixed_ex_ante", backtest, weights, args.cost_bps))
    return rows


def run_control_dimension(args: argparse.Namespace, project_root: Path, outdir: Path, ablation_type: str) -> list[pd.DataFrame]:
    """Run lambda-only or tau-only control replay."""
    run_name = "lambda_only_rl" if ablation_type == "lambda_only" else "tau_only_rl"
    role = "sac_lambda_dynamic_tau_fixed" if ablation_type == "lambda_only" else "sac_tau_dynamic_lambda_fixed"
    try:
        fixed_lambda, fixed_tau = select_fixed_pair(args, project_root)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.warning("Cannot run %s because fixed parameter selection failed: %s", ablation_type, exc)
        return [save_missing_run(outdir, ablation_type, run_name, role, str(exc))]
    predictions, returns, cov_files, months = load_common_inputs(args, project_root)
    actions = read_action_history(resolve_path(project_root, args.rl_action_history_file), "RL test action history")
    if ablation_type == "lambda_only":
        schedule = build_action_schedule(actions, months, fixed_tau=fixed_tau)
    elif ablation_type == "tau_only":
        schedule = build_action_schedule(actions, months, fixed_lambda=fixed_lambda)
    else:
        raise ValueError(f"Unsupported control ablation type: {ablation_type}")
    weights, backtest = run_action_path_backtest(months, predictions, returns, cov_files, schedule, args, args.cost_bps)
    return [save_run_outputs(outdir, ablation_type, run_name, role, backtest, weights, args.cost_bps)]


def run_state_ablation(args: argparse.Namespace, project_root: Path, outdir: Path) -> list[pd.DataFrame]:
    """Register a state-ablation backtest artifact for comparison."""
    variant = args.state_variant
    if args.state_backtest_file is None and variant == "full_summary_state":
        backtest_path = resolve_path(project_root, args.rl_backtest_file)
        action_path = resolve_path(project_root, args.rl_action_history_file)
    elif args.state_backtest_file is not None:
        backtest_path = resolve_path(project_root, args.state_backtest_file)
        action_path = resolve_path(project_root, args.state_action_history_file) if args.state_action_history_file else None
    else:
        default_path = f"data/rl_overlay_sac_{variant}/test_backtest.csv"
        backtest_path = resolve_path(project_root, default_path)
        action_path = resolve_path(project_root, f"data/rl_overlay_sac_{variant}/test_action_history.csv")

    if not backtest_path.exists():
        if args.allow_missing_state_variant:
            LOGGER.warning("State variant %s missing backtest artifact: %s", variant, backtest_path)
            return [save_missing_run(outdir, "state_ablation", variant, "state_variant", str(backtest_path))]
        raise FileNotFoundError(f"State variant {variant} backtest not found: {backtest_path}")

    backtest = filter_test_period(load_backtest(backtest_path, f"{variant} backtest"), args.test_start, args.test_end)
    if action_path is not None and action_path.exists():
        actions = read_action_history(action_path, f"{variant} action history")
        actions = normalize_month_end(actions, f"{variant} action history")
        action_cols = actions[["month_end", "lambda_t", "tau_t"]].drop_duplicates("month_end", keep="last")
        backtest = backtest.drop(columns=[col for col in ["lambda_t", "tau_t"] if col in backtest.columns], errors="ignore")
        backtest = backtest.merge(action_cols, on="month_end", how="left")
    return [save_run_outputs(outdir, "state_ablation", variant, "state_variant", backtest, None, args.cost_bps)]


def run_cost_robustness(args: argparse.Namespace, project_root: Path, outdir: Path) -> list[pd.DataFrame]:
    """Revalue the SAC action path under alternative transaction-cost assumptions."""
    predictions, returns, cov_files, months = load_common_inputs(args, project_root)
    actions = read_action_history(resolve_path(project_root, args.rl_action_history_file), "RL test action history")
    schedule = build_action_schedule(actions, months)
    rows = []
    for cost_bps in args.cost_grid:
        run_name = f"rl_overlay_sac_cost_{format_cost_label(cost_bps)}bps"
        weights, backtest = run_action_path_backtest(months, predictions, returns, cov_files, schedule, args, cost_bps)
        rows.append(save_run_outputs(outdir, "cost_robustness", run_name, "rl_action_path_cost_revalue", backtest, weights, cost_bps))
    return rows


def format_cost_label(cost_bps: float) -> str:
    """Format cost bps for filenames."""
    if float(cost_bps).is_integer():
        return str(int(cost_bps))
    return str(cost_bps).replace(".", "p")


def update_master_comparison(outdir: Path) -> pd.DataFrame:
    """Aggregate all run-level summaries under the ablation output directory."""
    summary_files = sorted(path for path in outdir.glob("*/*/*_summary.csv") if path.name != "tier1_ablation_master_comparison.csv")
    frames = []
    for path in summary_files:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable summary %s: %s", path, exc)
            continue
        frame["summary_file"] = str(path.relative_to(outdir))
        frames.append(frame)
    if frames:
        master = pd.concat(frames, ignore_index=True)
        master = master.sort_values(["ablation_type", "run_name"]).reset_index(drop=True)
    else:
        master = pd.DataFrame()
    master.to_csv(outdir / "tier1_ablation_master_comparison.csv", index=False)
    return master


def plot_comparison(master: pd.DataFrame, outdir: Path) -> None:
    """Create compact comparison plots from the master summary."""
    if master.empty:
        LOGGER.warning("Skipping comparison plots because master summary is empty.")
        return
    ok = master.loc[master["status"].astype(str) == "ok"].copy()
    if ok.empty:
        LOGGER.warning("Skipping comparison plots because no successful runs are available.")
        return
    plot_bar(ok, "cumulative_return", "Tier-1 Ablation Cumulative Return", "Cumulative return", outdir / "tier1_cumulative_return_comparison.png")
    plot_bar(ok, "mean_monthly_turnover", "Tier-1 Ablation Mean Turnover", "Mean L1 turnover", outdir / "tier1_turnover_comparison.png")
    action_cols = ok.loc[ok[["mean_lambda", "mean_tau"]].notna().any(axis=1), ["run_name", "mean_lambda", "mean_tau"]]
    if not action_cols.empty:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        x = np.arange(len(action_cols))
        axes[0].bar(x, action_cols["mean_lambda"].astype(float))
        axes[0].set_ylabel("mean lambda")
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[1].bar(x, action_cols["mean_tau"].astype(float))
        axes[1].set_ylabel("mean tau")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(action_cols["run_name"], rotation=45, ha="right")
        axes[1].grid(True, axis="y", alpha=0.3)
        fig.suptitle("Tier-1 Ablation Lambda/Tau Comparison")
        fig.tight_layout()
        fig.savefig(outdir / "tier1_lambda_tau_comparison.png", dpi=150)
        plt.close(fig)


def plot_bar(df: pd.DataFrame, y_col: str, title: str, ylabel: str, path: Path) -> None:
    """Save a simple bar comparison."""
    if y_col not in df.columns:
        return
    plot_df = df.dropna(subset=[y_col]).copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df[y_col].astype(float))
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["run_name"], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the requested Tier-1 ablation."""
    if args.cost_bps < 0:
        raise ValueError("--cost-bps must be nonnegative.")
    if args.max_weight is not None and not (0 < args.max_weight <= 1):
        raise ValueError("--max-weight must be in (0, 1] when provided.")
    if args.test_start is not None and args.test_end is not None and args.test_start > args.test_end:
        raise ValueError("--test-start must be less than or equal to --test-end.")

    project_root = Path(args.project_root).expanduser().resolve()
    outdir = resolve_path(project_root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.ablation_type == "fair_fixed_param":
        run_fair_fixed_param(args, project_root, outdir)
    elif args.ablation_type in {"lambda_only", "tau_only"}:
        run_control_dimension(args, project_root, outdir, args.ablation_type)
    elif args.ablation_type == "state_ablation":
        run_state_ablation(args, project_root, outdir)
    elif args.ablation_type == "cost_robustness":
        run_cost_robustness(args, project_root, outdir)
    else:
        raise ValueError(f"Unsupported ablation type: {args.ablation_type}")

    master = update_master_comparison(outdir)
    plot_comparison(master, outdir)
    LOGGER.info("Updated master comparison at %s.", outdir / "tier1_ablation_master_comparison.csv")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
