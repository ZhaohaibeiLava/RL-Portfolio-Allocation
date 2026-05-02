#!/usr/bin/env python3
"""Run a fixed-parameter static benchmark using RL-implied lambda and tau.

This diagnostic answers whether the SAC overlay mainly added value by changing
parameters through time, or whether its average parameter choice is already a
better static allocator configuration.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("run_static_fixed_parameter_benchmark")

PARQUET_COMPRESSION = "snappy"
SOLVER_CHOICES = ("CLARABEL", "ECOS", "SCS", "OSQP")
SUCCESS_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


@dataclass(frozen=True)
class CovarianceData:
    """Month-t covariance payload loaded from a risk-layer NPZ."""

    month_end: pd.Timestamp
    cov: np.ndarray
    permnos: np.ndarray
    vols: np.ndarray | None


@dataclass(frozen=True)
class OptimizationResult:
    """Result of one convex portfolio optimization."""

    status: str
    objective_value: float
    weights: np.ndarray | None


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
    parser = argparse.ArgumentParser(
        description="Run a static allocator benchmark using fixed lambda/tau from RL action history."
    )
    parser.add_argument("--project-root", default=".", help="Project root. Default: .")
    parser.add_argument("--pred-file", default="data/prediction/fm_oos_predictions.parquet")
    parser.add_argument("--risk-dir", default="data/risk/risk_cov_npz")
    parser.add_argument("--risk-meta-file", default="data/risk/risk_monthly_metadata.csv")
    parser.add_argument("--returns-file", default="data/panel/monthly_stock_panel_basic_full.parquet")
    parser.add_argument("--action-history-file", default="data/rl_overlay_sac/test_action_history.csv")
    parser.add_argument("--outdir", default="data/diagnostics/static_fixed_param")
    parser.add_argument("--fixed-lambda", type=float, default=None)
    parser.add_argument("--fixed-tau", type=float, default=None)
    parser.add_argument("--cost-bps", type=float, default=10.0)
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


def resolve_path(project_root: Path, path_like: str) -> Path:
    """Resolve a CLI path relative to project root unless absolute."""
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else project_root / path


def require_columns(df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> None:
    """Validate that required columns exist."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")


def normalize_month_end(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Ensure month_end exists and is normalized to calendar month-end."""
    result = df.copy()
    if "month_end" not in result.columns:
        if "date" not in result.columns:
            raise ValueError(f"{dataset_name} must contain month_end or date.")
        result["month_end"] = pd.to_datetime(result["date"], errors="coerce") + MonthEnd(0)
    else:
        result["month_end"] = pd.to_datetime(result["month_end"], errors="coerce") + MonthEnd(0)
    bad = int(result["month_end"].isna().sum())
    if bad:
        raise ValueError(f"{dataset_name} has {bad:,} invalid month_end values.")
    return result


def read_parquet_file(path: Path, dataset_name: str) -> pd.DataFrame:
    """Read a parquet file with clear error context."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {path}")
    LOGGER.info("Loading %s from %s.", dataset_name, path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {dataset_name} file {path}: {exc}") from exc


def read_risk_metadata(path: Path) -> pd.DataFrame:
    """Read risk metadata for logging and validation context."""
    if not path.exists():
        LOGGER.warning("Risk metadata file not found: %s", path)
        return pd.DataFrame()
    metadata = pd.read_csv(path)
    if "month_end" in metadata.columns:
        metadata = normalize_month_end(metadata, "risk metadata")
    return metadata


def clean_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Clean expected-return forecasts."""
    require_columns(predictions, ["permno", "month_end", "mu_hat"], "prediction file")
    result = normalize_month_end(predictions, "prediction file")
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce")
    result["mu_hat"] = pd.to_numeric(result["mu_hat"], errors="coerce")
    before = len(result)
    result = result.dropna(subset=["permno", "month_end", "mu_hat"]).copy()
    result["permno"] = result["permno"].astype("int64")
    duplicates = int(result.duplicated(["permno", "month_end"]).sum())
    if duplicates:
        LOGGER.warning("Dropping %s duplicate prediction keys.", f"{duplicates:,}")
        result = result.drop_duplicates(["permno", "month_end"], keep="last")
    LOGGER.info("Kept %s of %s prediction rows.", f"{len(result):,}", f"{before:,}")
    return result.sort_values(["month_end", "permno"]).reset_index(drop=True)


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Clean realized return panel."""
    require_columns(returns, ["permno", "month_end", "retadj"], "return panel")
    result = normalize_month_end(returns, "return panel")
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce")
    result["retadj"] = pd.to_numeric(result["retadj"], errors="coerce")
    before = len(result)
    result = result.dropna(subset=["permno", "month_end"]).copy()
    result["permno"] = result["permno"].astype("int64")
    exact_duplicates = int(result.duplicated(["permno", "month_end", "retadj"]).sum())
    if exact_duplicates:
        result = result.drop_duplicates(["permno", "month_end", "retadj"], keep="last")
    duplicate_keys = int(result.duplicated(["permno", "month_end"]).sum())
    if duplicate_keys:
        raise ValueError(f"Return panel has {duplicate_keys:,} duplicate (permno, month_end) keys.")
    LOGGER.info("Kept %s of %s return rows.", f"{len(result):,}", f"{before:,}")
    return result[["permno", "month_end", "retadj"]].sort_values(["month_end", "permno"])


def infer_fixed_parameters(
    action_history_file: Path,
    fixed_lambda: float | None,
    fixed_tau: float | None,
    test_start: pd.Timestamp | None,
    test_end: pd.Timestamp | None,
) -> tuple[float, float]:
    """Use CLI overrides when present; otherwise infer means from RL action history."""
    if fixed_lambda is not None and fixed_tau is not None:
        return float(fixed_lambda), float(fixed_tau)
    if not action_history_file.exists():
        raise FileNotFoundError(
            f"Action history file not found: {action_history_file}. Provide --fixed-lambda and --fixed-tau."
        )
    actions = pd.read_csv(action_history_file)
    require_columns(actions, ["lambda_t", "tau_t"], "action history file")
    if "month_end" in actions.columns:
        actions = normalize_month_end(actions, "action history file")
        if test_start is not None:
            actions = actions.loc[actions["month_end"] >= test_start]
        if test_end is not None:
            actions = actions.loc[actions["month_end"] <= test_end]
        if actions.empty:
            raise ValueError("No action-history rows remain after applying the requested test window.")
    lambda_value = float(fixed_lambda) if fixed_lambda is not None else float(pd.to_numeric(actions["lambda_t"], errors="coerce").mean())
    tau_value = float(fixed_tau) if fixed_tau is not None else float(pd.to_numeric(actions["tau_t"], errors="coerce").mean())
    if not np.isfinite(lambda_value) or not np.isfinite(tau_value):
        raise ValueError("Could not infer finite fixed lambda/tau from action history.")
    return lambda_value, tau_value


def discover_covariance_files(risk_dir: Path) -> dict[pd.Timestamp, Path]:
    """Discover cov_YYYYMM.npz files keyed by month-end timestamp."""
    if not risk_dir.exists():
        raise FileNotFoundError(f"Risk covariance directory not found: {risk_dir}")
    pattern = re.compile(r"^cov_(\d{6})\.npz$")
    files: dict[pd.Timestamp, Path] = {}
    for path in sorted(risk_dir.glob("cov_*.npz")):
        match = pattern.match(path.name)
        if not match:
            LOGGER.warning("Ignoring covariance file with unexpected name: %s", path.name)
            continue
        month = pd.Timestamp(f"{match.group(1)[:4]}-{match.group(1)[4:]}-01") + MonthEnd(0)
        files[month] = path
    LOGGER.info("Discovered %s covariance files in %s.", f"{len(files):,}", risk_dir)
    return files


def load_covariance_npz(path: Path, fallback_month: pd.Timestamp) -> CovarianceData:
    """Load and validate one covariance NPZ."""
    with np.load(path, allow_pickle=True) as payload:
        missing = sorted({"cov", "permnos"}.difference(payload.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
        cov = np.asarray(payload["cov"], dtype=float)
        permnos = np.asarray(payload["permnos"]).astype("int64")
        vols = np.asarray(payload["vols"], dtype=float) if "vols" in payload.files else None
        month_end = fallback_month
        if "month_end" in payload.files:
            month_end = pd.to_datetime(np.asarray(payload["month_end"]).reshape(-1)[0]) + MonthEnd(0)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != len(permnos):
        raise ValueError(f"{path} has inconsistent covariance/permno dimensions.")
    if vols is not None and len(vols) != len(permnos):
        vols = None
    return CovarianceData(pd.Timestamp(month_end), 0.5 * (cov + cov.T), permnos, vols)


def build_month_sequence(
    predictions: pd.DataFrame,
    cov_files: dict[pd.Timestamp, Path],
    test_start: pd.Timestamp | None,
    test_end: pd.Timestamp | None,
) -> list[pd.Timestamp]:
    """Find sorted rebalance months available in predictions and risk files."""
    months = sorted(set(pd.Series(predictions["month_end"].unique()).map(pd.Timestamp)).intersection(cov_files))
    if test_start is not None:
        months = [month for month in months if month >= test_start]
    if test_end is not None:
        months = [month for month in months if month <= test_end]
    return months


def align_month_inputs(
    month_end: pd.Timestamp,
    pred_month: pd.DataFrame,
    cov_data: CovarianceData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Align month-t mu and covariance in covariance-file permno order."""
    pred_lookup = pred_month.set_index("permno")["mu_hat"]
    keep_mask = np.array([permno in pred_lookup.index for permno in cov_data.permnos], dtype=bool)
    positions = np.flatnonzero(keep_mask)
    permnos = cov_data.permnos[positions]
    mu = pred_lookup.reindex(permnos).to_numpy(dtype=float)
    cov = cov_data.cov[np.ix_(positions, positions)]
    vols = cov_data.vols[positions] if cov_data.vols is not None else None
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError(f"{month_end.date()} aligned inputs contain non-finite values.")
    return permnos, mu, cov, vols


def align_previous_weights(current_permnos: np.ndarray, previous_weights: pd.Series | None) -> np.ndarray:
    """Align previous holdings to the current optimizer universe."""
    if previous_weights is None or previous_weights.empty:
        return np.repeat(1.0 / len(current_permnos), len(current_permnos))
    return previous_weights.reindex(current_permnos).fillna(0.0).to_numpy(dtype=float)


def solve_allocator(
    mu: np.ndarray,
    cov: np.ndarray,
    previous_weights_current: np.ndarray,
    fixed_lambda: float,
    fixed_tau: float,
    solver: str,
    max_weight: float | None,
) -> OptimizationResult:
    """Solve the long-only fully-invested mean-variance allocation problem."""
    n_assets = len(mu)
    w = cp.Variable(n_assets)
    objective = cp.Maximize(
        mu @ w - fixed_lambda * cp.quad_form(w, cp.psd_wrap(cov)) - fixed_tau * cp.norm1(w - previous_weights_current)
    )
    constraints = [w >= 0, cp.sum(w) == 1]
    if max_weight is not None:
        constraints.append(w <= max_weight)
    initial = np.clip(previous_weights_current, 0.0, None)
    w.value = initial / initial.sum() if initial.sum() > 0 else np.repeat(1.0 / n_assets, n_assets)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver, warm_start=True)
    except Exception as exc:
        LOGGER.warning("Solver %s failed with exception: %s", solver, exc)
        return OptimizationResult(f"solver_exception: {exc}", np.nan, None)
    status = str(problem.status)
    if status not in SUCCESS_STATUSES or w.value is None:
        return OptimizationResult(status, np.nan, None)
    weights = np.asarray(w.value, dtype=float).reshape(-1)
    weights[np.abs(weights) < 1e-12] = 0.0
    weights = np.clip(weights, 0.0, None)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return OptimizationResult(f"{status}_invalid_weights", np.nan, None)
    return OptimizationResult(status, float(problem.value), weights / weight_sum)


def compute_turnover(new_weights: pd.Series, previous_weights: pd.Series | None) -> float:
    """Compute full L1 turnover on the union of old and new holdings."""
    if previous_weights is None or previous_weights.empty:
        previous_weights = pd.Series(0.0, index=new_weights.index)
    union = new_weights.index.union(previous_weights.index)
    diff = new_weights.reindex(union).fillna(0.0) - previous_weights.reindex(union).fillna(0.0)
    return float(np.abs(diff.to_numpy(dtype=float)).sum())


def evaluate_next_return(month_end: pd.Timestamp, weights: pd.Series, returns_next: pd.DataFrame) -> float | None:
    """Evaluate realized t+1 return, renormalizing if coverage is partial."""
    realized = returns_next.set_index("permno")["retadj"].reindex(weights.index)
    observed = realized.notna()
    if not observed.any():
        return None
    if not observed.all():
        LOGGER.warning(
            "%s next-month returns partially missing; missing weight %.6f.",
            month_end.date(),
            float(weights.loc[~observed].sum()),
        )
    eval_weights = weights.loc[observed].astype(float)
    if eval_weights.sum() <= 0:
        return None
    eval_weights = eval_weights / eval_weights.sum()
    return float(eval_weights.to_numpy() @ realized.loc[observed].astype(float).to_numpy())


def annualized_return(monthly_returns: pd.Series) -> float:
    """Compute annualized geometric return."""
    clean = monthly_returns.dropna()
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() ** (12.0 / len(clean)) - 1.0)


def max_drawdown(nav: pd.Series) -> float:
    """Compute max drawdown from NAV series."""
    clean = nav.dropna()
    if clean.empty:
        return np.nan
    running_peak = clean.cummax().clip(lower=1.0)
    return float((clean / running_peak - 1.0).min())


def compute_summary(backtest: pd.DataFrame, attempted_months: int, fixed_lambda: float, fixed_tau: float) -> dict[str, float | int | str]:
    """Compute scalar performance diagnostics."""
    successes = backtest.loc[backtest["solver_status"].isin(SUCCESS_STATUSES)].copy()
    net = successes["net_return"].astype(float) if not successes.empty else pd.Series(dtype=float)
    gross = successes["gross_return"].astype(float) if not successes.empty else pd.Series(dtype=float)
    net_vol = float(net.std(ddof=1) * np.sqrt(12.0)) if len(net) > 1 else np.nan
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(12.0)) if len(net) > 1 and net.std(ddof=1) > 0 else np.nan
    return {
        "strategy": "static_fixed_param",
        "date_range": f"{successes['month_end'].min().date()} to {successes['month_end'].max().date()}" if not successes.empty else "n/a",
        "annualized_return": annualized_return(net),
        "annualized_gross_return": annualized_return(gross),
        "annualized_volatility": net_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown(successes["cumulative_nav"]) if not successes.empty else np.nan,
        "cumulative_return": float(successes["cumulative_nav"].iloc[-1] - 1.0) if not successes.empty else np.nan,
        "mean_monthly_turnover": float(successes["turnover"].mean()) if not successes.empty else np.nan,
        "mean_n_assets": float(successes["n_assets"].mean()) if not successes.empty else np.nan,
        "n_months": int(len(successes)),
        "attempted_months": int(attempted_months),
        "fixed_lambda": float(fixed_lambda),
        "fixed_tau": float(fixed_tau),
    }


def write_summary_text(path: Path, summary: dict[str, float | int | str]) -> None:
    """Write a human-readable summary."""
    lines = [
        "Static Fixed-Parameter Benchmark Summary",
        "",
        f"Date range: {summary['date_range']}",
        f"Fixed lambda used: {summary['fixed_lambda']}",
        f"Fixed tau used: {summary['fixed_tau']}",
        f"Attempted months: {summary['attempted_months']}",
        f"n_months: {summary['n_months']}",
        f"Annualized return: {summary['annualized_return']:.6f}",
        f"Annualized volatility: {summary['annualized_volatility']:.6f}",
        f"Sharpe ratio: {summary['sharpe_ratio']:.6f}",
        f"Max drawdown: {summary['max_drawdown']:.6f}",
        f"Cumulative return: {summary['cumulative_return']:.6f}",
        f"Mean monthly turnover: {summary['mean_monthly_turnover']:.6f}",
        f"Mean n_assets: {summary['mean_n_assets']:.2f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_series(df: pd.DataFrame, y_col: str, title: str, ylabel: str, path: Path) -> None:
    """Save a simple time-series plot."""
    if df.empty or y_col not in df:
        LOGGER.warning("Skipping plot %s.", path.name)
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(df["month_end"]), df[y_col], linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Rebalance month")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_backtest(
    months: list[pd.Timestamp],
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    cov_files: dict[pd.Timestamp, Path],
    fixed_lambda: float,
    fixed_tau: float,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the month-by-month fixed-parameter benchmark."""
    pred_by_month = {month: df for month, df in predictions.groupby("month_end", sort=False)}
    returns_by_month = {month: df for month, df in returns.groupby("month_end", sort=False)}
    previous_weights: pd.Series | None = None
    nav = 1.0
    weight_rows: list[pd.DataFrame] = []
    backtest_rows: list[dict[str, object]] = []

    for month_end in months:
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
        if args.max_weight is not None and args.max_weight * n_assets < 1.0 - 1e-10:
            LOGGER.warning("Skipping %s: max-weight infeasible.", month_end.date())
            continue

        result = solve_allocator(
            mu,
            cov,
            align_previous_weights(permnos, previous_weights),
            fixed_lambda,
            fixed_tau,
            args.solver,
            args.max_weight,
        )
        if result.weights is None:
            backtest_rows.append(
                {
                    "month_end": month_end,
                    "fixed_lambda": fixed_lambda,
                    "fixed_tau": fixed_tau,
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
        cost = (args.cost_bps / 10000.0) * turnover
        net_return = gross_return - cost
        nav *= 1.0 + net_return
        weight_rows.append(
            pd.DataFrame(
                {
                    "month_end": month_end,
                    "permno": permnos.astype("int64"),
                    "weight": result.weights,
                    "fixed_lambda": fixed_lambda,
                    "fixed_tau": fixed_tau,
                    "mu_hat": mu,
                    "vol_hat": vols if vols is not None else np.nan,
                    "in_optimizer_universe": True,
                }
            )
        )
        backtest_rows.append(
            {
                "month_end": month_end,
                "fixed_lambda": fixed_lambda,
                "fixed_tau": fixed_tau,
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
        LOGGER.info("%s solved: n=%s turnover=%.4f gross=%.4f net=%.4f", month_end.date(), n_assets, turnover, gross_return, net_return)

    weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    backtest = pd.DataFrame(backtest_rows)
    if not weights.empty:
        weights = weights.sort_values(["month_end", "permno"]).reset_index(drop=True)
    if not backtest.empty:
        backtest = backtest.sort_values("month_end").reset_index(drop=True)
    return weights, backtest


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the fixed-parameter benchmark workflow."""
    if args.cost_bps < 0:
        raise ValueError("--cost-bps must be nonnegative.")
    if args.max_weight is not None and not (0 < args.max_weight <= 1):
        raise ValueError("--max-weight must be in (0, 1] when provided.")
    if args.test_start is not None and args.test_end is not None and args.test_start > args.test_end:
        raise ValueError("--test-start must be less than or equal to --test-end.")

    project_root = Path(args.project_root).expanduser().resolve()
    pred_file = resolve_path(project_root, args.pred_file)
    risk_dir = resolve_path(project_root, args.risk_dir)
    risk_meta_file = resolve_path(project_root, args.risk_meta_file)
    returns_file = resolve_path(project_root, args.returns_file)
    action_history_file = resolve_path(project_root, args.action_history_file)
    outdir = resolve_path(project_root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fixed_lambda, fixed_tau = infer_fixed_parameters(
        action_history_file,
        args.fixed_lambda,
        args.fixed_tau,
        args.test_start,
        args.test_end,
    )
    if fixed_lambda < 0 or fixed_tau < 0:
        raise ValueError("Fixed lambda and tau must be nonnegative.")
    LOGGER.info("Using fixed lambda %.8f and fixed tau %.8f.", fixed_lambda, fixed_tau)

    predictions = clean_predictions(read_parquet_file(pred_file, "prediction file"))
    returns = clean_returns(read_parquet_file(returns_file, "return panel"))
    risk_metadata = read_risk_metadata(risk_meta_file)
    if not risk_metadata.empty:
        LOGGER.info("Loaded risk metadata with %s rows.", f"{len(risk_metadata):,}")
    cov_files = discover_covariance_files(risk_dir)
    months = build_month_sequence(predictions, cov_files, args.test_start, args.test_end)
    if not months:
        raise RuntimeError("No overlapping prediction and risk months found for requested test period.")
    LOGGER.info("Backtest month range: %s to %s (%s months).", months[0].date(), months[-1].date(), len(months))

    weights, backtest = run_backtest(months, predictions, returns, cov_files, fixed_lambda, fixed_tau, args)
    summary = compute_summary(backtest, len(months), fixed_lambda, fixed_tau)

    backtest.to_csv(outdir / "static_fixed_backtest.csv", index=False)
    weights.to_parquet(outdir / "static_fixed_weights.parquet", index=False, compression=PARQUET_COMPRESSION)
    pd.DataFrame([summary]).to_csv(outdir / "static_fixed_summary.csv", index=False)
    write_summary_text(outdir / "static_fixed_summary.txt", summary)
    plot_series(backtest, "cumulative_nav", "Static Fixed-Parameter Cumulative NAV", "Cumulative NAV", outdir / "static_fixed_cumret.png")
    plot_series(backtest, "turnover", "Static Fixed-Parameter Monthly Turnover", "L1 turnover", outdir / "static_fixed_turnover.png")
    plot_series(backtest, "n_assets", "Static Fixed-Parameter Universe Size", "Number of assets", outdir / "static_fixed_n_assets.png")

    LOGGER.info("Done. Annualized return %.4f, Sharpe %.4f.", summary["annualized_return"], summary["sharpe_ratio"])


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
