#!/usr/bin/env python3
"""Run a static mean-variance allocator baseline with turnover costs.

For each month t, this script uses only month-t expected returns and covariance
estimates to choose a long-only fully-invested portfolio, then evaluates the
realized portfolio return over month t+1. The optimizer has fixed risk and
turnover aversion parameters, so this is a no-RL baseline.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("run_static_allocator_baseline_v1")

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


def parse_bool(value: str | bool) -> bool:
    """Parse true/false command-line values."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value!r}.")


def parse_month(value: str | None) -> pd.Timestamp | None:
    """Parse an optional YYYY-MM month into a month-end timestamp."""
    if value is None:
        return None
    try:
        return pd.Timestamp(value) + MonthEnd(0)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Expected month formatted as YYYY-MM, got {value!r}."
        ) from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a no-RL static monthly portfolio allocator baseline."
    )
    parser.add_argument("--project-root", default=".", help="Project root. Default: .")
    parser.add_argument(
        "--pred-file",
        default="data/prediction/fm_oos_predictions.parquet",
        help="Prediction parquet path relative to project root unless absolute.",
    )
    parser.add_argument(
        "--risk-dir",
        default="data/risk/risk_cov_npz",
        help="Risk covariance NPZ directory relative to project root unless absolute.",
    )
    parser.add_argument(
        "--risk-meta-file",
        default="data/risk/risk_monthly_metadata.csv",
        help="Risk metadata CSV path relative to project root unless absolute.",
    )
    parser.add_argument(
        "--returns-file",
        default="data/panel/monthly_stock_panel_basic_full.parquet",
        help="Realized return panel path relative to project root unless absolute.",
    )
    parser.add_argument(
        "--outdir",
        default="data/allocator",
        help="Output directory relative to project root unless absolute.",
    )
    parser.add_argument(
        "--lambda-risk",
        type=float,
        default=10.0,
        help="Fixed risk-aversion coefficient. Default: 10.0.",
    )
    parser.add_argument(
        "--tau-turnover",
        type=float,
        default=0.001,
        help="Fixed L1 turnover-penalty coefficient. Default: 0.001.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Proportional transaction cost in basis points. Default: 10.",
    )
    parser.add_argument(
        "--solver",
        default="CLARABEL",
        choices=SOLVER_CHOICES,
        help="cvxpy solver to use. Default: CLARABEL.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help="Optional per-asset maximum portfolio weight.",
    )
    parser.add_argument(
        "--start-month",
        type=parse_month,
        default=None,
        help="Optional first rebalance month formatted YYYY-MM.",
    )
    parser.add_argument(
        "--end-month",
        type=parse_month,
        default=None,
        help="Optional last rebalance month formatted YYYY-MM.",
    )
    parser.add_argument(
        "--warm-start",
        type=parse_bool,
        default=True,
        help="Whether to warm-start cvxpy with prior weights. Default: true.",
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


def resolve_path(project_root: Path, path_like: str) -> Path:
    """Resolve a CLI path relative to the project root unless already absolute."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def require_columns(df: pd.DataFrame, columns: Iterable[str], dataset_name: str) -> None:
    """Validate that required columns exist."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def ensure_pyarrow_available() -> None:
    """Fail early when parquet support is unavailable."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet IO in the existing environment."
        ) from exc
    LOGGER.info("Using pyarrow %s for parquet IO.", pa.__version__)


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
    """Read risk metadata if present; keep the run robust if it is not needed."""
    if not path.exists():
        LOGGER.warning("Risk metadata file not found: %s", path)
        return pd.DataFrame()
    LOGGER.info("Loading risk metadata from %s.", path)
    metadata = pd.read_csv(path)
    if "month_end" in metadata.columns:
        metadata = normalize_month_end(metadata, "risk metadata")
    return metadata


def clean_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate expected-return forecasts."""
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
    LOGGER.info("Kept %s of %s prediction rows after cleaning.", f"{len(result):,}", f"{before:,}")
    return result.sort_values(["month_end", "permno"]).reset_index(drop=True)


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate realized returns."""
    require_columns(returns, ["permno", "month_end", "retadj"], "return panel")
    result = normalize_month_end(returns, "return panel")
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce")
    result["retadj"] = pd.to_numeric(result["retadj"], errors="coerce")
    before = len(result)
    result = result.dropna(subset=["permno", "month_end"]).copy()
    result["permno"] = result["permno"].astype("int64")
    exact_duplicates = int(result.duplicated(["permno", "month_end", "retadj"]).sum())
    if exact_duplicates:
        LOGGER.warning("Dropping %s exact duplicate return rows.", f"{exact_duplicates:,}")
        result = result.drop_duplicates(["permno", "month_end", "retadj"], keep="last")
    duplicate_keys = int(result.duplicated(["permno", "month_end"]).sum())
    if duplicate_keys:
        raise ValueError(
            f"Return panel has {duplicate_keys:,} duplicate (permno, month_end) keys "
            "after cleaning; expected one row per stock-month."
        )
    LOGGER.info("Kept %s of %s return rows after cleaning.", f"{len(result):,}", f"{before:,}")
    return result[["permno", "month_end", "retadj"]].sort_values(["month_end", "permno"])


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
    """Load one covariance NPZ and validate basic shape contracts."""
    with np.load(path, allow_pickle=True) as payload:
        required = {"cov", "permnos"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
        cov = np.asarray(payload["cov"], dtype=float)
        permnos = np.asarray(payload["permnos"]).astype("int64")
        vols = np.asarray(payload["vols"], dtype=float) if "vols" in payload.files else None
        if "month_end" in payload.files:
            month_values = np.asarray(payload["month_end"]).reshape(-1)
            month_end = pd.to_datetime(month_values[0]) + MonthEnd(0)
        else:
            month_end = fallback_month

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{path} covariance must be a square matrix; got {cov.shape}.")
    if cov.shape[0] != len(permnos):
        raise ValueError(
            f"{path} covariance dimension {cov.shape[0]} does not match "
            f"{len(permnos)} permnos."
        )
    if vols is not None and len(vols) != len(permnos):
        LOGGER.warning("%s vols length does not match permnos; ignoring vols.", path.name)
        vols = None

    cov = 0.5 * (cov + cov.T)
    return CovarianceData(month_end=pd.Timestamp(month_end), cov=cov, permnos=permnos, vols=vols)


def build_month_sequence(
    predictions: pd.DataFrame,
    cov_files: dict[pd.Timestamp, Path],
    start_month: pd.Timestamp | None,
    end_month: pd.Timestamp | None,
) -> list[pd.Timestamp]:
    """Find the sorted rebalance months available in both predictions and risk files."""
    prediction_months = set(pd.Series(predictions["month_end"].unique()).map(pd.Timestamp))
    risk_months = set(cov_files)
    months = sorted(prediction_months.intersection(risk_months))
    if start_month is not None:
        months = [month for month in months if month >= start_month]
    if end_month is not None:
        months = [month for month in months if month <= end_month]
    return months


def align_month_inputs(
    month_end: pd.Timestamp,
    pred_month: pd.DataFrame,
    cov_data: CovarianceData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Align month-t mu and covariance in the covariance file's permno order."""
    pred_lookup = pred_month.set_index("permno")["mu_hat"]
    keep_mask = np.array(
        [permno in pred_lookup.index for permno in cov_data.permnos],
        dtype=bool,
    )
    selected_positions = np.flatnonzero(keep_mask)
    selected_permnos = cov_data.permnos[selected_positions]
    mu = pred_lookup.reindex(selected_permnos).to_numpy(dtype=float)
    cov = cov_data.cov[np.ix_(selected_positions, selected_positions)]
    vols = cov_data.vols[selected_positions] if cov_data.vols is not None else None

    if not np.isfinite(mu).all():
        raise ValueError(f"{month_end.date()} aligned mu contains non-finite values.")
    if not np.isfinite(cov).all():
        raise ValueError(f"{month_end.date()} aligned covariance contains non-finite values.")
    return selected_permnos, mu, cov, selected_positions, vols


def align_previous_weights(
    current_permnos: np.ndarray,
    previous_weights: pd.Series | None,
) -> np.ndarray:
    """Align previous holdings to the current optimizer universe."""
    if previous_weights is None or previous_weights.empty:
        return np.repeat(1.0 / len(current_permnos), len(current_permnos))
    return previous_weights.reindex(current_permnos).fillna(0.0).to_numpy(dtype=float)


def solve_allocator(
    mu: np.ndarray,
    cov: np.ndarray,
    previous_weights_current: np.ndarray,
    lambda_risk: float,
    tau_turnover: float,
    solver: str,
    max_weight: float | None,
    warm_start: bool,
) -> OptimizationResult:
    """Solve the long-only fully-invested mean-variance allocation problem."""
    n_assets = len(mu)
    w = cp.Variable(n_assets)
    objective = cp.Maximize(
        mu @ w
        - lambda_risk * cp.quad_form(w, cp.psd_wrap(cov))
        - tau_turnover * cp.norm1(w - previous_weights_current)
    )
    constraints = [w >= 0, cp.sum(w) == 1]
    if max_weight is not None:
        constraints.append(w <= max_weight)

    if warm_start:
        initial = np.clip(previous_weights_current, 0.0, None)
        if initial.sum() > 0:
            w.value = initial / initial.sum()
        else:
            w.value = np.repeat(1.0 / n_assets, n_assets)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=solver, warm_start=warm_start)
    except Exception as exc:
        LOGGER.warning("Solver %s failed with exception: %s", solver, exc)
        return OptimizationResult(status=f"solver_exception: {exc}", objective_value=np.nan, weights=None)

    status = str(problem.status)
    if status not in SUCCESS_STATUSES or w.value is None:
        return OptimizationResult(status=status, objective_value=np.nan, weights=None)

    weights = np.asarray(w.value, dtype=float).reshape(-1)
    weights[np.abs(weights) < 1e-12] = 0.0
    weights = np.clip(weights, 0.0, None)
    weight_sum = weights.sum()
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        return OptimizationResult(status=f"{status}_invalid_weights", objective_value=np.nan, weights=None)
    weights = weights / weight_sum
    return OptimizationResult(status=status, objective_value=float(problem.value), weights=weights)


def compute_turnover(new_weights: pd.Series, previous_weights: pd.Series | None) -> float:
    """Compute full L1 turnover on the union of old and new holdings."""
    if previous_weights is None or previous_weights.empty:
        previous_weights = pd.Series(0.0, index=new_weights.index)
    union = new_weights.index.union(previous_weights.index)
    diff = new_weights.reindex(union).fillna(0.0) - previous_weights.reindex(union).fillna(0.0)
    return float(np.abs(diff.to_numpy(dtype=float)).sum())


def evaluate_next_return(
    month_end: pd.Timestamp,
    weights: pd.Series,
    returns_next: pd.DataFrame,
) -> float | None:
    """Evaluate realized t+1 return, renormalizing if return coverage is partial."""
    returns_lookup = returns_next.set_index("permno")["retadj"]
    realized = returns_lookup.reindex(weights.index)
    observed_mask = realized.notna()
    if not observed_mask.any():
        return None
    if not observed_mask.all():
        missing_weight = float(weights.loc[~observed_mask].sum())
        LOGGER.warning(
            "%s next-month returns are partially missing for optimizer holdings; "
            "renormalizing over observed subset for return evaluation. Missing weight: %.6f",
            month_end.date(),
            missing_weight,
        )
    eval_weights = weights.loc[observed_mask].astype(float)
    eval_weight_sum = float(eval_weights.sum())
    if eval_weight_sum <= 0 or not np.isfinite(eval_weight_sum):
        return None
    eval_weights = eval_weights / eval_weight_sum
    eval_returns = realized.loc[observed_mask].astype(float)
    return float(eval_weights.to_numpy() @ eval_returns.to_numpy())


def annualized_return(monthly_returns: pd.Series) -> float:
    """Compute annualized geometric return from monthly returns."""
    clean = monthly_returns.dropna()
    if clean.empty:
        return np.nan
    cumulative = float((1.0 + clean).prod())
    return cumulative ** (12.0 / len(clean)) - 1.0


def max_drawdown(nav: pd.Series) -> float:
    """Compute max drawdown from a cumulative NAV series."""
    clean = nav.dropna()
    if clean.empty:
        return np.nan
    running_max = clean.cummax()
    drawdowns = clean / running_max - 1.0
    return float(drawdowns.min())


def compute_summary(
    backtest: pd.DataFrame,
    attempted_months: int,
    lambda_risk: float,
    tau_turnover: float,
    cost_bps: float,
) -> dict[str, float | str]:
    """Compute scalar performance and run diagnostics."""
    successes = backtest.loc[backtest["solver_status"].isin(SUCCESS_STATUSES)].copy()
    net_returns = successes["net_return"].astype(float) if not successes.empty else pd.Series(dtype=float)
    gross_returns = successes["gross_return"].astype(float) if not successes.empty else pd.Series(dtype=float)
    net_vol = float(net_returns.std(ddof=1) * np.sqrt(12.0)) if len(net_returns) > 1 else np.nan
    sharpe = (
        float((net_returns.mean() / net_returns.std(ddof=1)) * np.sqrt(12.0))
        if len(net_returns) > 1 and net_returns.std(ddof=1) > 0
        else np.nan
    )
    cumulative_net = float(successes["cumulative_nav"].iloc[-1] - 1.0) if not successes.empty else np.nan
    date_range = (
        f"{successes['month_end'].min().date()} to {successes['month_end'].max().date()}"
        if not successes.empty
        else "n/a"
    )
    return {
        "date_range": date_range,
        "lambda_risk": lambda_risk,
        "tau_turnover": tau_turnover,
        "cost_bps": cost_bps,
        "attempted_months": attempted_months,
        "successful_months": int(len(successes)),
        "processed_months": int(len(successes)),
        "mean_n_assets": float(successes["n_assets"].mean()) if not successes.empty else np.nan,
        "mean_turnover": float(successes["turnover"].mean()) if not successes.empty else np.nan,
        "mean_monthly_gross_return": float(gross_returns.mean()) if not gross_returns.empty else np.nan,
        "mean_monthly_net_return": float(net_returns.mean()) if not net_returns.empty else np.nan,
        "annualized_gross_return": annualized_return(gross_returns),
        "annualized_net_return": annualized_return(net_returns),
        "annualized_net_volatility": net_vol,
        "sharpe_ratio": sharpe,
        "cumulative_net_return": cumulative_net,
        "max_drawdown": max_drawdown(successes["cumulative_nav"]) if not successes.empty else np.nan,
        "fraction_successfully_solved": float(len(successes) / attempted_months) if attempted_months else np.nan,
    }


def write_summary(path: Path, summary: dict[str, float | str]) -> None:
    """Write a human-readable summary text file."""
    lines = [
        "Static Allocator Baseline v1 Summary",
        "",
        f"Date range: {summary['date_range']}",
        f"lambda-risk: {summary['lambda_risk']}",
        f"tau-turnover: {summary['tau_turnover']}",
        f"cost-bps: {summary['cost_bps']}",
        f"Number of attempted months: {summary['attempted_months']:,}",
        f"Number of successful months: {summary['successful_months']:,}",
        f"Mean n_assets: {summary['mean_n_assets']:.2f}",
        f"Mean turnover: {summary['mean_turnover']:.6f}",
        f"Mean monthly gross return: {summary['mean_monthly_gross_return']:.6f}",
        f"Mean monthly net return: {summary['mean_monthly_net_return']:.6f}",
        f"Annualized gross return: {summary['annualized_gross_return']:.6f}",
        f"Annualized net return: {summary['annualized_net_return']:.6f}",
        f"Annualized net volatility: {summary['annualized_net_volatility']:.6f}",
        f"Sharpe ratio: {summary['sharpe_ratio']:.6f}",
        f"Cumulative net return: {summary['cumulative_net_return']:.6f}",
        f"Max drawdown: {summary['max_drawdown']:.6f}",
        f"Fraction of months successfully solved: {summary['fraction_successfully_solved']:.6f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_series(df: pd.DataFrame, y_col: str, title: str, ylabel: str, path: Path) -> None:
    """Save a simple time-series plot."""
    if df.empty:
        LOGGER.warning("Skipping plot %s because backtest table is empty.", path.name)
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
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the month-by-month allocator backtest."""
    pred_by_month = {month: df for month, df in predictions.groupby("month_end", sort=False)}
    returns_by_month = {month: df for month, df in returns.groupby("month_end", sort=False)}
    previous_weights: pd.Series | None = None
    nav = 1.0
    weight_rows: list[pd.DataFrame] = []
    backtest_rows: list[dict[str, object]] = []

    for month_end in months:
        month_end = pd.Timestamp(month_end)
        next_month = month_end + MonthEnd(1)
        cov_path = cov_files.get(month_end)
        pred_month = pred_by_month.get(month_end)
        returns_next = returns_by_month.get(next_month)

        if cov_path is None:
            LOGGER.warning("Skipping %s: covariance file is missing.", month_end.date())
            continue
        if pred_month is None or pred_month.empty:
            LOGGER.warning("Skipping %s: no prediction cross section.", month_end.date())
            continue
        if returns_next is None or returns_next["retadj"].notna().sum() == 0:
            LOGGER.warning("Skipping %s: no realized returns for %s.", month_end.date(), next_month.date())
            continue

        try:
            cov_data = load_covariance_npz(cov_path, month_end)
            permnos, mu, cov, _, vols = align_month_inputs(month_end, pred_month, cov_data)
        except Exception as exc:
            LOGGER.warning("Skipping %s: failed to align inputs: %s", month_end.date(), exc)
            continue

        n_assets = len(permnos)
        if n_assets < 2:
            LOGGER.warning("Skipping %s: only %s assets remain after alignment.", month_end.date(), n_assets)
            continue
        if args.max_weight is not None and args.max_weight * n_assets < 1.0 - 1e-10:
            LOGGER.warning(
                "Skipping %s: max-weight %.6f is infeasible for %s assets.",
                month_end.date(),
                args.max_weight,
                n_assets,
            )
            continue

        previous_current = align_previous_weights(permnos, previous_weights)
        result = solve_allocator(
            mu=mu,
            cov=cov,
            previous_weights_current=previous_current,
            lambda_risk=args.lambda_risk,
            tau_turnover=args.tau_turnover,
            solver=args.solver,
            max_weight=args.max_weight,
            warm_start=args.warm_start,
        )

        if result.weights is None:
            LOGGER.warning("%s solver did not produce usable weights: %s", month_end.date(), result.status)
            backtest_rows.append(
                {
                    "month_end": month_end,
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
        prior_for_accounting = previous_weights
        if prior_for_accounting is None or prior_for_accounting.empty:
            prior_for_accounting = pd.Series(
                np.repeat(1.0 / n_assets, n_assets),
                index=pd.Index(permnos, name="permno"),
            )
        turnover = compute_turnover(new_weights, prior_for_accounting)
        gross_return = evaluate_next_return(month_end, new_weights, returns_next)
        if gross_return is None:
            LOGGER.warning("Skipping %s: no usable realized returns after weight alignment.", month_end.date())
            continue
        cost = (args.cost_bps / 10000.0) * turnover
        net_return = gross_return - cost
        nav *= 1.0 + net_return

        month_weights = pd.DataFrame(
            {
                "month_end": month_end,
                "permno": permnos.astype("int64"),
                "weight": result.weights,
                "mu_hat": mu,
                "vol_hat": vols if vols is not None else np.nan,
                "in_optimizer_universe": True,
            }
        )
        weight_rows.append(month_weights)
        backtest_rows.append(
            {
                "month_end": month_end,
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

        LOGGER.info(
            "%s solved: n=%s turnover=%.4f gross=%.4f net=%.4f nav=%.4f",
            month_end.date(),
            n_assets,
            turnover,
            gross_return,
            net_return,
            nav,
        )

    weights = (
        pd.concat(weight_rows, ignore_index=True)
        if weight_rows
        else pd.DataFrame(
            columns=["month_end", "permno", "weight", "mu_hat", "vol_hat", "in_optimizer_universe"]
        )
    )
    backtest = pd.DataFrame(
        backtest_rows,
        columns=[
            "month_end",
            "n_assets",
            "solver_status",
            "objective_value",
            "turnover",
            "gross_return",
            "cost",
            "net_return",
            "cumulative_nav",
        ],
    )
    if not backtest.empty:
        backtest = backtest.sort_values("month_end").reset_index(drop=True)
    if not weights.empty:
        weights = weights.sort_values(["month_end", "permno"]).reset_index(drop=True)
    return weights, backtest


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full static allocator workflow."""
    if args.lambda_risk < 0:
        raise ValueError("--lambda-risk must be nonnegative.")
    if args.tau_turnover < 0:
        raise ValueError("--tau-turnover must be nonnegative.")
    if args.cost_bps < 0:
        raise ValueError("--cost-bps must be nonnegative.")
    if args.max_weight is not None and not (0 < args.max_weight <= 1):
        raise ValueError("--max-weight must be in (0, 1] when provided.")
    if args.start_month is not None and args.end_month is not None and args.start_month > args.end_month:
        raise ValueError("--start-month must be less than or equal to --end-month.")

    project_root = Path(args.project_root).expanduser().resolve()
    pred_file = resolve_path(project_root, args.pred_file)
    risk_dir = resolve_path(project_root, args.risk_dir)
    risk_meta_file = resolve_path(project_root, args.risk_meta_file)
    returns_file = resolve_path(project_root, args.returns_file)
    outdir = resolve_path(project_root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_pyarrow_available()
    predictions = clean_predictions(read_parquet_file(pred_file, "prediction file"))
    returns = clean_returns(read_parquet_file(returns_file, "return panel"))
    risk_metadata = read_risk_metadata(risk_meta_file)
    cov_files = discover_covariance_files(risk_dir)
    months = build_month_sequence(predictions, cov_files, args.start_month, args.end_month)

    if risk_metadata.empty:
        LOGGER.info("Risk metadata unavailable or empty; proceeding with covariance NPZ files.")
    else:
        LOGGER.info("Loaded risk metadata with %s rows.", f"{len(risk_metadata):,}")
    if not months:
        raise RuntimeError("No overlapping prediction and risk months found for backtest.")
    LOGGER.info(
        "Backtest month range: %s to %s (%s attempted months).",
        months[0].date(),
        months[-1].date(),
        f"{len(months):,}",
    )

    weights, backtest = run_backtest(months, predictions, returns, cov_files, args)
    summary = compute_summary(
        backtest=backtest,
        attempted_months=len(months),
        lambda_risk=args.lambda_risk,
        tau_turnover=args.tau_turnover,
        cost_bps=args.cost_bps,
    )

    weights_path = outdir / "static_allocator_weights.parquet"
    backtest_path = outdir / "static_allocator_backtest.csv"
    summary_path = outdir / "static_allocator_summary.txt"
    LOGGER.info("Writing weights to %s.", weights_path)
    weights.to_parquet(weights_path, index=False, compression=PARQUET_COMPRESSION)
    LOGGER.info("Writing backtest table to %s.", backtest_path)
    backtest.to_csv(backtest_path, index=False)
    LOGGER.info("Writing summary to %s.", summary_path)
    write_summary(summary_path, summary)

    plot_series(
        backtest,
        "cumulative_nav",
        "Static Allocator Cumulative NAV",
        "Cumulative NAV",
        outdir / "static_allocator_cumret.png",
    )
    plot_series(
        backtest,
        "turnover",
        "Static Allocator Monthly Turnover",
        "L1 turnover",
        outdir / "static_allocator_turnover.png",
    )
    plot_series(
        backtest,
        "n_assets",
        "Static Allocator Optimizer Universe Size",
        "Number of assets",
        outdir / "static_allocator_n_assets.png",
    )

    LOGGER.info(
        "Done. Successful months: %s / %s. Annualized net return: %.4f. Sharpe: %.4f.",
        summary["successful_months"],
        summary["attempted_months"],
        summary["annualized_net_return"],
        summary["sharpe_ratio"],
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
