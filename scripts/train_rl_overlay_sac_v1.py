#!/usr/bin/env python3
"""Train a SAC overlay that chooses allocator hyperparameters monthly.

The RL policy does not choose portfolio weights directly. At each rebalance
month t it chooses risk-aversion and turnover-penalty parameters, then this
script solves the same long-only fully-invested convex allocation problem used
by the static allocator baseline and evaluates realized month t+1 returns.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import cvxpy as cp
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from pandas.tseries.offsets import MonthEnd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


LOGGER = logging.getLogger("train_rl_overlay_sac_v1")

PARQUET_COMPRESSION = "snappy"
SOLVER_CHOICES = ("CLARABEL", "ECOS", "SCS", "OSQP")
SUCCESS_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
LAMBDA_RANGE = (0.1, 50.0)
TAU_RANGE = (0.0, 0.01)
MISSING_RETURN_WEIGHT_WARN_TOL = 1e-8


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


@dataclass(frozen=True)
class MonthData:
    """Fully aligned inputs for one no-lookahead monthly decision."""

    month_end: pd.Timestamp
    next_month: pd.Timestamp
    permnos: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    vols: np.ndarray
    avg_corr: float
    returns_next: pd.Series
    base_state: np.ndarray


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
        description="Train and evaluate a SAC hyperparameter overlay for monthly allocation."
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
    parser.add_argument("--ff-file", default=None, help="Optional FF factor CSV/parquet file.")
    parser.add_argument(
        "--outdir",
        default="data/rl_overlay_sac",
        help="Output directory relative to project root unless absolute.",
    )
    parser.add_argument("--train-start", type=parse_month, default=parse_month("2006-01"))
    parser.add_argument("--train-end", type=parse_month, default=parse_month("2016-12"))
    parser.add_argument("--val-start", type=parse_month, default=parse_month("2017-01"))
    parser.add_argument("--val-end", type=parse_month, default=parse_month("2019-12"))
    parser.add_argument("--test-start", type=parse_month, default=parse_month("2020-01"))
    parser.add_argument("--test-end", type=parse_month, default=parse_month("2025-12"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--max-weight", type=float, default=None)
    parser.add_argument("--solver", default="CLARABEL", choices=SOLVER_CHOICES)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau-soft", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-frequency", type=int, default=5000)
    parser.add_argument(
        "--use-vecnormalize",
        type=parse_bool,
        choices=(True, False),
        default=True,
        help="Whether to normalize observations/rewards with VecNormalize. Default: true.",
    )
    parser.add_argument(
        "--warm-start-cvxpy",
        type=parse_bool,
        choices=(True, False),
        default=True,
        help="Whether to warm-start cvxpy from prior weights. Default: true.",
    )
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


def set_global_seed(seed: int) -> None:
    """Seed common pseudo-random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(project_root: Path, path_like: str | None) -> Path | None:
    """Resolve a CLI path relative to the project root unless already absolute."""
    if path_like is None:
        return None
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
    """Read risk metadata if present."""
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


def load_ff_factors(path: Path | None) -> pd.DataFrame:
    """Load optional FF factors and return month_end-indexed mktrf/smb/hml/rf."""
    columns = ["mktrf", "smb", "hml", "rf"]
    if path is None:
        return pd.DataFrame(columns=["month_end", *columns])
    if not path.exists():
        raise FileNotFoundError(f"FF factor file not found: {path}")
    LOGGER.info("Loading optional FF factor file from %s.", path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        factors = pd.read_parquet(path)
    else:
        factors = pd.read_csv(path)
    factors = factors.rename(columns={col: col.lower() for col in factors.columns})
    date_col = "month_end" if "month_end" in factors.columns else "date"
    if date_col not in factors.columns:
        raise ValueError("FF factor file must contain month_end or date.")
    factors = factors.rename(columns={date_col: "month_end"})
    factors = normalize_month_end(factors, "FF factor file")
    missing = sorted(set(columns).difference(factors.columns))
    if missing:
        raise ValueError(f"FF factor file is missing columns: {', '.join(missing)}")
    for col in columns:
        factors[col] = pd.to_numeric(factors[col], errors="coerce")
    factors = factors.dropna(subset=["month_end", *columns])
    factors = factors.drop_duplicates("month_end", keep="last")
    return factors[["month_end", *columns]].sort_values("month_end").reset_index(drop=True)


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
    return CovarianceData(
        month_end=pd.Timestamp(month_end),
        cov=0.5 * (cov + cov.T),
        permnos=permnos,
        vols=vols,
    )


def build_month_sequence(
    predictions: pd.DataFrame,
    cov_files: dict[pd.Timestamp, Path],
    start_month: pd.Timestamp | None,
    end_month: pd.Timestamp | None,
) -> list[pd.Timestamp]:
    """Find sorted rebalance months available in both predictions and risk files."""
    prediction_months = set(pd.Series(predictions["month_end"].unique()).map(pd.Timestamp))
    months = sorted(prediction_months.intersection(cov_files))
    if start_month is not None:
        months = [month for month in months if month >= start_month]
    if end_month is not None:
        months = [month for month in months if month <= end_month]
    return months


def align_month_inputs(
    month_end: pd.Timestamp,
    pred_month: pd.DataFrame,
    cov_data: CovarianceData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align month-t mu and covariance in the covariance file's permno order."""
    pred_lookup = pred_month.set_index("permno")["mu_hat"]
    keep_mask = np.array([permno in pred_lookup.index for permno in cov_data.permnos], dtype=bool)
    positions = np.flatnonzero(keep_mask)
    permnos = cov_data.permnos[positions]
    mu = pred_lookup.reindex(permnos).to_numpy(dtype=float)
    cov = cov_data.cov[np.ix_(positions, positions)]
    vols = cov_data.vols[positions] if cov_data.vols is not None else np.sqrt(np.clip(np.diag(cov), 0.0, None))
    if not np.isfinite(mu).all():
        raise ValueError(f"{month_end.date()} aligned mu contains non-finite values.")
    if not np.isfinite(cov).all():
        raise ValueError(f"{month_end.date()} aligned covariance contains non-finite values.")
    if not np.isfinite(vols).all():
        raise ValueError(f"{month_end.date()} aligned vols contain non-finite values.")
    return permnos, mu, cov, vols


def average_pairwise_correlation(cov: np.ndarray, vols: np.ndarray) -> float:
    """Compute average off-diagonal correlation implied by covariance."""
    n_assets = cov.shape[0]
    if n_assets < 2:
        return 0.0
    denom = np.outer(vols, vols)
    corr = np.divide(cov, denom, out=np.zeros_like(cov, dtype=float), where=denom > 0)
    corr = np.clip(corr, -1.0, 1.0)
    return float((corr.sum() - np.trace(corr)) / (n_assets * (n_assets - 1)))


def make_base_state(
    mu: np.ndarray,
    vols: np.ndarray,
    avg_corr: float,
    n_assets: int,
    n_assets_scale: float,
    ff_values: np.ndarray | None,
) -> np.ndarray:
    """Build fixed month-t summary features before previous-step features."""
    mu_features = np.array(
        [
            np.mean(mu),
            np.std(mu, ddof=0),
            np.percentile(mu, 10),
            np.percentile(mu, 50),
            np.percentile(mu, 90),
        ],
        dtype=np.float32,
    )
    vol_features = np.array(
        [
            np.mean(vols),
            np.std(vols, ddof=0),
            np.percentile(vols, 90),
            avg_corr,
        ],
        dtype=np.float32,
    )
    universe_feature = np.array([n_assets / max(n_assets_scale, 1.0)], dtype=np.float32)
    if ff_values is None:
        pieces = [mu_features, vol_features, universe_feature]
    else:
        pieces = [mu_features, vol_features, universe_feature, ff_values.astype(np.float32)]
    state = np.concatenate(pieces).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("Base state contains non-finite values.")
    return state


def build_monthly_dataset(
    months: list[pd.Timestamp],
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    cov_files: dict[pd.Timestamp, Path],
    n_assets_scale: float,
    ff_factors: pd.DataFrame,
) -> list[MonthData]:
    """Build fully aligned no-lookahead monthly records, skipping invalid months."""
    pred_by_month = {month: df for month, df in predictions.groupby("month_end", sort=False)}
    returns_by_month = {month: df for month, df in returns.groupby("month_end", sort=False)}
    ff_lookup = (
        ff_factors.set_index("month_end")[["mktrf", "smb", "hml", "rf"]]
        if not ff_factors.empty
        else None
    )
    dataset: list[MonthData] = []

    for month_end in months:
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
        if ff_lookup is not None and month_end not in ff_lookup.index:
            LOGGER.warning("Skipping %s: FF factor row is missing.", month_end.date())
            continue

        try:
            cov_data = load_covariance_npz(cov_path, month_end)
            permnos, mu, cov, vols = align_month_inputs(month_end, pred_month, cov_data)
            n_assets = len(permnos)
            if n_assets < 2:
                LOGGER.warning("Skipping %s: only %s assets remain after alignment.", month_end.date(), n_assets)
                continue
            avg_corr = average_pairwise_correlation(cov, vols)
            ff_values = None if ff_lookup is None else ff_lookup.loc[month_end].to_numpy(dtype=np.float32)
            base_state = make_base_state(mu, vols, avg_corr, n_assets, n_assets_scale, ff_values)
            returns_lookup = returns_next.set_index("permno")["retadj"].astype(float)
        except Exception as exc:
            LOGGER.warning("Skipping %s: failed to align monthly inputs: %s", month_end.date(), exc)
            continue

        dataset.append(
            MonthData(
                month_end=month_end,
                next_month=next_month,
                permnos=permnos,
                mu=mu,
                cov=cov,
                vols=vols,
                avg_corr=avg_corr,
                returns_next=returns_lookup,
                base_state=base_state,
            )
        )

    if not dataset:
        raise RuntimeError("No valid monthly records were built for the requested date range.")
    LOGGER.info(
        "Built %s valid monthly records from %s to %s.",
        f"{len(dataset):,}",
        dataset[0].month_end.date(),
        dataset[-1].month_end.date(),
    )
    return dataset


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
    return OptimizationResult(status=status, objective_value=float(problem.value), weights=weights / weight_sum)


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
    returns_next: pd.Series,
) -> float | None:
    """Evaluate realized t+1 return, renormalizing if return coverage is partial."""
    realized = returns_next.reindex(weights.index)
    observed_mask = realized.notna()
    if not observed_mask.any():
        return None
    if not observed_mask.all():
        missing_weight = float(weights.loc[~observed_mask].sum())
        missing_count = int((~observed_mask).sum())
        log_fn = (
            LOGGER.warning
            if missing_weight > MISSING_RETURN_WEIGHT_WARN_TOL
            else LOGGER.debug
        )
        log_fn(
            "%s next-month returns are partially missing for %s optimizer holdings; "
            "renormalizing over observed subset for return evaluation. Missing weight: %.12f",
            month_end.date(),
            f"{missing_count:,}",
            missing_weight,
        )
    eval_weights = weights.loc[observed_mask].astype(float)
    eval_weight_sum = float(eval_weights.sum())
    if eval_weight_sum <= 0 or not np.isfinite(eval_weight_sum):
        return None
    eval_weights = eval_weights / eval_weight_sum
    eval_returns = realized.loc[observed_mask].astype(float)
    return float(eval_weights.to_numpy() @ eval_returns.to_numpy())


class PortfolioHyperparamEnv(gym.Env):
    """Sequential monthly env where actions are allocator hyperparameters."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        monthly_data: list[MonthData],
        cost_bps: float,
        solver: str,
        max_weight: float | None,
        warm_start_cvxpy: bool,
        env_name: str,
        failure_reward: float = -1.0,
    ) -> None:
        super().__init__()
        if not monthly_data:
            raise ValueError("monthly_data must not be empty.")
        self.monthly_data = monthly_data
        self.cost_bps = float(cost_bps)
        self.solver = solver
        self.max_weight = max_weight
        self.warm_start_cvxpy = warm_start_cvxpy
        self.env_name = env_name
        self.failure_reward = float(failure_reward)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        state_dim = len(monthly_data[0].base_state) + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.idx = 0
        self.previous_weights: pd.Series | None = None
        self.previous_return = 0.0
        self.previous_turnover = 0.0
        self.previous_lambda = float(np.mean(LAMBDA_RANGE))
        self.previous_tau = float(np.mean(TAU_RANGE))
        self.nav = 1.0
        self.history: list[dict[str, Any]] = []
        self.weight_history: list[pd.DataFrame] = []
        self.action_history: list[dict[str, Any]] = []
        self.last_episode_history: list[dict[str, Any]] = []
        self.last_episode_weight_history: list[pd.DataFrame] = []
        self.last_episode_action_history: list[dict[str, Any]] = []

    @staticmethod
    def map_action(action: np.ndarray) -> tuple[float, float]:
        """Map raw Box[-1, 1]^2 action to lambda/tau ranges."""
        clipped = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        lambda_t = LAMBDA_RANGE[0] + (clipped[0] + 1.0) * 0.5 * (LAMBDA_RANGE[1] - LAMBDA_RANGE[0])
        tau_t = TAU_RANGE[0] + (clipped[1] + 1.0) * 0.5 * (TAU_RANGE[1] - TAU_RANGE[0])
        return float(lambda_t), float(tau_t)

    def _state(self) -> np.ndarray:
        month = self.monthly_data[min(self.idx, len(self.monthly_data) - 1)]
        prev_features = np.array(
            [
                self.previous_return,
                self.previous_turnover,
                self.previous_lambda,
                self.previous_tau,
            ],
            dtype=np.float32,
        )
        state = np.concatenate([month.base_state, prev_features]).astype(np.float32)
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to the first chronological month in this split."""
        super().reset(seed=seed)
        if self.history:
            self.last_episode_history = self.history
            self.last_episode_weight_history = self.weight_history
            self.last_episode_action_history = self.action_history
        self.idx = 0
        self.previous_weights = None
        self.previous_return = 0.0
        self.previous_turnover = 0.0
        self.previous_lambda = float(np.mean(LAMBDA_RANGE))
        self.previous_tau = float(np.mean(TAU_RANGE))
        self.nav = 1.0
        self.history = []
        self.weight_history = []
        self.action_history = []
        return self._state(), {"month_end": self.monthly_data[0].month_end}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Solve one month-t allocation and return log net t+1 reward."""
        month = self.monthly_data[self.idx]
        lambda_t, tau_t = self.map_action(action)
        action_row = {
            "month_end": month.month_end,
            "lambda_t": lambda_t,
            "tau_t": tau_t,
            "previous_lambda": self.previous_lambda,
            "previous_tau": self.previous_tau,
        }
        previous_current = align_previous_weights(month.permnos, self.previous_weights)
        if self.max_weight is not None and self.max_weight * len(month.permnos) < 1.0 - 1e-10:
            result = OptimizationResult("infeasible_max_weight", np.nan, None)
        else:
            result = solve_allocator(
                mu=month.mu,
                cov=month.cov,
                previous_weights_current=previous_current,
                lambda_risk=lambda_t,
                tau_turnover=tau_t,
                solver=self.solver,
                max_weight=self.max_weight,
                warm_start=self.warm_start_cvxpy,
            )

        reward = self.failure_reward
        gross_return = np.nan
        cost = np.nan
        net_return = np.nan
        turnover = np.nan
        weights_for_log: np.ndarray | None = None
        new_weights: pd.Series | None = None

        if result.weights is not None:
            new_weights = pd.Series(result.weights, index=pd.Index(month.permnos, name="permno"))
            prior_for_accounting = self.previous_weights
            if prior_for_accounting is None or prior_for_accounting.empty:
                prior_for_accounting = pd.Series(
                    np.repeat(1.0 / len(month.permnos), len(month.permnos)),
                    index=pd.Index(month.permnos, name="permno"),
                )
            turnover = compute_turnover(new_weights, prior_for_accounting)
            gross = evaluate_next_return(month.month_end, new_weights, month.returns_next)
            if gross is not None:
                gross_return = gross
                cost = (self.cost_bps / 10000.0) * turnover
                net_return = gross_return - cost
                if net_return <= -1.0:
                    reward = self.failure_reward
                else:
                    reward = float(np.log1p(net_return))
                self.nav *= 1.0 + net_return
                self.previous_weights = new_weights
                self.previous_return = float(net_return)
                self.previous_turnover = float(turnover)
                weights_for_log = result.weights
            else:
                result = OptimizationResult(f"{result.status}_no_usable_returns", result.objective_value, None)

        if weights_for_log is None:
            self.previous_return = 0.0
            self.previous_turnover = 0.0

        self.history.append(
            {
                "month_end": month.month_end,
                "lambda_t": lambda_t,
                "tau_t": tau_t,
                "n_assets": len(month.permnos),
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "reward": reward,
                "cumulative_nav": self.nav,
                "solver_status": result.status,
                "objective_value": result.objective_value,
            }
        )
        self.action_history.append(action_row)
        if weights_for_log is not None:
            self.weight_history.append(
                pd.DataFrame(
                    {
                        "month_end": month.month_end,
                        "permno": month.permnos.astype("int64"),
                        "weight": weights_for_log,
                        "lambda_t": lambda_t,
                        "tau_t": tau_t,
                        "mu_hat": month.mu,
                        "in_optimizer_universe": True,
                    }
                )
            )

        self.previous_lambda = lambda_t
        self.previous_tau = tau_t
        self.idx += 1
        terminated = self.idx >= len(self.monthly_data)
        next_state = self._state() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = self.history[-1].copy()
        info["is_success"] = result.status in SUCCESS_STATUSES
        return next_state, float(reward), terminated, False, info

    def get_backtest(self) -> pd.DataFrame:
        """Return current episode backtest table."""
        return pd.DataFrame(self.history if self.history else self.last_episode_history)

    def get_weights(self) -> pd.DataFrame:
        """Return current episode long-format weight table."""
        weight_history = self.weight_history if self.weight_history else self.last_episode_weight_history
        if not weight_history:
            return pd.DataFrame(
                columns=[
                    "month_end",
                    "permno",
                    "weight",
                    "lambda_t",
                    "tau_t",
                    "mu_hat",
                    "in_optimizer_universe",
                ]
            )
        return pd.concat(weight_history, ignore_index=True)

    def get_action_history(self) -> pd.DataFrame:
        """Return current episode action history."""
        return pd.DataFrame(self.action_history if self.action_history else self.last_episode_action_history)


def make_env(
    monthly_data: list[MonthData],
    args: argparse.Namespace,
    env_name: str,
    seed: int,
) -> gym.Env:
    """Build a monitored environment instance."""
    env = PortfolioHyperparamEnv(
        monthly_data=monthly_data,
        cost_bps=args.cost_bps,
        solver=args.solver,
        max_weight=args.max_weight,
        warm_start_cvxpy=args.warm_start_cvxpy,
        env_name=env_name,
    )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return Monitor(env)


def unwrap_portfolio_env(env: Any) -> PortfolioHyperparamEnv:
    """Find the underlying PortfolioHyperparamEnv inside wrappers."""
    current = env
    while hasattr(current, "env"):
        current = current.env
    if not isinstance(current, PortfolioHyperparamEnv):
        raise TypeError(f"Could not unwrap PortfolioHyperparamEnv from {type(env)}.")
    return current


def sync_vecnormalize(source: VecNormalize, target: VecNormalize) -> None:
    """Copy train VecNormalize statistics into an eval VecNormalize wrapper."""
    target.obs_rms = source.obs_rms
    target.ret_rms = source.ret_rms
    target.clip_obs = source.clip_obs
    target.clip_reward = source.clip_reward
    target.gamma = source.gamma
    target.epsilon = source.epsilon
    target.training = False
    target.norm_reward = False


def make_vec_env(
    monthly_data: list[MonthData],
    args: argparse.Namespace,
    env_name: str,
    seed: int,
    vecnormalize_source: VecNormalize | None = None,
    training: bool = True,
) -> DummyVecEnv | VecNormalize:
    """Create a DummyVecEnv, optionally wrapped with VecNormalize."""
    vec_env: DummyVecEnv | VecNormalize = DummyVecEnv(
        [lambda: make_env(monthly_data, args, env_name=env_name, seed=seed)]
    )
    if args.use_vecnormalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=training, clip_obs=10.0)
        if vecnormalize_source is not None:
            sync_vecnormalize(vecnormalize_source, vec_env)
    return vec_env


def get_single_portfolio_env(vec_env: DummyVecEnv | VecNormalize) -> PortfolioHyperparamEnv:
    """Return the single underlying portfolio env from a vectorized env."""
    base = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    return unwrap_portfolio_env(base.envs[0])


def run_policy_episode(
    model: SAC,
    vec_env: DummyVecEnv | VecNormalize,
    deterministic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one full chronological episode and collect logs."""
    obs = vec_env.reset()
    done = np.array([False])
    while not bool(done[0]):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, done, _ = vec_env.step(action)
    env = get_single_portfolio_env(vec_env)
    return env.get_backtest(), env.get_weights(), env.get_action_history()


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
    return float((clean / running_max - 1.0).min())


def compute_metrics(backtest: pd.DataFrame) -> dict[str, float]:
    """Compute validation/test performance metrics from a backtest table."""
    if backtest.empty:
        return {
            "mean_monthly_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "cumulative_return": np.nan,
            "fraction_solved": np.nan,
        }
    successful = backtest.loc[backtest["solver_status"].isin(SUCCESS_STATUSES)].copy()
    returns = successful["net_return"].astype(float).dropna()
    vol = float(returns.std(ddof=1) * np.sqrt(12.0)) if len(returns) > 1 else np.nan
    sharpe = (
        float((returns.mean() / returns.std(ddof=1)) * np.sqrt(12.0))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    cumulative = float((1.0 + returns).prod() - 1.0) if not returns.empty else np.nan
    return {
        "mean_monthly_return": float(returns.mean()) if not returns.empty else np.nan,
        "annualized_return": annualized_return(returns),
        "annualized_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(successful["cumulative_nav"]) if not successful.empty else np.nan,
        "cumulative_return": cumulative,
        "fraction_solved": float(len(successful) / len(backtest)) if len(backtest) else np.nan,
    }


class ValidationCallback(BaseCallback):
    """Evaluate deterministic validation performance and save the best model."""

    def __init__(
        self,
        val_data: list[MonthData],
        args: argparse.Namespace,
        train_vec_env: DummyVecEnv | VecNormalize,
        models_dir: Path,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.val_data = val_data
        self.args = args
        self.train_vec_env = train_vec_env
        self.models_dir = models_dir
        self.best_sharpe = -np.inf
        self.rows: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.args.eval_frequency != 0:
            return True
        source = self.train_vec_env if isinstance(self.train_vec_env, VecNormalize) else None
        val_env = make_vec_env(
            self.val_data,
            self.args,
            env_name="validation",
            seed=self.args.seed + 1000 + self.n_calls,
            vecnormalize_source=source,
            training=False,
        )
        try:
            backtest, _, _ = run_policy_episode(self.model, val_env, deterministic=True)
            metrics = compute_metrics(backtest)
            row = {"timestep": int(self.num_timesteps), **metrics}
            self.rows.append(row)
            sharpe = metrics["sharpe"]
            LOGGER.info(
                "Validation at step %s: Sharpe=%.4f ann_ret=%.4f cum_ret=%.4f",
                self.num_timesteps,
                sharpe,
                metrics["annualized_return"],
                metrics["cumulative_return"],
            )
            if np.isfinite(sharpe) and sharpe > self.best_sharpe:
                self.best_sharpe = float(sharpe)
                self.model.save(self.models_dir / "best_model")
                if isinstance(self.train_vec_env, VecNormalize):
                    self.train_vec_env.save(self.models_dir / "best_vecnormalize.pkl")
                LOGGER.info("Saved new best model at step %s.", self.num_timesteps)
        finally:
            val_env.close()
        return True


def write_train_history(path: Path, model: SAC) -> None:
    """Write lightweight training progress information."""
    rows = [{"total_timesteps": int(model.num_timesteps)}]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary(
    path: Path,
    backtest: pd.DataFrame,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    """Write a human-readable test summary."""
    if backtest.empty:
        date_range = "n/a"
        total_months = 0
        mean_turnover = np.nan
        mean_lambda = np.nan
        mean_tau = np.nan
    else:
        date_range = f"{backtest['month_end'].min().date()} to {backtest['month_end'].max().date()}"
        total_months = len(backtest)
        mean_turnover = float(backtest["turnover"].mean())
        mean_lambda = float(backtest["lambda_t"].mean())
        mean_tau = float(backtest["tau_t"].mean())
    lines = [
        "RL Overlay SAC v1 Test Summary",
        "",
        f"Date range: {date_range}",
        f"Action ranges: lambda_t in [{LAMBDA_RANGE[0]}, {LAMBDA_RANGE[1]}], tau_t in [{TAU_RANGE[0]}, {TAU_RANGE[1]}]",
        f"cost-bps: {args.cost_bps}",
        f"Total test months: {total_months:,}",
        f"Fraction solved: {metrics['fraction_solved']:.6f}",
        f"Annualized net return: {metrics['annualized_return']:.6f}",
        f"Annualized net vol: {metrics['annualized_vol']:.6f}",
        f"Sharpe ratio: {metrics['sharpe']:.6f}",
        f"Max drawdown: {metrics['max_drawdown']:.6f}",
        f"Mean turnover: {mean_turnover:.6f}",
        f"Mean lambda: {mean_lambda:.6f}",
        f"Mean tau: {mean_tau:.6f}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_series(df: pd.DataFrame, y_col: str, title: str, ylabel: str, path: Path) -> None:
    """Save a simple time-series plot."""
    if df.empty or y_col not in df.columns:
        LOGGER.warning("Skipping plot %s because source table is empty or missing %s.", path.name, y_col)
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


def validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations before doing expensive work."""
    if args.cost_bps < 0:
        raise ValueError("--cost-bps must be nonnegative.")
    if args.max_weight is not None and not (0 < args.max_weight <= 1):
        raise ValueError("--max-weight must be in (0, 1] when provided.")
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive.")
    if args.eval_frequency <= 0:
        raise ValueError("--eval-frequency must be positive.")
    for left, right, name in [
        (args.train_start, args.train_end, "train"),
        (args.val_start, args.val_end, "validation"),
        (args.test_start, args.test_end, "test"),
    ]:
        if left is not None and right is not None and left > right:
            raise ValueError(f"--{name}-start must be less than or equal to --{name}-end.")
    if args.train_end is not None and args.val_start is not None and args.train_end >= args.val_start:
        raise ValueError("--train-end must be strictly before --val-start to avoid split leakage.")
    if args.val_end is not None and args.test_start is not None and args.val_end >= args.test_start:
        raise ValueError("--val-end must be strictly before --test-start to avoid split leakage.")


def run_pipeline(args: argparse.Namespace) -> None:
    """Run data prep, SAC training, model selection, and test evaluation."""
    validate_args(args)
    set_global_seed(args.seed)

    project_root = Path(args.project_root).expanduser().resolve()
    pred_file = resolve_path(project_root, args.pred_file)
    risk_dir = resolve_path(project_root, args.risk_dir)
    risk_meta_file = resolve_path(project_root, args.risk_meta_file)
    returns_file = resolve_path(project_root, args.returns_file)
    ff_file = resolve_path(project_root, args.ff_file)
    outdir = resolve_path(project_root, args.outdir)
    if outdir is None:
        raise ValueError("--outdir must not be empty.")
    models_dir = outdir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ensure_pyarrow_available()
    predictions = clean_predictions(read_parquet_file(pred_file, "prediction file"))
    returns = clean_returns(read_parquet_file(returns_file, "return panel"))
    risk_metadata = read_risk_metadata(risk_meta_file)
    cov_files = discover_covariance_files(risk_dir)
    ff_factors = load_ff_factors(ff_file)
    if risk_metadata.empty:
        LOGGER.info("Risk metadata unavailable or empty; proceeding with covariance NPZ files.")
    else:
        LOGGER.info("Loaded risk metadata with %s rows.", f"{len(risk_metadata):,}")

    train_months = build_month_sequence(predictions, cov_files, args.train_start, args.train_end)
    val_months = build_month_sequence(predictions, cov_files, args.val_start, args.val_end)
    test_months = build_month_sequence(predictions, cov_files, args.test_start, args.test_end)
    if not train_months or not val_months or not test_months:
        raise RuntimeError("Train, validation, and test splits must all have overlapping prediction/risk months.")

    train_max_assets = max(
        int(load_covariance_npz(cov_files[month], month).cov.shape[0])
        for month in train_months
        if month in cov_files
    )
    train_data = build_monthly_dataset(train_months, predictions, returns, cov_files, train_max_assets, ff_factors)
    val_data = build_monthly_dataset(val_months, predictions, returns, cov_files, train_max_assets, ff_factors)
    test_data = build_monthly_dataset(test_months, predictions, returns, cov_files, train_max_assets, ff_factors)

    LOGGER.info(
        "Split sizes after skipping invalid months: train=%s validation=%s test=%s.",
        len(train_data),
        len(val_data),
        len(test_data),
    )

    train_env = make_vec_env(train_data, args, env_name="train", seed=args.seed, training=True)
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau_soft,
        seed=args.seed,
        verbose=1,
    )

    callback = ValidationCallback(
        val_data=val_data,
        args=args,
        train_vec_env=train_env,
        models_dir=models_dir,
        verbose=1,
    )
    LOGGER.info("Starting SAC training for %s timesteps.", f"{args.total_timesteps:,}")
    model.learn(total_timesteps=args.total_timesteps, callback=callback, progress_bar=False)
    model.save(models_dir / "final_model")
    if isinstance(train_env, VecNormalize):
        train_env.save(models_dir / "final_vecnormalize.pkl")
    write_train_history(outdir / "train_history.csv", model)

    if not callback.rows:
        LOGGER.info("Running final validation because no periodic validation point was reached.")
        source = train_env if isinstance(train_env, VecNormalize) else None
        val_env = make_vec_env(
            val_data,
            args,
            env_name="validation_final",
            seed=args.seed + 3000,
            vecnormalize_source=source,
            training=False,
        )
        try:
            val_backtest, _, _ = run_policy_episode(model, val_env, deterministic=True)
            metrics = compute_metrics(val_backtest)
            callback.rows.append({"timestep": int(model.num_timesteps), **metrics})
            if np.isfinite(metrics["sharpe"]):
                callback.best_sharpe = float(metrics["sharpe"])
                model.save(models_dir / "best_model")
                if isinstance(train_env, VecNormalize):
                    train_env.save(models_dir / "best_vecnormalize.pkl")
        finally:
            val_env.close()

    validation_metrics = pd.DataFrame(callback.rows)
    validation_metrics.to_csv(outdir / "validation_metrics.csv", index=False)
    if not (models_dir / "best_model.zip").exists():
        LOGGER.warning("No finite-Sharpe validation checkpoint was selected; using final model as best.")
        model.save(models_dir / "best_model")
        if isinstance(train_env, VecNormalize):
            train_env.save(models_dir / "best_vecnormalize.pkl")

    if args.use_vecnormalize and (models_dir / "best_vecnormalize.pkl").exists():
        raw_test_env = DummyVecEnv(
            [lambda: make_env(test_data, args, env_name="test", seed=args.seed + 2000)]
        )
        test_env: DummyVecEnv | VecNormalize = VecNormalize.load(
            models_dir / "best_vecnormalize.pkl",
            raw_test_env,
        )
        test_env.training = False
        test_env.norm_reward = False
    else:
        test_env = make_vec_env(test_data, args, env_name="test", seed=args.seed + 2000, training=False)

    best_model = SAC.load(models_dir / "best_model", env=test_env, seed=args.seed)
    LOGGER.info("Evaluating best checkpoint once on the untouched test split.")
    test_backtest, test_weights, test_actions = run_policy_episode(best_model, test_env, deterministic=True)
    test_metrics = compute_metrics(test_backtest)

    if not test_backtest.empty:
        test_backtest = test_backtest[
            [
                "month_end",
                "lambda_t",
                "tau_t",
                "n_assets",
                "turnover",
                "gross_return",
                "cost",
                "net_return",
                "cumulative_nav",
                "solver_status",
            ]
        ].sort_values("month_end")
    if not test_weights.empty:
        test_weights = test_weights.sort_values(["month_end", "permno"]).reset_index(drop=True)
    if not test_actions.empty:
        test_actions = test_actions.sort_values("month_end").reset_index(drop=True)

    test_backtest.to_csv(outdir / "test_backtest.csv", index=False)
    test_weights.to_parquet(outdir / "test_weights.parquet", index=False, compression=PARQUET_COMPRESSION)
    test_actions.to_csv(outdir / "test_action_history.csv", index=False)
    write_summary(outdir / "test_summary.txt", test_backtest, test_metrics, args)

    plot_series(test_backtest, "cumulative_nav", "RL Overlay SAC Test Cumulative NAV", "Cumulative NAV", outdir / "test_cumret.png")
    plot_series(test_backtest, "turnover", "RL Overlay SAC Test Monthly Turnover", "L1 turnover", outdir / "test_turnover.png")
    plot_series(test_backtest, "lambda_t", "RL Overlay SAC Test Lambda", "lambda_t", outdir / "test_lambda_timeseries.png")
    plot_series(test_backtest, "tau_t", "RL Overlay SAC Test Tau", "tau_t", outdir / "test_tau_timeseries.png")

    train_env.close()
    test_env.close()
    LOGGER.info(
        "Done. Test annualized net return: %.4f. Test Sharpe: %.4f. Outputs: %s",
        test_metrics["annualized_return"],
        test_metrics["sharpe"],
        outdir,
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
