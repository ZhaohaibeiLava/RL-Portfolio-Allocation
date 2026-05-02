#!/usr/bin/env python3
"""Build Risk Layer v1: monthly covariance matrices for forecast universes.

For each forecast month t, the script uses the stocks with non-missing mu_hat in
the prediction file, collects realized CRSP monthly returns through month t only,
and estimates a shrinkage or sample covariance matrix for that month-t universe.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.covariance import LedoitWolf, OAS


LOGGER = logging.getLogger("build_risk_layer_v1")

DEFAULT_PRED_FILE = "fm_oos_predictions.parquet"
PREFERRED_RETURNS_FILE = "crsp_monthly_ever_sp500.parquet"
FALLBACK_RETURNS_FILE = "monthly_stock_panel_basic_full.parquet"
PARQUET_COMPRESSION = "snappy"
PSD_TOL = 1e-10
ILL_CONDITIONED_THRESHOLD = 1e12


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build monthly covariance matrices for the prediction universe."
    )
    parser.add_argument("--indir", required=True, help="Input folder.")
    parser.add_argument("--outdir", required=True, help="Output folder.")
    parser.add_argument(
        "--pred-file",
        default=DEFAULT_PRED_FILE,
        help=f"Prediction parquet filename. Default: {DEFAULT_PRED_FILE}.",
    )
    parser.add_argument(
        "--returns-file",
        default=None,
        help="Optional return-history parquet filename or path override.",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=60,
        help="Trailing return window length in months. Default: 60.",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=24,
        help="Minimum non-missing monthly returns required per asset. Default: 24.",
    )
    parser.add_argument(
        "--estimator",
        choices=("ledoit_wolf", "oas", "sample"),
        default="ledoit_wolf",
        help="Covariance estimator. Default: ledoit_wolf.",
    )
    parser.add_argument(
        "--save-corr",
        type=parse_bool,
        default=False,
        help="Whether to save correlation matrices in each NPZ. Default: false.",
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
    """Validate required columns."""
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
            "pyarrow is required for parquet IO. Install it in portfolio_allocation_rl."
        ) from exc
    LOGGER.info("Using pyarrow %s for parquet IO.", pa.__version__)


def read_parquet_file(path: Path, dataset_name: str) -> pd.DataFrame:
    """Read a parquet file with useful error context."""
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} file not found: {path}")
    LOGGER.info("Loading %s from %s.", dataset_name, path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {dataset_name} file {path}: {exc}") from exc


def candidate_input_dirs(indir: Path) -> list[Path]:
    """Return likely input directories for return-history files."""
    candidates = [
        indir,
        indir.parent,
        indir.parent / "raw",
        indir.parent / "panel",
        Path.cwd() / "data" / "raw",
        Path.cwd() / "data" / "panel",
    ]
    unique: list[Path] = []
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def resolve_return_file(indir: Path, returns_file: str | None) -> tuple[Path, str, bool]:
    """Resolve return-history file by override, preferred file, then fallback file."""
    if returns_file:
        override = Path(returns_file).expanduser()
        candidates = [override] if override.is_absolute() else [indir / override]
        candidates += [directory / override for directory in candidate_input_dirs(indir)]
        for path in candidates:
            if path.exists():
                return path.resolve(), path.name, False
        raise FileNotFoundError(f"--returns-file was provided but not found: {returns_file}")

    for directory in candidate_input_dirs(indir):
        preferred = directory / PREFERRED_RETURNS_FILE
        if preferred.exists():
            return preferred, PREFERRED_RETURNS_FILE, False

    for directory in candidate_input_dirs(indir):
        fallback = directory / FALLBACK_RETURNS_FILE
        if fallback.exists():
            LOGGER.warning(
                "Preferred return-history file %s was not found; using fallback %s. "
                "Return history may be incomplete for some assets because this file may "
                "already be filtered by membership-at-t.",
                PREFERRED_RETURNS_FILE,
                fallback,
            )
            return fallback, FALLBACK_RETURNS_FILE, True

    searched = ", ".join(str(path) for path in candidate_input_dirs(indir))
    raise FileNotFoundError(
        f"No return-history file found. Searched for {PREFERRED_RETURNS_FILE} and "
        f"{FALLBACK_RETURNS_FILE} in: {searched}"
    )


def normalize_month_end(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Ensure month_end exists and is month-end normalized."""
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


def clean_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Clean prediction universe input."""
    require_columns(predictions, ["permno", "month_end", "mu_hat"], "prediction file")
    result = normalize_month_end(predictions, "prediction file")
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce")
    result["mu_hat"] = pd.to_numeric(result["mu_hat"], errors="coerce")
    result = result.dropna(subset=["permno", "month_end", "mu_hat"]).copy()
    result["permno"] = result["permno"].astype("int64")
    result = result.drop_duplicates(["permno", "month_end"], keep="last")
    return result.sort_values(["month_end", "permno"]).reset_index(drop=True)


def clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Clean return-history input."""
    if "month_end" not in returns.columns and "date" not in returns.columns:
        raise ValueError("return-history file must contain month_end or date.")
    require_columns(returns, ["permno", "retadj"], "return-history file")
    result = normalize_month_end(returns, "return-history file")
    result["permno"] = pd.to_numeric(result["permno"], errors="coerce")
    result["retadj"] = pd.to_numeric(result["retadj"], errors="coerce")
    result = result.dropna(subset=["permno", "month_end"]).copy()
    result["permno"] = result["permno"].astype("int64")

    exact_duplicates = int(result.duplicated(["permno", "month_end", "retadj"]).sum())
    if exact_duplicates:
        LOGGER.warning("Dropping %s exact duplicate return rows.", f"{exact_duplicates:,}")
        result = result.drop_duplicates(["permno", "month_end", "retadj"], keep="last")

    duplicate_keys = int(result.duplicated(["permno", "month_end"]).sum())
    if duplicate_keys:
        raise ValueError(
            f"Return history has {duplicate_keys:,} duplicate (permno, month_end) keys "
            "after exact duplicate removal."
        )

    return result[["permno", "month_end", "retadj"]].sort_values(["permno", "month_end"])


def build_return_matrix(
    returns: pd.DataFrame,
    universe: np.ndarray,
    month_end: pd.Timestamp,
    lookback_window: int,
) -> pd.DataFrame:
    """Build a time-by-asset return matrix using months through month_end only."""
    window_start = month_end - MonthEnd(lookback_window - 1)
    window = returns.loc[
        returns["month_end"].between(window_start, month_end)
        & returns["permno"].isin(universe)
    ]
    if not window.empty and window["month_end"].max() > month_end:
        raise AssertionError(
            f"Lookahead violation: return window for {month_end.date()} contains "
            f"returns after month t."
        )
    matrix = window.pivot(index="month_end", columns="permno", values="retadj")
    full_months = pd.date_range(window_start, month_end, freq="ME")
    matrix = matrix.reindex(index=full_months, columns=universe)
    matrix.index.name = "month_end"
    return matrix


def filter_and_impute_returns(
    matrix: pd.DataFrame,
    min_history: int,
) -> tuple[pd.DataFrame, pd.Series, float, float]:
    """Drop sparse assets and impute remaining missing values with asset means."""
    missing_before = float(matrix.isna().to_numpy().mean()) if matrix.size else np.nan
    history_count = matrix.notna().sum(axis=0)
    kept_cols = history_count.loc[history_count >= min_history].index
    filtered = matrix.loc[:, kept_cols].copy()

    if filtered.empty:
        return filtered, history_count, missing_before, np.nan

    asset_means = filtered.mean(axis=0, skipna=True)
    imputed = filtered.fillna(asset_means)
    missing_after = float(imputed.isna().to_numpy().mean()) if imputed.size else np.nan
    return imputed, history_count, missing_before, missing_after


def estimate_covariance(
    returns_matrix: pd.DataFrame,
    estimator: str,
) -> tuple[np.ndarray, float]:
    """Estimate covariance matrix and return optional shrinkage intensity."""
    x = returns_matrix.to_numpy(dtype="float64")
    if estimator == "ledoit_wolf":
        model = LedoitWolf(assume_centered=False).fit(x)
        cov = model.covariance_
        shrinkage = float(model.shrinkage_)
    elif estimator == "oas":
        model = OAS(assume_centered=False).fit(x)
        cov = model.covariance_
        shrinkage = float(model.shrinkage_)
    elif estimator == "sample":
        cov = np.cov(x, rowvar=False, ddof=1)
        shrinkage = np.nan
    else:
        raise ValueError(f"Unsupported estimator: {estimator}")

    cov = np.asarray(cov, dtype="float64")
    cov = 0.5 * (cov + cov.T)
    return cov, shrinkage


def covariance_diagnostics(cov: np.ndarray) -> tuple[float, bool, float]:
    """Compute minimum eigenvalue, PSD flag, and condition number."""
    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(np.min(eigvals))
    is_psd = bool(min_eig >= -PSD_TOL)
    try:
        condition_number = float(np.linalg.cond(cov))
    except np.linalg.LinAlgError:
        condition_number = np.nan
    return min_eig, is_psd, condition_number


def covariance_to_correlation(cov: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """Convert covariance to correlation, handling zero-vol assets defensively."""
    denom = np.outer(vols, vols)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    return 0.5 * (corr + corr.T)


def save_covariance_npz(
    path: Path,
    cov: np.ndarray,
    permnos: np.ndarray,
    month_end: pd.Timestamp,
    vols: np.ndarray,
    estimator: str,
    shrinkage: float,
    corr: np.ndarray | None,
) -> None:
    """Save one compressed covariance artifact."""
    if cov.shape != (len(permnos), len(permnos)):
        raise ValueError(
            f"Covariance shape {cov.shape} does not match {len(permnos)} saved permnos."
        )
    if len(vols) != len(permnos):
        raise ValueError(
            f"Volatility vector length {len(vols)} does not match {len(permnos)} saved permnos."
        )
    payload = {
        "cov": cov,
        "permnos": permnos.astype("int64"),
        "month_end": np.array([np.datetime64(month_end)], dtype="datetime64[ns]"),
        "vols": vols,
        "estimator": np.array([estimator]),
        "shrinkage": np.array([shrinkage], dtype="float64"),
    }
    if corr is not None:
        if corr.shape != cov.shape:
            raise ValueError(f"Correlation shape {corr.shape} does not match covariance shape {cov.shape}.")
        payload["corr"] = corr
    np.savez_compressed(path, **payload)


def build_risk_layer(
    predictions: pd.DataFrame,
    returns: pd.DataFrame,
    outdir: Path,
    estimator: str,
    lookback_window: int,
    min_history: int,
    save_corr: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build monthly covariance matrices and collect metadata outputs."""
    if lookback_window < 2:
        raise ValueError("--lookback-window must be at least 2.")
    if min_history < 2:
        raise ValueError("--min-history must be at least 2.")
    if min_history > lookback_window:
        raise ValueError("--min-history cannot exceed --lookback-window.")

    cov_dir = outdir / "risk_cov_npz"
    cov_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    coverage_rows: list[pd.DataFrame] = []
    diag_rows: list[pd.DataFrame] = []

    for month_end, pred_month in predictions.groupby("month_end", sort=True):
        month_end = pd.Timestamp(month_end)
        universe = pred_month["permno"].drop_duplicates().sort_values().to_numpy(dtype="int64")
        n_assets_input = len(universe)
        return_matrix = build_return_matrix(returns, universe, month_end, lookback_window)
        imputed, history_count, missing_before, missing_after = filter_and_impute_returns(
            return_matrix, min_history
        )
        kept_permnos = imputed.columns.to_numpy(dtype="int64")
        n_assets_kept = len(kept_permnos)

        coverage = pd.DataFrame(
            {
                "month_end": month_end,
                "permno": universe,
                "in_prediction_universe": True,
                "history_count": [int(history_count.get(permno, 0)) for permno in universe],
            }
        )
        coverage["kept_in_risk_matrix"] = coverage["permno"].isin(kept_permnos)
        coverage_rows.append(coverage)

        if n_assets_kept < 2:
            LOGGER.warning(
                "Skipping %s: only %s assets remain after min-history filtering.",
                month_end.date(),
                n_assets_kept,
            )
            continue

        dropped_frac = 1.0 - n_assets_kept / n_assets_input if n_assets_input else np.nan
        if np.isfinite(dropped_frac) and dropped_frac > 0.25:
            LOGGER.warning(
                "%s dropped %.1f%% of assets after min-history filtering.",
                month_end.date(),
                100.0 * dropped_frac,
            )

        LOGGER.info(
            "%s risk window: assets input=%s, kept=%s, missing before=%.4f, missing after=%.4f.",
            month_end.date(),
            n_assets_input,
            n_assets_kept,
            missing_before,
            missing_after,
        )

        cov, shrinkage = estimate_covariance(imputed, estimator)
        if cov.shape != (n_assets_kept, n_assets_kept):
            raise ValueError(
                f"{month_end.date()} covariance shape {cov.shape} does not match "
                f"kept asset count {n_assets_kept}."
            )
        vols = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
        min_eig, is_psd, condition_number = covariance_diagnostics(cov)

        if np.isfinite(condition_number) and condition_number > ILL_CONDITIONED_THRESHOLD:
            LOGGER.warning(
                "%s covariance matrix is ill-conditioned: condition_number=%.3e.",
                month_end.date(),
                condition_number,
            )
        if not is_psd:
            LOGGER.warning("%s covariance matrix is not PSD within tolerance.", month_end.date())

        corr = covariance_to_correlation(cov, vols) if save_corr else None
        save_covariance_npz(
            cov_dir / f"cov_{month_end:%Y%m}.npz",
            cov,
            kept_permnos,
            month_end,
            vols,
            estimator,
            shrinkage,
            corr,
        )

        metadata_rows.append(
            {
                "month_end": month_end,
                "estimator": estimator,
                "lookback_window": lookback_window,
                "n_assets_input": n_assets_input,
                "n_assets_kept": n_assets_kept,
                "n_time_obs": int(imputed.shape[0]),
                "missing_frac_before": missing_before,
                "missing_frac_after": missing_after,
                "shrinkage": shrinkage,
                "min_eig": min_eig,
                "is_psd": is_psd,
                "condition_number": condition_number,
            }
        )
        diag_rows.append(
            pd.DataFrame({"month_end": month_end, "permno": kept_permnos, "vol_hat": vols})
        )

    if not metadata_rows:
        raise RuntimeError("No covariance matrices were produced.")

    metadata = pd.DataFrame(metadata_rows).sort_values("month_end").reset_index(drop=True)
    coverage_all = pd.concat(coverage_rows, ignore_index=True)
    diag_all = pd.concat(diag_rows, ignore_index=True)
    return metadata, coverage_all, diag_all


def write_summary(
    path: Path,
    metadata: pd.DataFrame,
    returns_filename: str,
    used_fallback: bool,
) -> None:
    """Write text summary."""
    shrinkage = metadata["shrinkage"].dropna()
    lines = [
        "Risk Layer v1 summary",
        "=====================",
        f"Return-history file: {returns_filename}",
        f"Used fallback return history: {used_fallback}",
        f"Date range: {metadata['month_end'].min().date()} to {metadata['month_end'].max().date()}",
        f"Number of processed months: {metadata['month_end'].nunique():,}",
        f"Average assets before filtering: {metadata['n_assets_input'].mean():.2f}",
        f"Average assets after filtering: {metadata['n_assets_kept'].mean():.2f}",
        f"Average missing fraction before imputation: {metadata['missing_frac_before'].mean():.6f}",
        f"Average shrinkage: {shrinkage.mean():.6f}" if len(shrinkage) else "Average shrinkage: NA",
        f"Fraction of months with PSD matrices: {metadata['is_psd'].mean():.6f}",
    ]
    if used_fallback:
        lines.extend(
            [
                "",
                "Fallback note:",
                "The fallback return-history file may already be filtered to S&P 500 member-months, "
                "so trailing return histories can be incomplete for some assets.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_series(df: pd.DataFrame, y_col: str, title: str, ylabel: str, path: Path) -> None:
    """Save a simple time-series plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["month_end"], df[y_col], linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_outputs(
    outdir: Path,
    metadata: pd.DataFrame,
    coverage: pd.DataFrame,
    diag: pd.DataFrame,
    returns_filename: str,
    used_fallback: bool,
) -> None:
    """Save metadata, coverage, diagnostics, summary, and plots."""
    outdir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(outdir / "risk_monthly_metadata.csv", index=False)
    coverage.to_parquet(
        outdir / "risk_asset_coverage.parquet",
        index=False,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
    )
    diag.to_parquet(
        outdir / "risk_diag.parquet",
        index=False,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
    )
    write_summary(outdir / "risk_summary.txt", metadata, returns_filename, used_fallback)

    plot_series(
        metadata,
        "n_assets_kept",
        "Risk Matrix Assets Kept",
        "Assets",
        outdir / "risk_n_assets_kept.png",
    )
    plot_series(
        metadata,
        "missing_frac_before",
        "Missing Return Fraction Before Imputation",
        "Missing fraction",
        outdir / "risk_missingness_before.png",
    )
    avg_vol = diag.groupby("month_end", as_index=False)["vol_hat"].mean()
    plot_series(avg_vol, "vol_hat", "Average Estimated Monthly Volatility", "Volatility", outdir / "risk_avg_vol.png")
    if metadata["shrinkage"].notna().any():
        plot_series(
            metadata,
            "shrinkage",
            "Shrinkage Intensity",
            "Shrinkage",
            outdir / "risk_shrinkage_timeseries.png",
        )


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the Risk Layer v1 pipeline."""
    ensure_pyarrow_available()
    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not indir.exists():
        raise FileNotFoundError(f"Input directory not found: {indir}")

    pred_path = indir / args.pred_file
    returns_path, returns_filename, used_fallback = resolve_return_file(indir, args.returns_file)

    predictions = clean_predictions(read_parquet_file(pred_path, "prediction file"))
    returns = clean_returns(read_parquet_file(returns_path, "return-history file"))
    metadata, coverage, diag = build_risk_layer(
        predictions,
        returns,
        outdir,
        args.estimator,
        args.lookback_window,
        args.min_history,
        args.save_corr,
    )
    save_outputs(outdir, metadata, coverage, diag, returns_filename, used_fallback)
    LOGGER.info(
        "Done. Processed %s months; output directory: %s.",
        metadata["month_end"].nunique(),
        outdir,
    )


def main() -> None:
    """Entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
