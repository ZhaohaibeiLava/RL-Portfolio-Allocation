#!/usr/bin/env python3
"""Run a CRSP-only rolling Fama-MacBeth expected-return baseline.

This script estimates month-by-month cross-sectional return regressions, then
forms true out-of-sample forecasts by averaging only past monthly coefficient
estimates over a rolling training window.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd


LOGGER = logging.getLogger("run_fm_baseline_v1")

DEFAULT_PANEL_FILE = "monthly_stock_panel_basic_model.parquet"
DEFAULT_TARGET_COL = "target_excess_1m"
DEFAULT_FEATURE_COLS = ["log_me", "rev_1m", "mom_12_2", "vol_12m"]
BASE_REQUIRED_COLUMNS = [
    "permno",
    "month_end",
    "target_ret_1m",
    "target_excess_1m",
]
PARQUET_COMPRESSION = "snappy"
DECILE_COLUMNS = [f"decile_{i}" for i in range(1, 11)]


def parse_bool(value: str | bool) -> bool:
    """Parse a robust command-line boolean."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value like true/false, got {value!r}."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a rolling Fama-MacBeth-style expected-return baseline on the "
            "model-ready monthly stock panel."
        )
    )
    parser.add_argument("--indir", required=True, help="Input folder.")
    parser.add_argument("--outdir", required=True, help="Output folder.")
    parser.add_argument(
        "--panel-file",
        default=DEFAULT_PANEL_FILE,
        help=f"Input model panel parquet filename. Default: {DEFAULT_PANEL_FILE}.",
    )
    parser.add_argument(
        "--target-col",
        default=DEFAULT_TARGET_COL,
        help=f"Target column to forecast. Default: {DEFAULT_TARGET_COL}.",
    )
    parser.add_argument(
        "--feature-cols",
        default=",".join(DEFAULT_FEATURE_COLS),
        help=(
            "Comma-separated feature columns. Default: "
            f"{','.join(DEFAULT_FEATURE_COLS)}."
        ),
    )
    parser.add_argument(
        "--train-window",
        type=int,
        default=60,
        help="Rolling coefficient window length in months. Default: 60.",
    )
    parser.add_argument(
        "--min-cross-section",
        type=int,
        default=50,
        help="Minimum usable stocks required per cross-sectional OLS. Default: 50.",
    )
    parser.add_argument(
        "--standardize-features",
        type=parse_bool,
        default=False,
        help="Whether to z-score features cross-sectionally within each month. Default: false.",
    )
    parser.add_argument(
        "--coef-aggregation",
        choices=("mean", "median"),
        default="mean",
        help="How to aggregate past monthly coefficients. Default: mean.",
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
    """Fail clearly when required columns are missing."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def ensure_pyarrow_available() -> None:
    """Fail early if pyarrow is unavailable for parquet output."""
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "The pyarrow package is required for parquet output. Install it with "
            "`pip install pyarrow` before running this script."
        ) from exc
    LOGGER.info("Using pyarrow %s for parquet output.", pa.__version__)


def parse_feature_cols(value: str) -> list[str]:
    """Parse comma-separated feature names."""
    features = [col.strip() for col in value.split(",") if col.strip()]
    if not features:
        raise ValueError("--feature-cols must contain at least one feature column.")
    duplicates = sorted({col for col in features if features.count(col) > 1})
    if duplicates:
        raise ValueError(f"--feature-cols contains duplicate names: {', '.join(duplicates)}")
    return features


def read_panel(path: Path) -> pd.DataFrame:
    """Load the model-ready panel."""
    if not path.exists():
        raise FileNotFoundError(f"Input panel file not found: {path}")
    LOGGER.info("Loading model panel from %s.", path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read panel parquet file {path}: {exc}") from exc


def clean_panel(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    standardize_features: bool,
) -> pd.DataFrame:
    """Validate, type-clean, sort, and optionally standardize the panel."""
    required = BASE_REQUIRED_COLUMNS + [target_col] + feature_cols
    require_columns(panel, required, "model panel")

    keep_cols = ["permno", "month_end", target_col] + feature_cols
    optional_cols = [col for col in ["me", "exchcd", "shrcd", "in_sp500_t"] if col in panel.columns]
    result = panel[keep_cols + optional_cols].copy()

    result["permno"] = pd.to_numeric(result["permno"], errors="coerce")
    result["month_end"] = pd.to_datetime(result["month_end"], errors="coerce") + MonthEnd(0)
    result[target_col] = pd.to_numeric(result[target_col], errors="coerce")
    for col in feature_cols + optional_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    before = len(result)
    result = result.dropna(subset=["permno", "month_end", target_col] + feature_cols).copy()
    result["permno"] = result["permno"].astype("int64")
    LOGGER.info("Kept %s of %s rows after dropping missing target/features.", len(result), before)

    duplicate_keys = int(result.duplicated(["permno", "month_end"]).sum())
    if duplicate_keys:
        raise ValueError(
            f"Model panel has {duplicate_keys:,} duplicate (permno, month_end) rows "
            "after cleaning; expected one row per stock-month."
        )

    result = result.sort_values(["month_end", "permno"]).reset_index(drop=True)
    if standardize_features:
        result = standardize_features_by_month(result, feature_cols)
    return result


def standardize_features_by_month(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Z-score features within each month using only that month's cross section."""
    LOGGER.info("Standardizing features cross-sectionally within each month.")
    result = panel.copy()
    grouped = result.groupby("month_end", sort=False)
    for col in feature_cols:
        means = grouped[col].transform("mean")
        stds = grouped[col].transform(lambda s: s.std(ddof=0))
        valid = stds.gt(0).fillna(False)
        result[col] = np.where(valid, (result[col] - means) / stds, np.nan)

    before = len(result)
    result = result.dropna(subset=feature_cols).copy()
    dropped = before - len(result)
    if dropped:
        LOGGER.warning(
            "Dropped %s rows after standardization because at least one feature had zero or missing monthly std.",
            f"{dropped:,}",
        )
    return result.sort_values(["month_end", "permno"]).reset_index(drop=True)


def run_cross_sectional_ols(
    month_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, float]:
    """Run OLS with intercept and return coefficients plus R-squared."""
    y = month_df[target_col].to_numpy(dtype="float64")
    x_features = month_df[feature_cols].to_numpy(dtype="float64")
    x = np.column_stack([np.ones(len(month_df)), x_features])

    rank = np.linalg.matrix_rank(x)
    if rank < x.shape[1]:
        raise np.linalg.LinAlgError(
            f"Design matrix is rank deficient: rank={rank}, columns={x.shape[1]}."
        )

    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    resid = y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return coef, r2


def estimate_monthly_betas(
    panel: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    min_cross_section: int,
) -> pd.DataFrame:
    """Estimate one cross-sectional OLS per month."""
    rows: list[dict[str, object]] = []
    for month_end, month_df in panel.groupby("month_end", sort=True):
        n_obs = len(month_df)
        if n_obs < min_cross_section:
            LOGGER.warning(
                "Skipping %s: only %s usable observations, below min-cross-section=%s.",
                pd.Timestamp(month_end).date(),
                n_obs,
                min_cross_section,
            )
            continue
        if n_obs <= len(feature_cols) + 1:
            LOGGER.warning(
                "Skipping %s: not enough observations for %s regressors plus intercept.",
                pd.Timestamp(month_end).date(),
                len(feature_cols),
            )
            continue

        try:
            coef, r2 = run_cross_sectional_ols(month_df, target_col, feature_cols)
        except np.linalg.LinAlgError as exc:
            LOGGER.warning("Skipping %s: %s", pd.Timestamp(month_end).date(), exc)
            continue

        row: dict[str, object] = {
            "month_end": month_end,
            "alpha": coef[0],
            "n_obs": n_obs,
            "r2": r2,
        }
        for feature, beta in zip(feature_cols, coef[1:], strict=True):
            row[f"beta_{feature}"] = beta
        rows.append(row)

    if not rows:
        raise RuntimeError("No monthly cross-sectional regressions were estimated.")

    betas = pd.DataFrame(rows).sort_values("month_end").reset_index(drop=True)
    LOGGER.info("Estimated monthly betas for %s months.", len(betas))
    return betas


def aggregate_coefficients(
    coef_window: pd.DataFrame,
    feature_cols: list[str],
    method: str,
) -> pd.Series:
    """Aggregate past monthly coefficients."""
    coef_cols = ["alpha"] + [f"beta_{feature}" for feature in feature_cols]
    if method == "mean":
        return coef_window[coef_cols].mean(axis=0)
    if method == "median":
        return coef_window[coef_cols].median(axis=0)
    raise ValueError(f"Unsupported coefficient aggregation method: {method}")


def make_oos_predictions(
    panel: pd.DataFrame,
    betas: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    train_window: int,
    coef_aggregation: str,
) -> pd.DataFrame:
    """Compute true rolling out-of-sample predictions."""
    if train_window < 1:
        raise ValueError("--train-window must be at least 1.")

    months = list(pd.Series(panel["month_end"].drop_duplicates()).sort_values())
    beta_lookup = betas.set_index("month_end").sort_index()
    prediction_frames: list[pd.DataFrame] = []

    if len(months) <= train_window:
        raise RuntimeError(
            f"Panel has only {len(months)} months after cleaning; train-window={train_window} "
            "leaves no out-of-sample forecast months."
        )

    for idx in range(train_window, len(months)):
        forecast_month = months[idx]
        past_months = months[idx - train_window : idx]
        coef_window = beta_lookup.loc[beta_lookup.index.intersection(past_months)]
        if coef_window.empty:
            LOGGER.warning(
                "Skipping forecast month %s: no valid coefficient estimates in prior %s months.",
                pd.Timestamp(forecast_month).date(),
                train_window,
            )
            continue

        if len(coef_window) < train_window:
            LOGGER.warning(
                "Forecast month %s uses %s valid coefficient months out of the %s-month window.",
                pd.Timestamp(forecast_month).date(),
                len(coef_window),
                train_window,
            )

        coef_bar = aggregate_coefficients(coef_window, feature_cols, coef_aggregation)
        month_df = panel.loc[panel["month_end"].eq(forecast_month)].copy()
        if month_df.empty:
            continue

        mu_hat = np.full(len(month_df), float(coef_bar["alpha"]))
        for feature in feature_cols:
            mu_hat += month_df[feature].to_numpy(dtype="float64") * float(coef_bar[f"beta_{feature}"])

        month_df["mu_hat"] = mu_hat
        month_df["realized_target"] = month_df[target_col]
        month_df["n_coef_months"] = len(coef_window)
        prediction_frames.append(
            month_df[["permno", "month_end", "mu_hat", "realized_target", "n_coef_months"] + feature_cols]
        )

    if not prediction_frames:
        raise RuntimeError("No out-of-sample predictions were produced.")

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        .sort_values(["month_end", "permno"])
        .reset_index(drop=True)
    )
    LOGGER.info(
        "Produced %s out-of-sample predictions across %s months.",
        f"{len(predictions):,}",
        predictions["month_end"].nunique(),
    )
    return predictions


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    """Compute a correlation, returning NaN when it is undefined."""
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return np.nan
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(valid["x"].corr(valid["y"], method="pearson"))
    if method == "spearman":
        x_rank = valid["x"].rank(method="average")
        y_rank = valid["y"].rank(method="average")
        return float(x_rank.corr(y_rank, method="pearson"))
    raise ValueError(f"Unsupported correlation method: {method}")


def compute_decile_returns(month_df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Compute equal-weight realized-target returns by predicted-return decile."""
    output = {col: np.nan for col in DECILE_COLUMNS}
    if len(month_df) < 10:
        LOGGER.warning(
            "Skipping decile returns for %s: fewer than 10 stocks.",
            pd.Timestamp(month_df["month_end"].iloc[0]).date(),
        )
        return output, np.nan

    unique_predictions = int(month_df["mu_hat"].nunique(dropna=True))
    if unique_predictions < 10:
        LOGGER.warning(
            "Skipping decile returns for %s: only %s unique predicted values.",
            pd.Timestamp(month_df["month_end"].iloc[0]).date(),
            unique_predictions,
        )
        return output, np.nan

    ranks = month_df["mu_hat"].rank(method="first")
    deciles = pd.qcut(ranks, q=10, labels=False) + 1
    decile_means = month_df.assign(decile=deciles).groupby("decile")["realized_target"].mean()

    for decile, value in decile_means.items():
        output[f"decile_{int(decile)}"] = float(value)

    spread = output["decile_10"] - output["decile_1"]
    return output, spread


def evaluate_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Compute monthly ICs, decile returns, and full-sample summary metrics."""
    ic_rows: list[dict[str, object]] = []
    decile_rows: list[dict[str, object]] = []

    for month_end, month_df in predictions.groupby("month_end", sort=True):
        ic = safe_corr(month_df["mu_hat"], month_df["realized_target"], method="pearson")
        rank_ic = safe_corr(month_df["mu_hat"], month_df["realized_target"], method="spearman")
        decile_returns, spread = compute_decile_returns(month_df)

        ic_rows.append(
            {
                "month_end": month_end,
                "ic": ic,
                "rank_ic": rank_ic,
                "n_obs": len(month_df),
                "spread_return": spread,
            }
        )
        decile_rows.append({"month_end": month_end, **decile_returns})

    monthly_ic = pd.DataFrame(ic_rows).sort_values("month_end").reset_index(drop=True)
    decile_returns = pd.DataFrame(decile_rows).sort_values("month_end").reset_index(drop=True)

    ic_series = monthly_ic["ic"].dropna()
    rank_ic_series = monthly_ic["rank_ic"].dropna()
    spread_series = monthly_ic["spread_return"].dropna()

    ic_t_stat = ordinary_t_stat(ic_series)
    monthly_spread_mean = float(spread_series.mean()) if len(spread_series) else np.nan
    monthly_spread_std = float(spread_series.std(ddof=1)) if len(spread_series) > 1 else np.nan

    summary = {
        "forecast_months": float(monthly_ic["month_end"].nunique()),
        "average_cross_sectional_sample_size": float(monthly_ic["n_obs"].mean()),
        "mean_ic": float(ic_series.mean()) if len(ic_series) else np.nan,
        "mean_rank_ic": float(rank_ic_series.mean()) if len(rank_ic_series) else np.nan,
        "ic_t_stat": ic_t_stat,
        "mean_spread_return": monthly_spread_mean,
        "annualized_spread_return": 12.0 * monthly_spread_mean if np.isfinite(monthly_spread_mean) else np.nan,
        "annualized_spread_vol": np.sqrt(12.0) * monthly_spread_std if np.isfinite(monthly_spread_std) else np.nan,
        "spread_sharpe": (
            np.sqrt(12.0) * monthly_spread_mean / monthly_spread_std
            if np.isfinite(monthly_spread_mean) and np.isfinite(monthly_spread_std) and monthly_spread_std > 0
            else np.nan
        ),
    }
    return monthly_ic, decile_returns, summary


def ordinary_t_stat(series: pd.Series) -> float:
    """Compute an ordinary time-series t-statistic for the mean."""
    values = series.dropna().to_numpy(dtype="float64")
    if len(values) < 2:
        return np.nan
    std = values.std(ddof=1)
    if std <= 0 or not np.isfinite(std):
        return np.nan
    return float(values.mean() / (std / np.sqrt(len(values))))


def format_date(value: object) -> str:
    """Format a timestamp-like object for reports."""
    if value is None or pd.isna(value):
        return "NA"
    return pd.Timestamp(value).date().isoformat()


def write_eval_summary(
    path: Path,
    predictions: pd.DataFrame,
    summary: dict[str, float],
    target_col: str,
    feature_cols: list[str],
    train_window: int,
    min_cross_section: int,
    standardize_features: bool,
    coef_aggregation: str,
) -> None:
    """Write the text evaluation summary."""
    date_min = predictions["month_end"].min() if not predictions.empty else None
    date_max = predictions["month_end"].max() if not predictions.empty else None
    lines = [
        "Fama-MacBeth baseline v1 evaluation summary",
        "===========================================",
        f"Target column: {target_col}",
        f"Feature columns: {', '.join(feature_cols)}",
        f"Train window: {train_window}",
        f"Minimum cross section: {min_cross_section}",
        f"Standardize features: {standardize_features}",
        f"Coefficient aggregation: {coef_aggregation}",
        f"Date range: {format_date(date_min)} to {format_date(date_max)}",
        f"Number of forecast months: {summary['forecast_months']:.0f}",
        f"Average cross-sectional sample size: {summary['average_cross_sectional_sample_size']:.2f}",
        f"Mean monthly IC: {summary['mean_ic']:.6f}",
        f"Mean monthly rank IC: {summary['mean_rank_ic']:.6f}",
        f"IC t-stat: {summary['ic_t_stat']:.6f}",
        f"Mean decile spread return: {summary['mean_spread_return']:.6f}",
        f"Annualized spread return: {summary['annualized_spread_return']:.6f}",
        f"Annualized spread vol: {summary['annualized_spread_vol']:.6f}",
        f"Spread Sharpe ratio: {summary['spread_sharpe']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_plots(monthly_ic: pd.DataFrame, outdir: Path) -> None:
    """Save IC and spread diagnostic plots using matplotlib."""
    plot_series(
        monthly_ic,
        y_col="ic",
        title="Monthly Pearson IC",
        ylabel="IC",
        path=outdir / "fm_ic_timeseries.png",
    )
    plot_series(
        monthly_ic,
        y_col="rank_ic",
        title="Monthly Spearman Rank IC",
        ylabel="Rank IC",
        path=outdir / "fm_rank_ic_timeseries.png",
    )

    spread = monthly_ic[["month_end", "spread_return"]].copy()
    spread["cumulative_spread"] = (1.0 + spread["spread_return"].fillna(0.0)).cumprod() - 1.0
    plot_series(
        spread,
        y_col="cumulative_spread",
        title="Cumulative Top-Minus-Bottom Decile Spread",
        ylabel="Cumulative return",
        path=outdir / "fm_cumulative_spread.png",
    )


def plot_series(df: pd.DataFrame, y_col: str, title: str, ylabel: str, path: Path) -> None:
    """Save a single time-series line plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["month_end"], df[y_col], linewidth=1.2)
    ax.axhline(0.0, color="black", linewidth=0.8)
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
    betas: pd.DataFrame,
    predictions: pd.DataFrame,
    monthly_ic: pd.DataFrame,
    decile_returns: pd.DataFrame,
    summary: dict[str, float],
    target_col: str,
    feature_cols: list[str],
    train_window: int,
    min_cross_section: int,
    standardize_features: bool,
    coef_aggregation: str,
) -> None:
    """Save parquet, CSV, TXT, and plot outputs."""
    outdir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving monthly betas.")
    betas.to_parquet(
        outdir / "fm_monthly_betas.parquet",
        index=False,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
    )

    LOGGER.info("Saving out-of-sample predictions.")
    predictions.to_parquet(
        outdir / "fm_oos_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression=PARQUET_COMPRESSION,
    )

    monthly_ic.to_csv(outdir / "fm_monthly_ic.csv", index=False)
    decile_returns.to_csv(outdir / "fm_decile_returns.csv", index=False)
    write_eval_summary(
        outdir / "fm_eval_summary.txt",
        predictions,
        summary,
        target_col,
        feature_cols,
        train_window,
        min_cross_section,
        standardize_features,
        coef_aggregation,
    )
    save_plots(monthly_ic, outdir)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full Fama-MacBeth baseline pipeline."""
    ensure_pyarrow_available()
    feature_cols = parse_feature_cols(args.feature_cols)
    if args.train_window < 1:
        raise ValueError("--train-window must be at least 1.")
    if args.min_cross_section < len(feature_cols) + 2:
        raise ValueError(
            "--min-cross-section must be larger than the number of regressors plus intercept."
        )

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not indir.exists():
        raise FileNotFoundError(f"Input directory not found: {indir}")

    panel = read_panel(indir / args.panel_file)
    clean = clean_panel(panel, args.target_col, feature_cols, args.standardize_features)
    betas = estimate_monthly_betas(clean, args.target_col, feature_cols, args.min_cross_section)
    predictions = make_oos_predictions(
        clean,
        betas,
        args.target_col,
        feature_cols,
        args.train_window,
        args.coef_aggregation,
    )
    monthly_ic, decile_returns, summary = evaluate_predictions(predictions)
    save_outputs(
        outdir,
        betas,
        predictions,
        monthly_ic,
        decile_returns,
        summary,
        args.target_col,
        feature_cols,
        args.train_window,
        args.min_cross_section,
        args.standardize_features,
        args.coef_aggregation,
    )

    LOGGER.info(
        "Done. Forecast months: %s; predictions: %s; output directory: %s.",
        int(summary["forecast_months"]),
        f"{len(predictions):,}",
        outdir,
    )


def main() -> None:
    """Entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
