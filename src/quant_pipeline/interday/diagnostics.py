from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CandidateDiagnosticBundle:
    summary: pd.DataFrame
    quantile_curves: pd.DataFrame
    historical_folds: pd.DataFrame
    calendar_years: pd.DataFrame
    recent_windows: pd.DataFrame
    concentration: pd.DataFrame
    exposure: pd.DataFrame
    timing_path: pd.DataFrame
    execution_sensitivity: pd.DataFrame
    cost_headroom: pd.DataFrame
    context: pd.DataFrame


def calculate_trade_path_metrics(*, path_timestamp: pd.Series, path_total_value: pd.Series, entry_total_value: float, terminal_total_value: float) -> dict:
    if not np.isfinite(entry_total_value):
        raise ValueError("entry_total_value must be finite")
    if entry_total_value <= 0:
        raise ValueError("entry_total_value must be positive")
    values = pd.to_numeric(path_total_value, errors="coerce").to_numpy(dtype=float)
    timestamps = pd.to_datetime(path_timestamp, utc=True).to_numpy()
    valid = np.isfinite(values)
    if not valid.any():
        return {"mfe": np.nan, "mae": np.nan, "mfe_timestamp": pd.NaT, "mae_timestamp": pd.NaT, "terminal_return": np.nan, "giveback": np.nan}
    returns = values[valid] / entry_total_value - 1.0
    valid_timestamps = timestamps[valid]
    mfe_index = int(np.argmax(returns))
    mae_index = int(np.argmin(returns))
    terminal_return = terminal_total_value / entry_total_value - 1.0
    return {
        "mfe": float(returns[mfe_index]),
        "mae": float(returns[mae_index]),
        "mfe_timestamp": pd.Timestamp(valid_timestamps[mfe_index]),
        "mae_timestamp": pd.Timestamp(valid_timestamps[mae_index]),
        "terminal_return": float(terminal_return),
        "giveback": float(returns[mfe_index] - terminal_return),
    }


def diagnostics_complete(*, candidates: pd.DataFrame, bundle: CandidateDiagnosticBundle | None) -> bool:
    if candidates.empty:
        return bundle is None
    if bundle is None:
        return False
    required = (bundle.summary, bundle.historical_folds, bundle.recent_windows, bundle.concentration, bundle.timing_path, bundle.execution_sensitivity, bundle.cost_headroom)
    return all(isinstance(frame, pd.DataFrame) and not frame.empty for frame in required)


def net_spread_after_cost(
    gross_spread: float,
    one_way_turnover: float,
    cost_bps_per_side: float,
) -> float:
    round_trip_cost = (
        2.0
        * cost_bps_per_side
        / 10_000.0
        * one_way_turnover
    )
    return gross_spread - round_trip_cost


def exact_path_diagnostics(
    path: pd.DataFrame,
    *,
    entry_column: str = "entry",
    exit_column: str = "exit",
) -> dict:
    if path.empty:
        return {"status": "insufficient_data"}
    entry = path[entry_column].to_numpy(float)
    exits = path[exit_column].to_numpy(float)
    valid = np.isfinite(entry) & np.isfinite(exits) & (entry > 0)
    returns = np.where(valid, exits / entry - 1.0, np.nan)
    finite = np.isfinite(returns)
    return {
        "n": int(finite.sum()),
        "mean_return": float(np.nanmean(returns)) if finite.any() else np.nan,
        "mfe": float(np.nanmax(returns)) if finite.any() else np.nan,
        "mae": float(np.nanmin(returns)) if finite.any() else np.nan,
        "status": "built",
    }


def cost_sensitivity(returns, scenarios=(1.0, 3.0, 5.0)) -> pd.DataFrame:
    values = np.asarray(returns, float)
    gross = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
    return pd.DataFrame(
        [
            {
                "cost_bps_per_side": float(cost),
                "mean_net_return": net_spread_after_cost(gross, 1.0, float(cost)),
            }
            for cost in scenarios
        ]
    )


def empty_no_candidate_diagnostics() -> CandidateDiagnosticBundle:
    empty = pd.DataFrame()
    return CandidateDiagnosticBundle(
        summary=empty.copy(),
        quantile_curves=empty.copy(),
        historical_folds=empty.copy(),
        calendar_years=empty.copy(),
        recent_windows=empty.copy(),
        concentration=empty.copy(),
        exposure=empty.copy(),
        timing_path=empty.copy(),
        execution_sensitivity=empty.copy(),
        cost_headroom=empty.copy(),
        context=empty.copy(),
    )
