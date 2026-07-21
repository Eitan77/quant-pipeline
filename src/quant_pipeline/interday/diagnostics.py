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
