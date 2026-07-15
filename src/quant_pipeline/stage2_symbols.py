from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .holdout import assert_pre_holdout_frame


TEMPLATES = (
    "session_range_position__0935__d0__120m__long_short__n1",
    "first_bar_close_location__0935__d5__60m__long_short__n1",
    "first_bar_close_location__0935__d5__120m__long_short__n1",
    "session_range_position__0955__d0__120m__long_short__n1",
    "session_range_position__0935__d0__eod__long_short__q10",
)


@dataclass(frozen=True)
class SymbolConfig:
    phase2_run_dir: str
    sealed_holdout_start: str = "2026-05-01"
    symbols_per_template: int = 20
    shrinkage_prior_trades: int = 50
    cost_bps_per_side: float = 2.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SymbolConfig":
        return cls(**(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}))


def run_symbol_search(config: SymbolConfig) -> dict:
    root = Path(config.phase2_run_dir); trades = pd.read_parquet(root / "baseline_trades.parquet")
    daily = pd.read_parquet(root / "baseline_daily_portfolios.parquet", columns=["strategy_id", "session_date"])
    assert_pre_holdout_frame(trades, config.sealed_holdout_start, "Phase 2 symbol input trades")
    assert_pre_holdout_frame(daily, config.sealed_holdout_start, "Phase 2 symbol input sessions")
    available = set(trades.strategy_id.unique()); missing = set(TEMPLATES) - available
    if missing:
        raise ValueError(f"Missing predeclared symbol templates: {sorted(missing)}")
    results = []
    for strategy_id in TEMPLATES:
        template = trades.loc[trades.strategy_id.eq(strategy_id)].copy()
        selected = (template.groupby("symbol").size().rename("n").reset_index()
                    .sort_values(["n", "symbol"], ascending=[False, True]).head(config.symbols_per_template).symbol)
        opportunity_dates = pd.Index(sorted(pd.to_datetime(daily.loc[daily.strategy_id.eq(strategy_id), "session_date"]).dt.normalize().unique()))
        universe_mean = float(template.gross_trade_return.mean())
        for symbol in selected:
            z = template.loc[template.symbol.eq(symbol)].sort_values("session_date").copy()
            results.append(symbol_metrics(z, opportunity_dates, strategy_id, universe_mean, config))
    output = pd.DataFrame(results)
    if len(output) != len(TEMPLATES) * config.symbols_per_template:
        raise ValueError(f"Symbol grid produced {len(output)} tests, expected {len(TEMPLATES) * config.symbols_per_template}")
    output = output.sort_values(["promotion_status", "shrunk_net_average_trade"], ascending=[True, False])
    output.to_csv(root / "symbol_specialization_leaderboard.csv", index=False)
    manifest = {
        "phase": "phase2_symbol_specialization", "executed_at": datetime.now(timezone.utc).isoformat(),
        "sealed_holdout_start": config.sealed_holdout_start, "holdout_access": False,
        "trial_count": len(output), "templates": list(TEMPLATES), "symbols_per_template": config.symbols_per_template,
    }
    (root / "symbol_specialization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def symbol_metrics(z: pd.DataFrame, opportunity_dates: pd.Index, strategy_id: str, universe_mean: float,
                   config: SymbolConfig) -> dict:
    cost = 2 * config.cost_bps_per_side / 10_000; z = z.copy()
    z["session_date"] = pd.to_datetime(z.session_date).dt.normalize(); z["net_return"] = z.gross_trade_return - cost
    daily = pd.Series(0.0, index=opportunity_dates); daily.loc[z.session_date] = z.net_return.to_numpy()
    n = len(z); weight = n / (n + config.shrinkage_prior_trades)
    shrunk = weight * float(z.net_return.mean()) + (1 - weight) * (universe_mean - cost)
    years = sorted(z.session_date.dt.year.unique()); yearly = {int(y): float(z.loc[z.session_date.dt.year.eq(y), "net_return"].mean()) for y in years}
    leave_out = []
    for year in years:
        mask = pd.to_datetime(daily.index).year != year; leave_out.append(_cagr(daily[mask]))
    without_best = z.drop(z.nlargest(min(5, n), "net_return").index).net_return if n > 5 else pd.Series(dtype=float)
    recent = z.loc[z.session_date.ge(pd.Timestamp("2025-05-01")), "net_return"]
    positive = z.loc[z.net_return.gt(0), "net_return"]; positive_total = positive.sum()
    result = {
        "symbol": str(z.symbol.iloc[0]), "strategy_id": strategy_id, "trade_count": n,
        "independent_sessions": int(z.session_date.nunique()), "years": len(years),
        "long_trades": int(z.side.eq(1).sum()), "short_trades": int(z.side.eq(-1).sum()),
        "gross_average_trade": float(z.gross_trade_return.mean()), "net_average_trade": float(z.net_return.mean()),
        "median_net_trade": float(z.net_return.median()), "win_rate": float(z.net_return.gt(0).mean()),
        "net_cagr": _cagr(daily), "maximum_drawdown": _max_drawdown(daily),
        "recent_average_net_trade": float(recent.mean()) if len(recent) else np.nan,
        "return_by_year": json.dumps(yearly, sort_keys=True),
        "leave_one_year_out_worst_cagr": min(leave_out) if leave_out else np.nan,
        "largest_trade_contribution": float(z.net_return.max()),
        "top_five_profit_share": float(positive.nlargest(5).sum() / positive_total) if positive_total > 0 else np.nan,
        "average_after_removing_best_five": float(without_best.mean()) if len(without_best) else np.nan,
        "shrunk_net_average_trade": shrunk, "universe_template_net_average_trade": universe_mean - cost,
    }
    broad = (n >= 50 and len(years) >= 4 and result["average_after_removing_best_five"] > 0
             and result["recent_average_net_trade"] > 0 and shrunk > result["universe_template_net_average_trade"])
    result["promotion_status"] = "eligible_for_walk_forward" if broad else "insufficient_symbol_robustness"
    return result


def _cagr(series: pd.Series) -> float:
    total = float((1 + series.fillna(0)).prod()); years = len(series) / 252
    return float(total ** (1 / years) - 1) if total > 0 and years > 0 else -1.0


def _max_drawdown(series: pd.Series) -> float:
    equity = (1 + series.fillna(0)).cumprod(); return float((equity / equity.cummax() - 1).min())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Phase 2 symbol-specialization diagnostics")
    parser.add_argument("config"); args = parser.parse_args(argv)
    print(json.dumps(run_symbol_search(SymbolConfig.from_yaml(args.config)), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
