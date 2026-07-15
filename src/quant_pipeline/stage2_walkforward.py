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
from .stage2_symbols import TEMPLATES


FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025), (2025, 2026))


@dataclass(frozen=True)
class WalkForwardConfig:
    phase2_run_dir: str
    sealed_holdout_start: str = "2026-05-01"
    cost_bps_per_side: float = 2.0
    shrinkage_prior_trades: int = 30
    minimum_history_trades: int = 15
    specialist_pool_sizes: tuple[int, ...] = (3, 5, 10, 20)
    minimum_edges_bps: tuple[float, ...] = (0, 5)
    trial_budget: int = 40

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WalkForwardConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values["specialist_pool_sizes"] = tuple(values.get("specialist_pool_sizes", cls.specialist_pool_sizes))
        values["minimum_edges_bps"] = tuple(values.get("minimum_edges_bps", cls.minimum_edges_bps))
        return cls(**values)


def run_walk_forward(config: WalkForwardConfig) -> dict:
    root = Path(config.phase2_run_dir); trades = pd.read_parquet(root / "baseline_trades.parquet")
    sessions = pd.read_parquet(root / "baseline_daily_portfolios.parquet", columns=["strategy_id", "session_date"])
    assert_pre_holdout_frame(trades, config.sealed_holdout_start, "Phase 2 walk-forward trades")
    assert_pre_holdout_frame(sessions, config.sealed_holdout_start, "Phase 2 walk-forward sessions")
    trades["session_date"] = pd.to_datetime(trades.session_date).dt.normalize()
    sessions["session_date"] = pd.to_datetime(sessions.session_date).dt.normalize()
    rows = []; selected_tables = []
    for template in TEMPLATES:
        z = trades.loc[trades.strategy_id.eq(template)].copy()
        dates = pd.Index(sorted(sessions.loc[sessions.strategy_id.eq(template), "session_date"].unique()))
        for pool_size in config.specialist_pool_sizes:
            for minimum_edge in config.minimum_edges_bps:
                strategy_id = f"wf_specialists__{template}__pool{pool_size}__edge{minimum_edge:g}bps"
                metrics, selected = evaluate_walk_forward(z, dates, template, pool_size, minimum_edge, config)
                metrics["strategy_id"] = strategy_id; rows.append(metrics)
                if not selected.empty:
                    selected_tables.append(selected.assign(strategy_id=strategy_id))
    output = pd.DataFrame(rows)
    if len(output) != config.trial_budget:
        raise ValueError(f"Walk-forward grid produced {len(output)} trials, expected {config.trial_budget}")
    output.sort_values("walk_forward_net_cagr", ascending=False).to_csv(root / "walk_forward_specialist_leaderboard.csv", index=False)
    selected = pd.concat(selected_tables, ignore_index=True) if selected_tables else pd.DataFrame()
    if not selected.empty:
        assert_pre_holdout_frame(selected, config.sealed_holdout_start, "Phase 2 walk-forward selections")
        selected.to_parquet(root / "walk_forward_specialist_trades.parquet", index=False)
    manifest = {
        "phase": "phase2_walk_forward_specialists", "executed_at": datetime.now(timezone.utc).isoformat(),
        "sealed_holdout_start": config.sealed_holdout_start, "holdout_access": False,
        "trial_count": len(output), "folds": [{"train_through": a, "test": b} for a, b in FOLDS],
    }
    (root / "walk_forward_specialist_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def evaluate_walk_forward(trades: pd.DataFrame, opportunity_dates: pd.Index, template: str, pool_size: int,
                          minimum_edge_bps: float, config: WalkForwardConfig) -> tuple[dict, pd.DataFrame]:
    cost = 2 * config.cost_bps_per_side / 10_000; chosen = []; fold_returns = {}
    test_dates_all = []
    for train_through, test_year in FOLDS:
        train = trades.loc[trades.session_date.dt.year.le(train_through)].copy()
        test = trades.loc[trades.session_date.dt.year.eq(test_year)].copy()
        fold_dates = opportunity_dates[pd.to_datetime(opportunity_dates).year == test_year]; test_dates_all.extend(fold_dates)
        if train.empty or test.empty:
            fold_returns[str(test_year)] = 0.0; continue
        train["net"] = train.gross_trade_return - cost
        prior = float(train.net.mean())
        grouped = train.groupby(["symbol", "side"]).net.agg(["size", "mean"]).reset_index()
        grouped = grouped.loc[grouped["size"].ge(config.minimum_history_trades)].copy()
        grouped["expected_net"] = ((grouped["size"] * grouped["mean"] + config.shrinkage_prior_trades * prior)
                                   / (grouped["size"] + config.shrinkage_prior_trades))
        pool = grouped.sort_values(["expected_net", "symbol", "side"], ascending=[False, True, True]).head(pool_size)
        pool = pool.loc[pool.expected_net.gt(minimum_edge_bps / 10_000)]
        candidates = test.merge(pool[["symbol", "side", "expected_net"]], on=["symbol", "side"], how="inner")
        candidates = candidates.sort_values(["session_date", "expected_net", "symbol", "side"], ascending=[True, False, True, True])
        picks = candidates.drop_duplicates("session_date", keep="first").copy(); picks["net_return"] = picks.gross_trade_return - cost
        chosen.append(picks)
        fold_daily = pd.Series(0.0, index=fold_dates)
        if not picks.empty:
            fold_daily.loc[picks.session_date] = picks.net_return.to_numpy()
        fold_returns[str(test_year)] = float((1 + fold_daily).prod() - 1)
    selected = pd.concat(chosen, ignore_index=True) if chosen else pd.DataFrame()
    test_dates = pd.Index(sorted(set(test_dates_all))); daily = pd.Series(0.0, index=test_dates)
    if not selected.empty:
        daily.loc[selected.session_date] = selected.net_return.to_numpy()
    recent = selected.loc[selected.session_date.dt.year.eq(2026), "net_return"] if not selected.empty else pd.Series(dtype=float)
    trade_returns = selected.net_return if not selected.empty else pd.Series(dtype=float)
    cagr = _cagr(daily); worst_fold = min(fold_returns.values()) if fold_returns else np.nan; median_fold = float(np.median(list(fold_returns.values()))) if fold_returns else np.nan
    if cagr >= 1 and worst_fold > -.25:
        classification = "robust_100pct_candidate"
    elif cagr >= .50:
        classification = "high_return_but_fragile"
    elif cagr >= .20 and median_fold > 0 and worst_fold > -.20 and (len(recent) == 0 or recent.sum() > 0):
        classification = "eligible_for_detailed_phase3"
    else:
        classification = "insufficient_walk_forward_return"
    metrics = {
        "template": template, "pool_size": pool_size, "minimum_expected_edge_bps": minimum_edge_bps,
        "walk_forward_net_cagr": cagr, "walk_forward_total_return": float((1 + daily).prod() - 1),
        "trade_count": len(selected), "trades_per_year": len(selected) / max(len(daily) / 252, 1 / 252),
        "average_net_trade": float(trade_returns.mean()) if len(trade_returns) else np.nan,
        "median_net_trade": float(trade_returns.median()) if len(trade_returns) else np.nan,
        "win_rate": float(trade_returns.gt(0).mean()) if len(trade_returns) else np.nan,
        "maximum_drawdown": _max_drawdown(daily), "worst_trade": float(trade_returns.min()) if len(trade_returns) else np.nan,
        "fold_returns": json.dumps(fold_returns, sort_keys=True), "median_fold_return": median_fold,
        "worst_fold_return": worst_fold, "recent_2026_return": float((1 + recent).prod() - 1) if len(recent) else 0.0,
        "selected_symbol_distribution": selected.symbol.value_counts().head(20).to_json() if not selected.empty else "{}",
        "classification": classification,
    }
    return metrics, selected


def _cagr(series: pd.Series) -> float:
    total = float((1 + series.fillna(0)).prod()); years = len(series) / 252
    return float(total ** (1 / years) - 1) if total > 0 and years > 0 else -1.0


def _max_drawdown(series: pd.Series) -> float:
    equity = (1 + series.fillna(0)).cumprod(); return float((equity / equity.cummax() - 1).min())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run causal walk-forward symbol-specialist strategies")
    parser.add_argument("config"); args = parser.parse_args(argv)
    print(json.dumps(run_walk_forward(WalkForwardConfig.from_yaml(args.config)), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
