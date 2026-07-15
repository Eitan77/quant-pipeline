from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from .holdout import assert_pre_holdout_frame
from .stage2 import FEATURE_BLOCKS, TARGET_BLOCKS, Phase2BaselineRunner, Stage2Config, _add_minutes, _preliminary_classification
from .strategy import StrategySpec, evaluate_preselected_strategy, evaluate_strategy


@dataclass(frozen=True)
class PersistenceConfig(Stage2Config):
    persistence_trial_budget: int = 40

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PersistenceConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values["cost_grid_bps"] = tuple(values.get("cost_grid_bps", cls.cost_grid_bps))
        return cls(**values)


class PersistenceRunner(Phase2BaselineRunner):
    config: PersistenceConfig

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        specs = persistence_specs(self.config)
        metrics: list[dict] = []; trades: list[pd.DataFrame] = []; daily: list[pd.DataFrame] = []; ledger: list[dict] = []
        for index, (spec, confirm_time, entry_time, mode) in enumerate(specs, start=1):
            try:
                frame = self._load_persistence_frame(confirm_time, entry_time, spec.holding_period_minutes)
                selected = mark_persistence(frame, spec, mode)
                summary, trade_table, daily_table = evaluate_preselected_strategy(selected, spec, self.config.sealed_holdout_start)
                comparator = self._one_time_comparator(frame, spec)
                summary.update({
                    "confirmation_time": confirm_time,
                    "entry_time": entry_time,
                    "persistence_mode": mode,
                    "one_time_gross_average_portfolio_return": comparator["gross_average_portfolio_return"],
                    "one_time_net_cagr_2bps": comparator["net_cagr_2bps"],
                    "incremental_gross_edge_bps": (summary["gross_average_portfolio_return"] - comparator["gross_average_portfolio_return"]) * 10_000,
                    "trade_count_reduction_fraction": 1 - summary["trade_count"] / max(comparator["trade_count"], 1),
                    "classification": _preliminary_classification(summary),
                })
                metrics.append(summary); trades.append(trade_table); daily.append(daily_table)
                ledger.append({"strategy_id": spec.strategy_id, "status": "completed", "error": ""})
            except Exception as exc:
                ledger.append({"strategy_id": spec.strategy_id, "status": "failed", "error": str(exc)})
                raise RuntimeError(f"Persistence trial failed: {spec.strategy_id}") from exc
            print(f"[{index}/{len(specs)}] {spec.strategy_id}", flush=True)
        results = pd.DataFrame(metrics).sort_values("net_cagr_2bps", ascending=False)
        trade_frame = pd.concat(trades, ignore_index=True); daily_frame = pd.concat(daily, ignore_index=True)
        for name, frame in (("results", results), ("trades", trade_frame), ("daily", daily_frame)):
            assert_pre_holdout_frame(frame, self.config.sealed_holdout_start, f"Phase 2 persistence {name}")
        results.to_csv(self.output / "persistence_leaderboard.csv", index=False)
        pd.DataFrame(ledger).to_csv(self.output / "persistence_trial_ledger.csv", index=False)
        trade_frame.to_parquet(self.output / "persistence_trades.parquet", index=False)
        daily_frame.to_parquet(self.output / "persistence_daily_portfolios.parquet", index=False)
        with (self.output / "persistence_strategy_specs.jsonl").open("w", encoding="utf-8") as handle:
            for spec, confirm, entry, mode in specs:
                row = {**spec.as_dict(), "confirmation_time": confirm, "entry_time": entry, "persistence_mode": mode}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest = {
            "phase": "phase2_opening_persistence", "executed_at": datetime.now(timezone.utc).isoformat(),
            "phase1_fingerprint": self.manifest["fingerprint"], "discovery_end": self.manifest["discovery_end"],
            "sealed_holdout_start": self.config.sealed_holdout_start, "holdout_access": False,
            "trial_count": len(specs), "completed_trials": len(metrics), "failed_trials": 0,
            "config": asdict(self.config),
        }
        manifest["configuration_hash"] = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True, default=list).encode()).hexdigest()
        (self.output / "persistence_manifest.json").write_text(json.dumps(manifest, indent=2, default=list), encoding="utf-8")
        return manifest

    def _load_persistence_frame(self, confirm_time: str, entry_time: str, hold: int | None) -> pd.DataFrame:
        key = ("persistence", confirm_time, entry_time, hold)
        if key in self.frame_cache:
            return self.frame_cache[key]
        feature = (self.phase1 / "blocks" / "features" / FEATURE_BLOCKS["session_range_position"]).as_posix()
        target_name, target_column = TARGET_BLOCKS[hold]
        target = (self.phase1 / "blocks" / "targets" / target_name).as_posix()
        query = f"""
        with initial as (
          select symbol, cast(session_date as date) session_date, session_range_position initial_signal
          from read_parquet('{feature}') where analysis_eligible
            and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
            and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '09:35'
        ), confirmation as (
          select symbol, cast(session_date as date) session_date, decision_ts, session_range_position signal
          from read_parquet('{feature}') where analysis_eligible
            and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
            and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '{confirm_time}'
        ), outcomes as (
          select symbol, cast(session_date as date) session_date, entry_ts,
                 exit_ts__{target_column} exit_ts, {target_column} raw_return
          from read_parquet('{target}') where analysis_eligible
            and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
            and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '{entry_time}'
        )
        select c.symbol, c.session_date, c.decision_ts, o.entry_ts, o.exit_ts,
               c.signal, i.initial_signal, o.raw_return
        from confirmation c join initial i using(symbol, session_date) join outcomes o using(symbol, session_date)
        where c.signal is not null and i.initial_signal is not null and o.raw_return is not null
          and o.entry_ts >= c.decision_ts and o.exit_ts > o.entry_ts
        """
        con = duckdb.connect()
        try:
            con.execute("set threads=8"); frame = con.execute(query).fetchdf()
        finally:
            con.close()
        assert_pre_holdout_frame(frame, self.config.sealed_holdout_start, f"Phase 2 persistence frame {key}")
        self.frame_cache[key] = frame
        return frame

    def _one_time_comparator(self, frame: pd.DataFrame, spec: StrategySpec) -> dict:
        comparator = StrategySpec(**{**spec.as_dict(), "strategy_id": spec.strategy_id + "__one_time", "required_confirmation": None})
        summary, _, _ = evaluate_strategy(frame, comparator, self.config.sealed_holdout_start)
        return summary


def mark_persistence(frame: pd.DataFrame, spec: StrategySpec, mode: str) -> pd.DataFrame:
    work = frame.sort_values(["session_date", "symbol"], kind="mergesort").reset_index(drop=True).copy()
    initial = work.groupby("session_date", sort=True).initial_signal
    confirm = work.groupby("session_date", sort=True).signal
    if spec.selection == "quantile":
        q = float(spec.quantile); initial_pct = initial.rank(method="first", pct=True); confirm_pct = confirm.rank(method="first", pct=True)
        high = initial_pct.gt(1 - q) & confirm_pct.gt(1 - q)
        low = initial_pct.le(q) & confirm_pct.le(q)
    else:
        n = int(spec.positions); initial_high = initial.rank(method="first", ascending=False); confirm_high = confirm.rank(method="first", ascending=False)
        initial_low = initial.rank(method="first", ascending=True); confirm_low = confirm.rank(method="first", ascending=True)
        high = initial_high.le(n) & confirm_high.le(n); low = initial_low.le(n) & confirm_low.le(n)
        initial_pct = initial.rank(method="first", pct=True); confirm_pct = confirm.rank(method="first", pct=True)
    if mode == "strengthening":
        high &= work.signal.gt(work.initial_signal); low &= work.signal.lt(work.initial_signal)
    work["side"] = np.select([high, low], [1, -1], default=0)
    return work


def persistence_specs(config: PersistenceConfig) -> list[tuple[StrategySpec, str, str, str]]:
    specs: list[tuple[StrategySpec, str, str, str]] = []
    selectors = (("quantile", None, .20), ("quantile", None, .10), ("quantile", None, .05), ("top_n", 3, None))
    for confirm in ("09:50", "09:55"):
        for entry in ("09:55", "10:00"):
            for hold in (60, 120):
                for selection, positions, quantile in selectors:
                    specs.append((_persistence_spec(config, confirm, entry, hold, selection, positions, quantile, "persistent"), confirm, entry, "persistent"))
    for confirm in ("09:50", "09:55"):
        for entry in ("09:55", "10:00"):
            for selection, positions, quantile in (("quantile", None, .10), ("top_n", 3, None)):
                specs.append((_persistence_spec(config, confirm, entry, 120, selection, positions, quantile, "strengthening"), confirm, entry, "strengthening"))
    if len(specs) != config.persistence_trial_budget:
        raise ValueError(f"Persistence grid has {len(specs)} trials, expected {config.persistence_trial_budget}")
    return specs


def _persistence_spec(config: PersistenceConfig, confirm: str, entry: str, hold: int, selection: str,
                      positions: int | None, quantile: float | None, mode: str) -> StrategySpec:
    selector = f"q{int(quantile * 100):02d}" if quantile else f"n{positions}"
    strategy_id = f"opening_persistence__0935_{confirm.replace(':','')}__entry{entry.replace(':','')}__{hold}m__{selector}__{mode}"
    return StrategySpec(
        strategy_id=strategy_id, family="opening_range_position_persistence",
        economic_hypothesis="Repeated extreme opening range position identifies sustained buying or selling pressure.",
        decision_time=confirm, signal="session_range_position", direction="long_short", selection=selection,
        positions=positions, quantile=quantile,
        entry_delay_minutes=int((pd.Timestamp('2000-01-01 '+entry)-pd.Timestamp('2000-01-01 '+confirm)).total_seconds()//60),
        exit_rule="fixed_holding_period", holding_period_minutes=hold, cost_grid_bps=config.cost_grid_bps,
        required_confirmation=f"same extreme tail at 09:35 and {confirm}; mode={mode}",
        concentration_mode="concentrated" if positions == 3 else "diversified",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Phase 2 opening-persistence research")
    parser.add_argument("config"); args = parser.parse_args(argv)
    manifest = PersistenceRunner(PersistenceConfig.from_yaml(args.config)).run()
    print(json.dumps(manifest, indent=2, default=list)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
