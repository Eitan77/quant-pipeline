from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from .holdout import assert_pre_holdout_frame, assert_pre_holdout_parquet
from .stage2 import Phase2BaselineRunner, Stage2Config, _preliminary_classification
from .strategy import StrategySpec, evaluate_strategy


SIGNAL_BLOCKS = {
    30: ("feature_006_12e0ab3dec.parquet", "return_6", "feature_032_67ed1b2937.parquet", "market_return_6"),
    60: ("feature_010_7c72a54ca2.parquet", "return_12", "feature_033_33008a6582.parquet", "market_return_12"),
}
OUTCOME_BLOCKS = {
    15: ("target_000.parquet", "fwd_return_15m"),
    30: ("target_001.parquet", "fwd_return_30m"),
    None: ("target_019.parquet", "fwd_return_eod"),
}


@dataclass(frozen=True)
class AfternoonConfig(Stage2Config):
    afternoon_trial_budget: int = 50

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AfternoonConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values["cost_grid_bps"] = tuple(values.get("cost_grid_bps", cls.cost_grid_bps))
        return cls(**values)


class AfternoonRunner(Phase2BaselineRunner):
    config: AfternoonConfig

    def _validate_inputs(self) -> dict:
        manifest = super()._validate_inputs(); paths = set()
        for stock_file, _, market_file, _ in SIGNAL_BLOCKS.values():
            paths.add(self.phase1 / "blocks" / "features" / stock_file); paths.add(self.phase1 / "blocks" / "features" / market_file)
        for filename, _ in OUTCOME_BLOCKS.values(): paths.add(self.phase1 / "blocks" / "targets" / filename)
        hashes = set()
        for path in paths:
            assert_pre_holdout_parquet(path, self.config.sealed_holdout_start, f"Phase 2 afternoon input {path.name}", verify_key_rows=False)
            meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8"))
            if meta["fingerprint"] != manifest["fingerprint"]: raise ValueError(f"Fingerprint mismatch: {path.name}")
            hashes.add(meta["row_key_hash"])
        if len(hashes) != 1: raise ValueError("Afternoon feature/target row-key hashes do not match")
        return manifest

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True); specs = afternoon_specs(self.config)
        metrics = []; trades = []; daily = []; ledger = []
        for index, (spec, signal_horizon) in enumerate(specs, start=1):
            try:
                frame = self._load_afternoon_frame(spec.decision_time, signal_horizon, spec.holding_period_minutes)
                summary, trade_table, daily_table = evaluate_strategy(frame, spec, self.config.sealed_holdout_start)
                summary["signal_horizon_minutes"] = signal_horizon; summary["classification"] = _preliminary_classification(summary)
                metrics.append(summary); trades.append(trade_table); daily.append(daily_table)
                ledger.append({"strategy_id": spec.strategy_id, "status": "completed", "error": ""})
            except Exception as exc:
                ledger.append({"strategy_id": spec.strategy_id, "status": "failed", "error": str(exc)})
                raise RuntimeError(f"Afternoon trial failed: {spec.strategy_id}") from exc
            print(f"[{index}/{len(specs)}] {spec.strategy_id}", flush=True)
        results = pd.DataFrame(metrics).sort_values("net_cagr_2bps", ascending=False)
        trade_frame = pd.concat(trades, ignore_index=True); daily_frame = pd.concat(daily, ignore_index=True)
        for name, frame in (("results", results), ("trades", trade_frame), ("daily", daily_frame)):
            assert_pre_holdout_frame(frame, self.config.sealed_holdout_start, f"Phase 2 afternoon {name}")
        results.to_csv(self.output / "afternoon_reversal_leaderboard.csv", index=False)
        pd.DataFrame(ledger).to_csv(self.output / "afternoon_reversal_trial_ledger.csv", index=False)
        trade_frame.to_parquet(self.output / "afternoon_reversal_trades.parquet", index=False)
        daily_frame.to_parquet(self.output / "afternoon_reversal_daily.parquet", index=False)
        with (self.output / "afternoon_reversal_specs.jsonl").open("w", encoding="utf-8") as handle:
            for spec, signal_horizon in specs: handle.write(json.dumps({**spec.as_dict(), "signal_horizon_minutes": signal_horizon}, sort_keys=True) + "\n")
        manifest = {"phase": "phase2_afternoon_beta_residual_reversal", "executed_at": datetime.now(timezone.utc).isoformat(),
                    "phase1_fingerprint": self.manifest["fingerprint"], "discovery_end": self.manifest["discovery_end"],
                    "sealed_holdout_start": self.config.sealed_holdout_start, "holdout_access": False,
                    "trial_count": len(specs), "completed_trials": len(metrics), "failed_trials": 0, "config": asdict(self.config)}
        manifest["configuration_hash"] = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True, default=list).encode()).hexdigest()
        (self.output / "afternoon_reversal_manifest.json").write_text(json.dumps(manifest, indent=2, default=list), encoding="utf-8")
        return manifest

    def _load_afternoon_frame(self, decision_time: str, signal_horizon: int, hold: int | None) -> pd.DataFrame:
        key = ("afternoon", decision_time, signal_horizon, hold)
        if key in self.frame_cache: return self.frame_cache[key]
        stock_file, stock_col, market_file, market_col = SIGNAL_BLOCKS[signal_horizon]
        outcome_file, outcome_col = OUTCOME_BLOCKS[hold]
        stock = (self.phase1 / "blocks" / "features" / stock_file).as_posix(); market = (self.phase1 / "blocks" / "features" / market_file).as_posix()
        outcome = (self.phase1 / "blocks" / "targets" / outcome_file).as_posix()
        query = f"""
        with stock as (select symbol, cast(session_date as date) session_date, decision_ts, {stock_col} stock_return
          from read_parquet('{stock}') where analysis_eligible and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
          and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '{decision_time}'),
        market as (select symbol, cast(session_date as date) session_date, {market_col} market_return
          from read_parquet('{market}') where analysis_eligible and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
          and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '{decision_time}'),
        outcomes as (select symbol, cast(session_date as date) session_date, entry_ts, beta_at_decision,
          exit_ts__{outcome_col} exit_ts, {outcome_col} raw_return
          from read_parquet('{outcome}') where analysis_eligible and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
          and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '{decision_time}')
        select s.symbol, s.session_date, s.decision_ts, o.entry_ts, o.exit_ts,
          (s.stock_return - o.beta_at_decision * m.market_return) signal, o.raw_return
        from stock s join market m using(symbol, session_date) join outcomes o using(symbol, session_date)
        where s.stock_return is not null and m.market_return is not null and o.beta_at_decision is not null
          and o.raw_return is not null and o.entry_ts >= s.decision_ts and o.exit_ts > o.entry_ts
        """
        con = duckdb.connect()
        try: con.execute("set threads=8"); frame = con.execute(query).fetchdf()
        finally: con.close()
        assert_pre_holdout_frame(frame, self.config.sealed_holdout_start, f"Phase 2 afternoon frame {key}")
        self.frame_cache[key] = frame; return frame


def afternoon_specs(config: AfternoonConfig) -> list[tuple[StrategySpec, int]]:
    specs = []; times = ("14:30", "14:45", "15:00", "15:15", "15:30", "15:45")
    for time in times:
        for signal_horizon in (30, 60):
            for hold in (15, 30, None):
                specs.append((_spec(config, time, signal_horizon, hold, "long_short", "quantile", None, .10), signal_horizon))
    for time in times:
        for positions in (3, 1): specs.append((_spec(config, time, 60, None, "long_short", "top_n", positions, None), 60))
    for time in ("15:00", "15:30"):
        specs.append((_spec(config, time, 60, 30, "adaptive", "top_n", 1, None), 60))
    if len(specs) != config.afternoon_trial_budget: raise ValueError(f"Afternoon grid has {len(specs)} trials, expected {config.afternoon_trial_budget}")
    return specs


def _spec(config: AfternoonConfig, time: str, signal_horizon: int, hold: int | None, direction: str,
          selection: str, positions: int | None, quantile: float | None) -> StrategySpec:
    selector = f"q{int(quantile*100):02d}" if quantile else f"n{positions}"; exit_name = "eod" if hold is None else f"{hold}m"
    sid = f"afternoon_beta_residual_reversal__{time.replace(':','')}__sig{signal_horizon}m__{exit_name}__{direction}__{selector}"
    return StrategySpec(strategy_id=sid, family="afternoon_beta_residual_reversal",
        economic_hypothesis="Extreme beta-residual moves partially reverse late in the session.", decision_time=time,
        signal="beta_residual_return", direction=direction, selection=selection, positions=positions, quantile=quantile,
        entry_delay_minutes=0, exit_rule="end_of_day" if hold is None else "fixed_holding_period", holding_period_minutes=hold,
        signal_direction=-1, cost_grid_bps=config.cost_grid_bps,
        concentration_mode="full_port_single_stock" if direction == "adaptive" else "concentrated" if positions in (1,3) else "diversified",
        adaptive_min_history_sessions=60, adaptive_minimum_edge_bps=2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Phase 2 afternoon reversal research"); parser.add_argument("config"); args = parser.parse_args(argv)
    manifest = AfternoonRunner(AfternoonConfig.from_yaml(args.config)).run(); print(json.dumps(manifest, indent=2, default=list)); return 0


if __name__ == "__main__": raise SystemExit(main())
