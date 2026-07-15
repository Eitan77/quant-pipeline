from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .holdout import assert_pre_holdout_frame, assert_pre_holdout_parquet
from .strategy import StrategySpec, evaluate_strategy


@dataclass(frozen=True)
class Stage2Config:
    phase1_run_dir: str
    output_dir: str
    sealed_holdout_start: str = "2026-05-01"
    cost_grid_bps: tuple[float, ...] = (0, 1, 2, 4, 6, 10)
    first_bar_trial_budget: int = 50
    session_range_trial_budget: int = 50

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Stage2Config":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values["cost_grid_bps"] = tuple(values.get("cost_grid_bps", cls.cost_grid_bps))
        return cls(**values)


TARGET_BLOCKS = {
    30: ("target_001.parquet", "fwd_return_30m"),
    60: ("target_002.parquet", "fwd_return_60m"),
    120: ("target_005.parquet", "fwd_return_120m"),
    None: ("target_019.parquet", "fwd_return_eod"),
}
FEATURE_BLOCKS = {
    "close_location": "feature_021_2d96de588a.parquet",
    "session_range_position": "feature_022_cec864b8ac.parquet",
}


class Phase2BaselineRunner:
    def __init__(self, config: Stage2Config):
        self.config = config
        self.phase1 = Path(config.phase1_run_dir)
        self.output = Path(config.output_dir)
        self.frame_cache: dict[tuple, pd.DataFrame] = {}
        self.manifest = self._validate_inputs()

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        specs = first_bar_specs(self.config) + session_range_specs(self.config)
        metrics, trades, daily, ledger = [], [], [], []
        for index, spec in enumerate(specs, start=1):
            try:
                frame = self._load_frame(spec)
                summary, trade_table, daily_table = evaluate_strategy(frame, spec, self.config.sealed_holdout_start)
                summary["classification"] = _preliminary_classification(summary)
                metrics.append(summary); trades.append(trade_table); daily.append(daily_table)
                ledger.append({"strategy_id": spec.strategy_id, "status": "completed", "error": ""})
            except Exception as exc:
                ledger.append({"strategy_id": spec.strategy_id, "status": "failed", "error": str(exc)})
                raise RuntimeError(f"Phase 2 trial failed: {spec.strategy_id}") from exc
            print(f"[{index}/{len(specs)}] {spec.strategy_id}", flush=True)

        results = pd.DataFrame(metrics)
        trade_frame = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
        daily_frame = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
        ledger_frame = pd.DataFrame(ledger)
        for name, frame in [("results", results), ("trades", trade_frame), ("daily", daily_frame)]:
            if not frame.empty:
                assert_pre_holdout_frame(frame, self.config.sealed_holdout_start, f"Phase 2 {name} output")
        results.sort_values("net_cagr_2bps", ascending=False).to_csv(self.output / "baseline_leaderboard.csv", index=False)
        ledger_frame.to_csv(self.output / "trial_ledger.csv", index=False)
        trade_frame.to_parquet(self.output / "baseline_trades.parquet", index=False)
        daily_frame.to_parquet(self.output / "baseline_daily_portfolios.parquet", index=False)
        with (self.output / "strategy_specs.jsonl").open("w", encoding="utf-8") as handle:
            for spec in specs:
                handle.write(json.dumps(spec.as_dict(), sort_keys=True) + "\n")
        output_manifest = {
            "phase": "phase2_high_return_baselines",
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "phase1_run": str(self.phase1),
            "phase1_fingerprint": self.manifest["fingerprint"],
            "discovery_end": self.manifest["discovery_end"],
            "sealed_holdout_start": self.config.sealed_holdout_start,
            "holdout_access": False,
            "trial_count": len(specs),
            "completed_trials": int(ledger_frame.status.eq("completed").sum()),
            "failed_trials": int(ledger_frame.status.eq("failed").sum()),
            "first_bar_trials": sum(s.family == "first_bar_close_location" for s in specs),
            "session_range_trials": sum(s.family == "session_range_position" for s in specs),
            "config": asdict(self.config),
        }
        output_manifest["configuration_hash"] = hashlib.sha256(json.dumps(output_manifest["config"], sort_keys=True, default=list).encode()).hexdigest()
        (self.output / "manifest.json").write_text(json.dumps(output_manifest, indent=2, default=list), encoding="utf-8")
        return output_manifest

    def _validate_inputs(self) -> dict:
        manifest = json.loads((self.phase1 / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("holdout_access"):
            raise ValueError("Phase 1 manifest indicates holdout access")
        if pd.Timestamp(manifest["discovery_end"]) >= pd.Timestamp(self.config.sealed_holdout_start):
            raise ValueError("Phase 1 discovery reaches the sealed holdout")
        paths = [self.phase1 / "blocks" / "features" / name for name in FEATURE_BLOCKS.values()]
        paths += [self.phase1 / "blocks" / "targets" / name for name, _ in TARGET_BLOCKS.values()]
        hashes = set()
        for path in paths:
            assert_pre_holdout_parquet(path, self.config.sealed_holdout_start, f"Phase 2 input {path.name}", verify_key_rows=False)
            meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8"))
            hashes.add(meta["row_key_hash"])
            if meta["fingerprint"] != manifest["fingerprint"]:
                raise ValueError(f"Fingerprint mismatch: {path.name}")
        if len(hashes) != 1:
            raise ValueError("Phase 2 feature/target row-key hashes do not match")
        return manifest

    def _load_frame(self, spec: StrategySpec) -> pd.DataFrame:
        key = (spec.signal, spec.decision_time, spec.entry_delay_minutes, spec.holding_period_minutes)
        if key in self.frame_cache:
            return self.frame_cache[key]
        feature_path = self.phase1 / "blocks" / "features" / FEATURE_BLOCKS[spec.signal]
        target_name, target_column = TARGET_BLOCKS[spec.holding_period_minutes]
        target_path = self.phase1 / "blocks" / "targets" / target_name
        entry_time = _add_minutes(spec.decision_time, spec.entry_delay_minutes)
        f = feature_path.as_posix(); t = target_path.as_posix()
        query = f"""
        with signals as (
          select symbol, cast(session_date as date) session_date, decision_ts,
                 {spec.signal} signal
          from read_parquet('{f}')
          where analysis_eligible
            and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
            and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '{spec.decision_time}'
        ), outcomes as (
          select symbol, cast(session_date as date) session_date, decision_ts entry_decision_ts,
                 entry_ts, exit_ts__{target_column} exit_ts, {target_column} raw_return
          from read_parquet('{t}')
          where analysis_eligible
            and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
            and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '{entry_time}'
        )
        select s.symbol, s.session_date, s.decision_ts, o.entry_ts, o.exit_ts, s.signal, o.raw_return
        from signals s join outcomes o using(symbol, session_date)
        where s.signal is not null and o.raw_return is not null
          and o.entry_ts >= s.decision_ts and o.exit_ts > o.entry_ts
        """
        con = duckdb.connect()
        try:
            con.execute("set threads=8")
            frame = con.execute(query).fetchdf()
        finally:
            con.close()
        assert_pre_holdout_frame(frame, self.config.sealed_holdout_start, f"Phase 2 loaded frame {key}")
        self.frame_cache[key] = frame
        return frame


def first_bar_specs(config: Stage2Config) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    hypothesis = "The first completed bar's close location predicts same-direction intraday continuation."
    for delay in (5, 10, 15):
        for hold in (30, 60, 120, None):
            for quantile in (.20, .10, .05):
                specs.append(_spec("first_bar_close_location", "close_location", "09:35", delay, hold,
                                   "long_short", "quantile", None, quantile, config, hypothesis))
    for hold in (30, 60, 120, None):
        for positions in (3, 1):
            specs.append(_spec("first_bar_close_location", "close_location", "09:35", 5, hold,
                               "long_short", "top_n", positions, None, config, hypothesis))
    for direction in ("long_only", "short_only"):
        for positions in (1, 3, 5):
            specs.append(_spec("first_bar_close_location", "close_location", "09:35", 5, 120,
                               direction, "top_n", positions, None, config, hypothesis))
    if len(specs) != config.first_bar_trial_budget:
        raise ValueError(f"First-bar grid has {len(specs)} trials, expected budget {config.first_bar_trial_budget}")
    return specs


def session_range_specs(config: Stage2Config) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    hypothesis = "Stocks near their session-range extremes continue in the same direction."
    times = ("09:35", "09:40", "09:50", "09:55", "10:25")
    for time in times:
        for hold in (30, 60, 120, None):
            specs.append(_spec("session_range_position", "session_range_position", time, 0, hold,
                               "long_short", "quantile", None, .10, config, hypothesis))
            specs.append(_spec("session_range_position", "session_range_position", time, 0, hold,
                               "long_short", "top_n", 3, None, config, hypothesis))
    for time in ("09:35", "09:55"):
        for selection, positions, quantile in (("quantile", None, .20), ("quantile", None, .05), ("top_n", 1, None)):
            specs.append(_spec("session_range_position", "session_range_position", time, 0, 120,
                               "long_short", selection, positions, quantile, config, hypothesis))
        specs.append(_spec("session_range_position", "session_range_position", time, 0, 120,
                           "long_only", "top_n", 1, None, config, hypothesis))
        specs.append(_spec("session_range_position", "session_range_position", time, 0, 120,
                           "short_only", "top_n", 1, None, config, hypothesis))
    if len(specs) != config.session_range_trial_budget:
        raise ValueError(f"Session-range grid has {len(specs)} trials, expected budget {config.session_range_trial_budget}")
    return specs


def _spec(family: str, signal: str, time: str, delay: int, hold: int | None, direction: str,
          selection: str, positions: int | None, quantile: float | None, config: Stage2Config,
          hypothesis: str) -> StrategySpec:
    exit_name = "eod" if hold is None else f"{hold}m"
    selector = f"q{int(quantile * 100):02d}" if quantile else f"n{positions}"
    strategy_id = f"{family}__{time.replace(':','')}__d{delay}__{exit_name}__{direction}__{selector}"
    concentration = "full_port_single_stock" if positions == 1 and direction != "long_short" else "concentrated" if positions in (1, 3) else "diversified"
    return StrategySpec(strategy_id, family, hypothesis, time, signal, direction, selection, positions,
                        quantile, delay, "end_of_day" if hold is None else "fixed_holding_period", hold,
                        cost_grid_bps=config.cost_grid_bps, concentration_mode=concentration)


def _add_minutes(value: str, minutes: int) -> str:
    stamp = pd.Timestamp(f"2000-01-01 {value}") + pd.Timedelta(minutes=minutes)
    return stamp.strftime("%H:%M")


def _preliminary_classification(summary: dict) -> str:
    if summary.get("trade_count", 0) < 100:
        return "needs_more_data"
    net = summary.get("net_cagr_2bps", -1)
    if net >= .20 and summary.get("maximum_drawdown", -1) > -.50:
        return "high_return_in_sample_only"
    if summary.get("gross_cagr", -1) >= .20 and net > 0:
        return "high_return_but_fragile"
    if net > 0:
        return "interesting_but_small"
    return "rejected"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Phase 2 baseline strategy research")
    parser.add_argument("config")
    args = parser.parse_args(argv)
    config = Stage2Config.from_yaml(args.config)
    manifest = Phase2BaselineRunner(config).run()
    print(json.dumps(manifest, indent=2, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
