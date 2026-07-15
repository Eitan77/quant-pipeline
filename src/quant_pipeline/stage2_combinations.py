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

from .holdout import assert_pre_holdout_frame, assert_pre_holdout_parquet
from .stage2 import FEATURE_BLOCKS, TARGET_BLOCKS, Phase2BaselineRunner, Stage2Config, _preliminary_classification
from .strategy import StrategySpec, evaluate_preselected_strategy, evaluate_strategy


CONFIRMATION_BLOCKS = {
    "relative_volume": ("feature_027_050122f146.parquet", "tod_relative_volume_20", "high"),
    "vwap_slope": ("feature_022_cec864b8ac.parquet", "vwap_slope", "directional"),
    "market_adjusted_return": ("feature_032_67ed1b2937.parquet", "stock_minus_market_return_1", "directional"),
    "volatility_normalized_return": ("feature_003_c135c6ac83.parquet", "return_vol_ratio_4", "directional"),
    "opening_gap": ("feature_021_2d96de588a.parquet", "overnight_gap", "directional"),
}


@dataclass(frozen=True)
class CombinationConfig(Stage2Config):
    combination_trial_budget: int = 60

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CombinationConfig":
        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values["cost_grid_bps"] = tuple(values.get("cost_grid_bps", cls.cost_grid_bps))
        return cls(**values)


class CombinationRunner(Phase2BaselineRunner):
    config: CombinationConfig

    def _validate_inputs(self) -> dict:
        manifest = super()._validate_inputs(); hashes = set()
        for filename, _, _ in CONFIRMATION_BLOCKS.values():
            path = self.phase1 / "blocks" / "features" / filename
            assert_pre_holdout_parquet(path, self.config.sealed_holdout_start, f"Phase 2 combination input {filename}", verify_key_rows=False)
            meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8"))
            if meta["fingerprint"] != manifest["fingerprint"]:
                raise ValueError(f"Fingerprint mismatch: {filename}")
            hashes.add(meta["row_key_hash"])
        if len(hashes) != 1:
            raise ValueError("Combination feature row-key hashes do not match")
        return manifest

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True); specs = combination_specs(self.config)
        metrics = []; trades = []; daily = []; ledger = []
        for index, (spec, feature, threshold) in enumerate(specs, start=1):
            try:
                frame = self._load_combination_frame(spec.holding_period_minutes)
                selected = mark_combination(frame, spec, feature, threshold)
                summary, trade_table, daily_table = evaluate_preselected_strategy(selected, spec, self.config.sealed_holdout_start)
                base_spec = StrategySpec(**{**spec.as_dict(), "strategy_id": spec.strategy_id + "__component_a", "required_confirmation": None})
                base, _, _ = evaluate_strategy(frame, base_spec, self.config.sealed_holdout_start)
                _, confirm_column, behavior = CONFIRMATION_BLOCKS[feature]
                if behavior == "directional":
                    component = frame.copy(); component["signal"] = component[confirm_column]
                    b_spec = StrategySpec(**{**spec.as_dict(), "strategy_id": spec.strategy_id + "__component_b", "signal": confirm_column, "required_confirmation": None})
                    component_b, _, _ = evaluate_strategy(component, b_spec, self.config.sealed_holdout_start)
                    component_b_gross = component_b.get("gross_average_portfolio_return", np.nan)
                else:
                    component_b_gross = np.nan
                summary.update({
                    "confirmation_feature": feature, "confirmation_percentile": threshold,
                    "component_a_gross_average_portfolio_return": base["gross_average_portfolio_return"],
                    "component_b_gross_average_portfolio_return": component_b_gross,
                    "incremental_gross_edge_bps": (summary["gross_average_portfolio_return"] - base["gross_average_portfolio_return"]) * 10_000,
                    "trade_count_reduction_fraction": 1 - summary["trade_count"] / max(base["trade_count"], 1),
                    "classification": _preliminary_classification(summary),
                })
                metrics.append(summary); trades.append(trade_table); daily.append(daily_table)
                ledger.append({"strategy_id": spec.strategy_id, "status": "completed", "error": ""})
            except Exception as exc:
                ledger.append({"strategy_id": spec.strategy_id, "status": "failed", "error": str(exc)})
                raise RuntimeError(f"Combination trial failed: {spec.strategy_id}") from exc
            print(f"[{index}/{len(specs)}] {spec.strategy_id}", flush=True)
        results = pd.DataFrame(metrics).sort_values("net_cagr_2bps", ascending=False)
        trade_frame = pd.concat(trades, ignore_index=True); daily_frame = pd.concat(daily, ignore_index=True)
        for name, frame in (("results", results), ("trades", trade_frame), ("daily", daily_frame)):
            assert_pre_holdout_frame(frame, self.config.sealed_holdout_start, f"Phase 2 combinations {name}")
        results.to_csv(self.output / "combination_leaderboard.csv", index=False)
        pd.DataFrame(ledger).to_csv(self.output / "combination_trial_ledger.csv", index=False)
        trade_frame.to_parquet(self.output / "combination_trades.parquet", index=False)
        daily_frame.to_parquet(self.output / "combination_daily_portfolios.parquet", index=False)
        with (self.output / "combination_strategy_specs.jsonl").open("w", encoding="utf-8") as handle:
            for spec, feature, threshold in specs:
                handle.write(json.dumps({**spec.as_dict(), "confirmation_feature": feature, "confirmation_percentile": threshold}, sort_keys=True) + "\n")
        manifest = {
            "phase": "phase2_focused_opening_combinations", "executed_at": datetime.now(timezone.utc).isoformat(),
            "phase1_fingerprint": self.manifest["fingerprint"], "discovery_end": self.manifest["discovery_end"],
            "sealed_holdout_start": self.config.sealed_holdout_start, "holdout_access": False,
            "trial_count": len(specs), "completed_trials": len(metrics), "failed_trials": 0, "config": asdict(self.config),
        }
        manifest["configuration_hash"] = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True, default=list).encode()).hexdigest()
        (self.output / "combination_manifest.json").write_text(json.dumps(manifest, indent=2, default=list), encoding="utf-8")
        return manifest

    def _load_combination_frame(self, hold: int) -> pd.DataFrame:
        key = ("combinations", hold)
        if key in self.frame_cache:
            return self.frame_cache[key]
        initial_path = (self.phase1 / "blocks" / "features" / FEATURE_BLOCKS["close_location"]).as_posix()
        target_name, target_column = TARGET_BLOCKS[hold]; target = (self.phase1 / "blocks" / "targets" / target_name).as_posix()
        ctes = [f"""initial as (
          select symbol, cast(session_date as date) session_date, decision_ts, close_location signal
          from read_parquet('{initial_path}') where analysis_eligible and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
            and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '09:35')"""]
        aliases = []
        for i, (_, (filename, column, _)) in enumerate(CONFIRMATION_BLOCKS.items()):
            alias = f"c{i}"; aliases.append((alias, column)); path = (self.phase1 / "blocks" / "features" / filename).as_posix()
            ctes.append(f"""{alias} as (
              select symbol, cast(session_date as date) session_date, {column}
              from read_parquet('{path}') where analysis_eligible and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
                and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '09:55')""")
        ctes.append(f"""outcomes as (
          select symbol, cast(session_date as date) session_date, entry_ts, exit_ts__{target_column} exit_ts, {target_column} raw_return
          from read_parquet('{target}') where analysis_eligible and cast(session_date as date) < date '{self.config.sealed_holdout_start}'
            and strftime(timezone('America/New_York', decision_ts), '%H:%M') = '10:00')""")
        joins = " ".join(f"join {alias} using(symbol, session_date)" for alias, _ in aliases)
        columns = ", ".join(f"{alias}.{column}" for alias, column in aliases)
        query = f"""with {', '.join(ctes)}
        select i.symbol, i.session_date, i.decision_ts, o.entry_ts, o.exit_ts, i.signal, o.raw_return, {columns}
        from initial i {joins} join outcomes o using(symbol, session_date)
        where i.signal is not null and o.raw_return is not null and o.entry_ts >= i.decision_ts and o.exit_ts > o.entry_ts"""
        con = duckdb.connect()
        try:
            con.execute("set threads=8"); frame = con.execute(query).fetchdf()
        finally:
            con.close()
        assert_pre_holdout_frame(frame, self.config.sealed_holdout_start, f"Phase 2 combination frame {hold}")
        self.frame_cache[key] = frame; return frame


def mark_combination(frame: pd.DataFrame, spec: StrategySpec, feature: str, threshold: float) -> pd.DataFrame:
    work = frame.sort_values(["session_date", "symbol"], kind="mergesort").reset_index(drop=True).copy()
    signal = work.groupby("session_date", sort=True).signal
    if spec.selection == "quantile":
        pct = signal.rank(method="first", pct=True); high = pct.gt(1 - float(spec.quantile)); low = pct.le(float(spec.quantile))
    else:
        high = signal.rank(method="first", ascending=False).le(int(spec.positions)); low = signal.rank(method="first", ascending=True).le(int(spec.positions))
    _, column, behavior = CONFIRMATION_BLOCKS[feature]
    confirm_pct = work.groupby("session_date", sort=True)[column].rank(method="first", pct=True)
    valid = work[column].notna()
    if behavior == "high":
        valid &= confirm_pct.ge(threshold)
        high &= valid; low &= valid
    else:
        high &= valid & confirm_pct.ge(threshold)
        low &= valid & confirm_pct.le(1 - threshold)
    work["side"] = np.select([high, low], [1, -1], default=0)
    return work


def combination_specs(config: CombinationConfig) -> list[tuple[StrategySpec, str, float]]:
    specs = []
    for feature in CONFIRMATION_BLOCKS:
        for threshold in (.50, .75, .90):
            for hold in (60, 120):
                for selection, positions, quantile in (("quantile", None, .10), ("top_n", 3, None)):
                    selector = "q10" if quantile else "n3"
                    sid = f"first_bar_plus_{feature}__0955__entry1000__{hold}m__{selector}__p{int(threshold*100)}"
                    spec = StrategySpec(
                        strategy_id=sid, family="focused_opening_combination",
                        economic_hypothesis=f"Extreme first-bar close location continues when confirmed by {feature}.",
                        decision_time="09:55", signal="close_location", direction="long_short", selection=selection,
                        positions=positions, quantile=quantile, entry_delay_minutes=5, exit_rule="fixed_holding_period",
                        holding_period_minutes=hold, cost_grid_bps=config.cost_grid_bps,
                        required_confirmation=f"{feature} percentile {threshold:.2f}",
                        concentration_mode="concentrated" if positions == 3 else "diversified")
                    specs.append((spec, feature, threshold))
    if len(specs) != config.combination_trial_budget:
        raise ValueError(f"Combination grid has {len(specs)} trials, expected {config.combination_trial_budget}")
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded Phase 2 opening-confirmation combinations")
    parser.add_argument("config"); args = parser.parse_args(argv)
    manifest = CombinationRunner(CombinationConfig.from_yaml(args.config)).run()
    print(json.dumps(manifest, indent=2, default=list)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
