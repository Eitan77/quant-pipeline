from __future__ import annotations

import pandas as pd
import pytest

from quant_pipeline.stage2 import Stage2Config, _preliminary_classification, first_bar_specs, session_range_specs
from quant_pipeline.strategy import StrategySpec, evaluate_strategy


def _frame() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2024-01-02", periods=4, freq="B"):
        for symbol, signal, outcome in [("A", 4.0, .04), ("B", 3.0, .02), ("C", 2.0, -.02), ("D", 1.0, -.04)]:
            decision = pd.Timestamp(day.date(), tz="America/New_York") + pd.Timedelta(hours=9, minutes=35)
            rows.append({"symbol": symbol, "session_date": day, "decision_ts": decision,
                         "entry_ts": decision + pd.Timedelta(minutes=5),
                         "exit_ts": decision + pd.Timedelta(minutes=35),
                         "signal": signal, "raw_return": outcome})
    return pd.DataFrame(rows)


def _spec(**overrides) -> StrategySpec:
    values = dict(strategy_id="test", family="test", economic_hypothesis="test",
                  decision_time="09:35", signal="signal", direction="long_short",
                  selection="top_n", positions=1, quantile=None, entry_delay_minutes=5,
                  exit_rule="fixed_holding_period", holding_period_minutes=30,
                  cost_grid_bps=(0, 1, 2))
    values.update(overrides)
    return StrategySpec(**values)


def test_long_short_ranking_and_costs() -> None:
    metrics, trades, daily = evaluate_strategy(_frame(), _spec(), "2026-05-01")
    assert metrics["trade_count"] == 8
    assert metrics["gross_average_portfolio_return"] == pytest.approx(.04)
    assert metrics["net_average_portfolio_return_1bps"] == pytest.approx(.0398)
    assert set(trades.groupby("session_date").side.sum()) == {0}
    assert daily.gross_return.eq(.04).all()


def test_full_port_long_and_short_are_separate() -> None:
    long_metrics, _, _ = evaluate_strategy(_frame(), _spec(direction="long_only"), "2026-05-01")
    short_metrics, _, _ = evaluate_strategy(_frame(), _spec(direction="short_only"), "2026-05-01")
    assert long_metrics["gross_average_portfolio_return"] == pytest.approx(.04)
    assert short_metrics["gross_average_portfolio_return"] == pytest.approx(.04)


def test_phase2_rejects_holdout_in_any_timestamp() -> None:
    frame = _frame()
    frame.loc[0, "exit_ts"] = pd.Timestamp("2026-05-01", tz="UTC")
    with pytest.raises(ValueError, match="Sealed holdout"):
        evaluate_strategy(frame, _spec(), "2026-05-01")


def test_baseline_trial_budgets_and_reproducible_ids() -> None:
    config = Stage2Config("phase1", "phase2")
    first = first_bar_specs(config); session = session_range_specs(config)
    assert len(first) == 50
    assert len(session) == 50
    ids = [spec.strategy_id for spec in first + session]
    assert len(ids) == len(set(ids))
    assert all(spec.cost_grid_bps == (0, 1, 2, 4, 6, 10) for spec in first + session)


def test_phase2_yaml_loads() -> None:
    config = Stage2Config.from_yaml("configs/phase2_baselines.yaml")
    assert config.sealed_holdout_start == "2026-05-01"
    assert config.first_bar_trial_budget == 50


def test_preliminary_classification_preserves_positive_results() -> None:
    base = {"trade_count": 500, "maximum_drawdown": -0.25}
    assert _preliminary_classification({**base, "gross_cagr": .21, "net_cagr_2bps": .08}) == "high_return_but_fragile"
    assert _preliminary_classification({**base, "gross_cagr": .12, "net_cagr_2bps": .03}) == "interesting_but_small"
    assert _preliminary_classification({**base, "gross_cagr": .05, "net_cagr_2bps": -.01}) == "rejected"


def test_tied_signal_selection_is_independent_of_input_order() -> None:
    frame = _frame()
    frame["signal"] = 1.0
    first, first_trades, _ = evaluate_strategy(frame, _spec(), "2026-05-01")
    shuffled, shuffled_trades, _ = evaluate_strategy(
        frame.sample(frac=1, random_state=7), _spec(), "2026-05-01"
    )
    assert first["gross_total_return"] == shuffled["gross_total_return"]
    assert first_trades[["session_date", "symbol", "side"]].reset_index(drop=True).equals(
        shuffled_trades[["session_date", "symbol", "side"]].reset_index(drop=True)
    )
