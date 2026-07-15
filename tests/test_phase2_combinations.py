from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quant_pipeline.stage2_combinations import CombinationConfig, combination_specs, mark_combination


def _frame() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2024-01-02", periods=2, freq="B"):
        for symbol, signal, confirm in (("A", .9, .9), ("B", .8, .2), ("C", .2, .8), ("D", .1, .1)):
            rows.append({"session_date": day, "symbol": symbol, "signal": signal, "vwap_slope": confirm})
    return pd.DataFrame(rows)


def test_combination_grid_is_bounded_and_unique() -> None:
    specs = combination_specs(CombinationConfig("phase1", "phase2"))
    assert len(specs) == 60
    assert len({row[0].strategy_id for row in specs}) == 60


def test_directional_confirmation_is_side_aware() -> None:
    config = CombinationConfig("phase1", "phase2")
    original = [row for row in combination_specs(config) if row[1] == "vwap_slope" and row[2] == .75 and row[0].selection == "top_n"][0]
    spec = replace(original[0], positions=1)
    marked = mark_combination(_frame(), spec, original[1], original[2])
    assert set(marked.loc[marked.side.ne(0), ["symbol", "side"]].itertuples(index=False, name=None)) == {("A", 1), ("D", -1)}
