from __future__ import annotations

import pandas as pd
from dataclasses import replace

from quant_pipeline.stage2_persistence import PersistenceConfig, mark_persistence, persistence_specs


def _frame() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2024-01-02", periods=2, freq="B"):
        for symbol, initial, confirm in (("A", .9, .95), ("B", .8, .7), ("C", .2, .3), ("D", .1, .05)):
            rows.append({"session_date": day, "symbol": symbol, "initial_signal": initial, "signal": confirm})
    return pd.DataFrame(rows)


def test_persistence_grid_is_bounded_and_unique() -> None:
    config = PersistenceConfig("phase1", "phase2")
    specs = persistence_specs(config)
    assert len(specs) == 40
    assert len({row[0].strategy_id for row in specs}) == 40


def test_persistence_requires_same_tail_at_both_times() -> None:
    config = PersistenceConfig("phase1", "phase2")
    spec = replace(persistence_specs(config)[3][0], positions=1)
    marked = mark_persistence(_frame(), spec, "persistent")
    chosen = marked.loc[marked.side.ne(0), ["symbol", "side"]]
    assert set(chosen.itertuples(index=False, name=None)) == {("A", 1), ("D", -1)}


def test_strengthening_requires_rank_improvement() -> None:
    config = PersistenceConfig("phase1", "phase2")
    spec = replace([row[0] for row in persistence_specs(config) if row[3] == "strengthening" and row[0].selection == "top_n"][0], positions=1)
    marked = mark_persistence(_frame(), spec, "strengthening")
    assert set(marked.loc[marked.side.ne(0), "symbol"]) == {"A", "D"}
