from __future__ import annotations

import pandas as pd

from quant_pipeline.stage2_walkforward import WalkForwardConfig, evaluate_walk_forward


def test_walk_forward_uses_only_prior_year_specialists() -> None:
    rows = []
    for year in range(2019, 2027):
        for i in range(20):
            day = pd.Timestamp(year=year, month=1, day=2) + pd.Timedelta(days=i)
            rows.append({"symbol": "A", "side": 1, "session_date": day, "gross_trade_return": .01})
            rows.append({"symbol": "B", "side": -1, "session_date": day, "gross_trade_return": -.01 if year <= 2021 else .10})
    trades = pd.DataFrame(rows); dates = pd.Index(sorted(trades.session_date.unique()))
    metrics, selected = evaluate_walk_forward(trades, dates, "test", 1, 0, WalkForwardConfig("run", minimum_history_trades=5))
    first_fold = selected.loc[selected.session_date.dt.year.eq(2022)]
    assert set(first_fold.symbol) == {"A"}
    assert metrics["trade_count"] > 0
