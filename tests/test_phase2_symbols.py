from __future__ import annotations

import pandas as pd

from quant_pipeline.stage2_symbols import SymbolConfig, symbol_metrics


def test_symbol_metrics_apply_costs_shrinkage_and_best_trade_removal() -> None:
    dates = pd.date_range("2022-01-03", periods=60, freq="B")
    z = pd.DataFrame({"symbol": "A", "session_date": dates, "side": [1] * 60,
                      "gross_trade_return": [.002] * 59 + [.10]})
    result = symbol_metrics(z, pd.Index(dates), "test", .001, SymbolConfig("run"))
    assert result["net_average_trade"] < result["gross_average_trade"]
    assert result["shrunk_net_average_trade"] < result["net_average_trade"]
    assert result["average_after_removing_best_five"] < result["net_average_trade"]
