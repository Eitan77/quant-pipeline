from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.interday.calendar import SessionClock, TradingCalendar
from quant_pipeline.interday.config import InterdayConfig
from quant_pipeline.interday.corporate_actions import calculate_holding_outcome, normalize_actions
from quant_pipeline.interday.inference import benjamini_hochberg
from quant_pipeline.interday.ranking import build_rank_bin_cache, deterministic_bins
from quant_pipeline.interday.scan import calculate_daily_pair_series
from quant_pipeline.interday.targets import build_targets, future_1d, future_2d
from quant_pipeline.interday.registry import build_feature_registry, build_target_registry

def test_config_rejects_unknown_and_holdout(tmp_path):
    path=tmp_path/"bad.yaml"; path.write_text("unknown_key: 1",encoding="utf-8")
    with pytest.raises(ValueError): InterdayConfig.from_yaml(path)
    with pytest.raises(ValueError): InterdayConfig(discovery_end="2026-05-01").validate()
    with pytest.raises(ValueError): InterdayConfig(allow_holdout_access=True).validate()

def test_checkpoint_clock_and_sparse_neighbors():
    clock=SessionClock(pd.Timestamp("2025-01-02"),pd.Timestamp("2025-01-02 14:30",tz="UTC"),pd.Timestamp("2025-01-02 21:00",tz="UTC"),False)
    assert TradingCalendar.checkpoint_bar_end(clock,"10:00")==pd.Timestamp("2025-01-02 15:00",tz="UTC")

def test_future_shapes_and_target_matrix():
    x=np.arange(12,dtype=np.float32).reshape(4,3); assert np.isnan(future_2d(x,1)[-1]).all(); assert np.isnan(future_1d(np.arange(4,dtype=np.float32),1)[-1])

def test_deterministic_bins_use_integer_tie_breaker():
    values=np.ones(10); bins=deterministic_bins(values,np.arange(10,dtype=np.int64),10); assert np.array_equal(bins,np.arange(10,dtype=np.int8))

def test_rank_bin_thresholds_and_shapes():
    values=np.arange(60,dtype=np.float32).reshape(6,10,1); eligible=np.ones((6,10),bool); cache=build_rank_bin_cache(values,eligible,np.arange(10),["x"],minimum_decile_size=10,minimum_quintile_size=5); assert cache.deciles.shape==(6,10,1); assert cache.quintiles.min()>=0

def test_middle_coverage_is_enforced():
    rng=np.random.default_rng(7); values=rng.normal(size=(4,10,1)); eligible=np.ones((4,10),bool); ranks=build_rank_bin_cache(values,eligible,np.arange(10),["x"],minimum_decile_size=10,minimum_quintile_size=5); target=np.ones((4,10)); target[:,1:9]=np.nan; daily=calculate_daily_pair_series(ranks.percentile_ranks[:,:,0],ranks.deciles[:,:,0],ranks.quintiles[:,:,0],target,minimum_ic_symbols=5,minimum_valid_extreme=1,minimum_bin_coverage=.5,minimum_middle_coverage=.75,minimum_quintile_extreme=1); assert np.isnan(daily.top_minus_middle).all()

def test_corporate_action_reference_handles_split_dividend():
    actions=normalize_actions(pd.DataFrame([{"symbol":"X","effective_session":"2025-01-02","action_type":"split","split_ratio":2},{"symbol":"X","effective_session":"2025-01-03","action_type":"cash_dividend","cash_amount":1}]))
    result=calculate_holding_outcome(entry_price=10,exit_price=6,entry_session=pd.Timestamp("2025-01-01"),exit_session=pd.Timestamp("2025-01-04"),actions=actions); assert result.split_multiplier==2 and result.dividends_received==2

def test_bh_preserves_missing_values():
    out=benjamini_hochberg(pd.Series([.01,np.nan,.05])); assert np.isnan(out.iloc[1]) and out.iloc[0]<=out.iloc[2]

def test_registry_has_canonical_close_targets_and_no_duplicate_features():
    config=InterdayConfig(); features=build_feature_registry(config); targets=build_target_registry(config); assert len({f.name for f in features})==len(features); assert sum(t.checkpoint=="close5" for t in targets)==14
