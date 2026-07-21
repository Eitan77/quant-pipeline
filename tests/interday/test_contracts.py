from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.interday.calendar import SessionClock, TradingCalendar
from quant_pipeline.interday.cache import InterdayCacheMismatch, validate_matrix, write_matrix
from quant_pipeline.interday.candidates import apply_interday_fdr
from quant_pipeline.interday.config import InterdayConfig
from quant_pipeline.interday.corporate_actions import build_corporate_action_index, interval_total_return, normalize_actions
from quant_pipeline.interday.inference import benjamini_hochberg
from quant_pipeline.interday.ranking import build_rank_bin_cache, deterministic_bins
from quant_pipeline.interday.scan import calculate_daily_pair_series
from quant_pipeline.interday.targets import build_targets, future_1d, future_2d
from quant_pipeline.interday.telemetry import StageLedger
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
    x=np.arange(12,dtype=np.float32).reshape(4,3); assert np.isnan(future_2d(x,1)[-1]).all(); assert np.isnan(future_1d(np.arange(4,dtype=np.float32),1)[-1]); assert np.array_equal(__import__('quant_pipeline.interday.targets',fromlist=['future_date_ids']).future_date_ids(4,1),np.array([1,2,3,-1],np.int32))

def test_deterministic_bins_use_integer_tie_breaker():
    values=np.ones(10); bins=deterministic_bins(values,np.arange(10,dtype=np.int64),10); assert np.array_equal(bins,np.arange(10,dtype=np.int8))

def test_rank_bin_thresholds_and_shapes():
    values=np.arange(600,dtype=np.float32).reshape(1,6,100); eligible=np.ones((6,100),bool); cache=build_rank_bin_cache(values,eligible,np.arange(100),["x"],minimum_decile_size=80,minimum_quintile_size=50); assert cache.deciles.shape==(1,6,100); assert cache.quintiles.min()>=0

def test_middle_coverage_is_enforced():
    rng=np.random.default_rng(7); values=rng.normal(size=(1,4,100)); eligible=np.ones((4,100),bool); ranks=build_rank_bin_cache(values,eligible,np.arange(100),["x"],minimum_decile_size=80,minimum_quintile_size=50); target=np.ones((4,100)); target[:,1:99]=np.nan; daily=calculate_daily_pair_series(ranks.percentile_ranks[0],ranks.deciles[0],ranks.quintiles[0],target,minimum_ic_symbols=5,minimum_valid_extreme=1,minimum_bin_coverage=.5,minimum_middle_coverage=.75,minimum_quintile_extreme=1); assert np.isnan(daily.top_minus_middle).all()

def test_corporate_action_reference_handles_split_dividend():
    actions=normalize_actions(pd.DataFrame([{"security_id":"X","effective_session":"2025-01-02","action_type":"split","split_ratio":2},{"security_id":"X","effective_session":"2025-01-03","action_type":"cash_dividend","cash_amount":1}]),discovery_end="2025-01-04")
    index=build_corporate_action_index(actions,pd.date_range("2025-01-01",periods=4),np.array(["X"])); total,price,unresolved=interval_total_return(entry_price=np.array([[10.],[10.],[10.],[10.]]),exit_price=np.array([[10.],[10.],[6.],[6.]]),entry_date_ids=np.array([0,0,0,0]),exit_date_ids=np.array([0,1,2,3]),action_index=index); assert total[3,0] > price[3,0] and not unresolved[3,0]


def test_bh_preserves_missing_values():
    out=benjamini_hochberg(pd.Series([.01,np.nan,.05])); assert np.isnan(out.iloc[1]) and out.iloc[0]<=out.iloc[2]

def test_fdr_groups_all_test_types_in_one_family():
    frame=pd.DataFrame({"feature":["f"]*4,"target":["t"]*4,"test_type":["rank_ic","top_minus_bottom_decile","top_decile_minus_middle","middle_minus_bottom_decile"],"fdr_family":["family"]*4,"raw_p":[.01,.02,.03,.04]})
    out=apply_interday_fdr(frame); assert out.global_fdr.notna().all(); assert out.global_fdr.max() <= .04

def test_matrix_manifest_is_atomic_and_digest_checked(tmp_path):
    path=tmp_path/"x.npy"; arr=np.zeros((2,3,4),np.float32)
    write_matrix(path,arr,names=["a","b"],dates=pd.date_range("2025-01-01",periods=3),security_ids=np.arange(4),fingerprint="fp",schema_version="v",axis_order=("feature","date","security"))
    assert validate_matrix(path,fingerprint="fp",schema_version="v",shape=(2,3,4),axis_order=("feature","date","security"))["dtype"]=="float32"
    path.write_bytes(path.read_bytes()+b"x")
    with pytest.raises(InterdayCacheMismatch): validate_matrix(path,fingerprint="fp",schema_version="v")

def test_ledger_rejects_empty_completion(tmp_path):
    ledger=StageLedger(tmp_path)
    with pytest.raises(ValueError): ledger.complete("diagnostics","fp",[])

def test_registry_has_canonical_close_targets_and_no_duplicate_features():
    config=InterdayConfig(); features=build_feature_registry(config); targets=build_target_registry(config); assert len({f.name for f in features})==len(features); assert sum(t.checkpoint=="close5" for t in targets)==14
