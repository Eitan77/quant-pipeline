from __future__ import annotations

import numpy as np
import pandas as pd

from quant_pipeline.config import ScanConfig
from quant_pipeline.dual_features import _materialize
from quant_pipeline.dual_registry import compile_dual_plan, plan_hash
from quant_pipeline.effects import effect_value, feature_scan_kind
from quant_pipeline.registry import FeatureSpec, feature_registry
from quant_pipeline.scanner import binary_scan_batch, scan


def test_phase1b_manifest_is_deterministic_and_uses_existing_parents():
    config=ScanConfig(dual_factor_enabled=True,dual_factor_manifest_path="configs/phase1b_dual_factors.yaml")
    first=compile_dual_plan(config.dual_factor_manifest_path,feature_registry(config.lookbacks),config)
    second=compile_dual_plan(config.dual_factor_manifest_path,feature_registry(config.lookbacks),config)
    assert len(first)==5
    assert [item.spec.name for item in first] == [item.spec.name for item in second]
    assert plan_hash(first)==plan_hash(second)
    assert all(item.spec.discovery_phase=="1B" and item.spec.arity==2 for item in first)


def test_dual_intersection_keeps_missing_as_missing():
    config=ScanConfig(dual_factor_enabled=True,dual_factor_manifest_path="configs/phase1b_dual_factors.yaml")
    item=compile_dual_plan(config.dual_factor_manifest_path,feature_registry(config.lookbacks),config)[0]
    timestamp=pd.Timestamp("2026-04-01 14:35",tz="UTC")
    frame=pd.DataFrame({"symbol":["A","B","C"],"session_date":["2026-04-01"]*3,"decision_ts":[timestamp]*3,"session_range_position":[1.,.2,np.nan],"tod_relative_volume_20":[2.,.2,2.]})
    result=_materialize(frame,item)
    assert result.tolist()[:2]==[1.0,0.0]
    assert np.isnan(result.iloc[2])


def test_binary_effect_is_on_minus_off_not_decile_spread():
    spec=FeatureSpec("signal","", "test", dtype="binary")
    frame=pd.DataFrame({"signal":[0,0,1,1],"target":[-.0005,-.0005,.001,.001]})
    assert feature_scan_kind(spec)=="binary"
    assert np.isclose(effect_value(frame,spec,"target"),.0015)


def test_binary_screen_reports_exact_on_minus_off_effect():
    spec=FeatureSpec("signal","", "test", dtype="binary")
    dates=pd.date_range("2025-01-01",periods=8,freq="D")
    frame=pd.DataFrame({"signal":[0,1]*8,"target":[-.0005,.001]*8,"session_date":np.repeat(dates,2),"symbol":["A","B"]*8,"decision_ts":pd.date_range("2025-01-01",periods=16,freq="5min",tz="UTC"),"analysis_eligible":True})
    config=ScanConfig(min_observations=4,min_sessions=2,min_symbols=2,min_decision_timestamps=2,min_years=1)
    result=binary_scan_batch(frame,[spec],["target"],config).iloc[0]
    assert result["scan_kind"]=="binary"
    assert np.isclose(result["top_bottom_spread"],.0015)
    assert np.isnan(result["monotonicity"])


def test_exact_binary_scan_does_not_construct_deciles():
    spec=FeatureSpec("signal","", "test", dtype="binary")
    frame=pd.DataFrame({"signal":[0,1]*8,"target":[-.0005,.001]*8,"session_date":np.repeat(pd.date_range("2025-01-01",periods=8,freq="D"),2),"symbol":["A","B"]*8,"decision_ts":pd.date_range("2025-01-01",periods=16,freq="5min",tz="UTC"),"analysis_eligible":True})
    config=ScanConfig(min_observations=4,min_sessions=2,min_symbols=2,min_decision_timestamps=2,min_years=1,use_cuda=False)
    result,tables=scan(frame,[spec],["target"],config)
    assert result.iloc[0]["effect_kind"]=="binary_on_minus_off"
    assert list(tables[("signal","target")]["signal"]) == [0,1]


def test_production_yaml_loads_with_phase1b_fields():
    config=ScanConfig.from_yaml("configs/discovery_5m.yaml")
    assert not config.dual_factor_enabled
    assert config.discovery_end=="2026-04-30"
