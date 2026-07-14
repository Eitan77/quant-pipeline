from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.config import ScanConfig
from quant_pipeline.diagnostics import _causal_dispersion_bucket,exact_time_diagnostics,phase2_recommendation,regime_diagnostics,scope_diagnostics


def panel(days=130):
    rng=np.random.default_rng(31); rows=[]
    for day_index,day in enumerate(pd.bdate_range("2025-01-02",periods=days)):
        for symbol_index,symbol in enumerate(["AAA","BBB","CCC","DDD"]):
            start=(pd.Timestamp(day.date()).tz_localize("America/New_York")+pd.Timedelta(hours=9,minutes=30)).tz_convert("UTC"); decision=start+pd.Timedelta(minutes=5 if day_index%2 else 30); x=rng.normal()
            rows.append({"symbol":symbol,"session_date":day.normalize(),"bar_start_ts":start,"decision_ts":decision,"analysis_eligible":True,"x":x,"y":.002*x+rng.normal(0,.002),"high_market_vol":float(day_index%2),"universe_breadth_positive":.65 if day_index%2 else .35,"universe_return_dispersion":.01+day_index/10000,"benchmark_return_since_open":.003 if day_index%3==0 else 0,"benchmark_distance_session_vwap":.001 if day_index%3==0 else 0,"benchmark_overnight_gap":.003 if day_index%4==0 else -.003})
    return pd.DataFrame(rows)


def diagnostic_config():
    return ScanConfig(start="2025-01-01",discovery_end="2026-04-30",regime_min_observations=20,regime_min_sessions=5,regime_min_symbols=3,exact_time_min_observations=20,exact_time_min_sessions=5,exact_time_min_symbols=3,scope_min_observations=20,scope_min_sessions=5,scope_min_symbols=2)


def test_dispersion_regime_uses_prior_history_only():
    frame=panel(); before=_causal_dispersion_bucket(frame)
    cutoff=frame.decision_ts.sort_values().iloc[420]; changed=frame.copy(); changed.loc[changed.decision_ts.gt(cutoff),"universe_return_dispersion"]*=1000; after=_causal_dispersion_bucket(changed)
    mask=frame.decision_ts.le(cutoff)
    pd.testing.assert_series_equal(before.loc[mask],after.loc[mask])


def test_regime_diagnostics_use_existing_causal_context():
    summary,table=regime_diagnostics(panel(),"x","y",1,diagnostic_config())
    assert set(table.regime_type)=={"market_volatility","market_breadth","cross_sectional_dispersion","trend_state","gap_direction"}
    assert summary["volatility_regime_status"]=="available"
    assert set(table.minimum_sample_status)<={"sufficient","insufficient_data"}


def test_missing_point_in_time_scope_metadata_is_explicit():
    summary,tables=scope_diagnostics(panel(),"x","y",1,"broad_across_symbols",diagnostic_config())
    assert summary["sector_scope_status"]=="unavailable_missing_point_in_time_sector_data"
    assert summary["industry_scope_status"]=="unavailable_missing_point_in_time_industry_data"
    assert tables["sector"].iloc[0].sector_scope_status==summary["sector_scope_status"]


def test_exact_time_diagnostics_use_decision_timestamp_not_bar_start():
    frame=panel(); summary,table=exact_time_diagnostics(frame,"x","y",1,diagnostic_config())
    assert set(table.decision_time)=={"09:35","10:00"}
    assert frame.bar_start_ts.dt.tz_convert("America/New_York").dt.strftime("%H:%M").nunique()==1
    assert summary["time_concentration_label"] in {"insufficient_time_evidence","persistent_through_session","time_unstable","opening_only"}


@pytest.mark.parametrize(("scope","breadth","expected"),[("sector_specific","moderately_concentrated","advance_as_sector_specific_candidate"),("industry_specific","moderately_concentrated","advance_as_industry_specific_candidate"),("symbol_specific","single_symbol_dominated","advance_as_symbol_specific_candidate")])
def test_phase2_recommendations_are_scope_aware(scope,breadth,expected):
    row=pd.Series({"status":"robust_phase1_anomaly_candidate","scope_classification":scope,"symbol_breadth_classification":breadth,"regime_summary_label":"regime_persistent","time_concentration_label":"persistent_through_session","recent_classification":"persistent"})
    result=phase2_recommendation(row)
    assert result["phase2_recommendation"]==expected
    assert result["phase2_recommendation_reason"] and result["phase2_main_limitation"] and result["phase2_suggested_test"]


def test_descriptive_diagnostics_do_not_mutate_fdr_ranking_or_status():
    frame=pd.DataFrame({"feature":["a","b"],"primary_global_fdr":[.01,.02],"anomaly_score":[.9,.8],"status":["robust_phase1_anomaly_candidate","stable_anomaly_candidate"]}); before=frame.copy(deep=True)
    _=[phase2_recommendation(row) for _,row in frame.iterrows()]
    pd.testing.assert_frame_equal(frame,before)


def test_all_new_diagnostics_reject_holdout_rows():
    frame=panel(2); frame.loc[frame.index[-1],"session_date"]=pd.Timestamp("2026-05-01"); frame.loc[frame.index[-1],"decision_ts"]=pd.Timestamp("2026-05-01 13:35",tz="UTC")
    with pytest.raises(ValueError,match="holdout"):regime_diagnostics(frame,"x","y",1,diagnostic_config())
    with pytest.raises(ValueError,match="holdout"):scope_diagnostics(frame,"x","y",1,"broad_across_symbols",diagnostic_config())
    with pytest.raises(ValueError,match="holdout"):exact_time_diagnostics(frame,"x","y",1,diagnostic_config())
