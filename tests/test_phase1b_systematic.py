from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_pipeline.config import ScanConfig
from quant_pipeline.registry import FeatureSpec
from quant_pipeline.systematic_dual_registry import (
    classify_systematic_candidate, compile_systematic_dual_plan,
    finalize_systematic_results, generate_systematic_pairs,
    select_systematic_parents,
)


def _config(**parent_changes):
    base=ScanConfig()
    parent=replace(base.systematic_phase1b.parent_selection,minimum_valid_observations=4,minimum_sessions=2,minimum_symbols=2,minimum_decision_timestamps=2,**parent_changes)
    pair=replace(base.systematic_phase1b.pair_generation,minimum_joint_observations=4,minimum_joint_sessions=2,minimum_joint_symbols=2,minimum_joint_decision_timestamps=2,max_parent_pairs=20,maximum_pairs_per_family_pair=20)
    return replace(base,systematic_phase1b=replace(base.systematic_phase1b,parent_selection=parent,pair_generation=pair))


def _cache(tmp_path,name,values):
    frame=pd.DataFrame({"symbol":["A","B"]*4,"session_date":np.repeat(pd.date_range("2025-01-01",periods=2),4),"decision_ts":pd.date_range("2025-01-01",periods=8,freq="5min",tz="UTC"),"analysis_eligible":True,name:values})
    path=tmp_path/f"{name}.parquet";frame.to_parquet(path,index=False);return path


def _results(names):
    return pd.DataFrame([{"feature":name,"target":"fwd_return_5m","target_tier":"primary","target_family":"fwd_return_5m","primary_global_fdr":.20,"top_bottom_spread":.00005,"anomaly_score":1-i/10,"valid_observations":4,"valid_sessions":2,"valid_symbols":2,"valid_decision_timestamps":2} for i,name in enumerate(names)])


def test_parent_selection_boundaries_missing_cache_and_caps(tmp_path):
    specs=[FeatureSpec("a","","f",redundancy_group="r"),FeatureSpec("b","","f",redundancy_group="r"),FeatureSpec("c","","f",redundancy_group="r"),FeatureSpec("missing","","g")]
    paths={name:_cache(tmp_path,name,np.arange(8)+(i%2)) for i,name in enumerate(("a","b","c"))}
    parents,ledger=select_systematic_parents(_results([s.name for s in specs]),specs,paths,_config(max_per_redundancy_group=2),pd.DataFrame({"feature":["a","b","c","missing"],"status":"built"}))
    assert len(parents)==2 and all(p.best_primary_global_fdr==.20 and p.best_absolute_effect==.00005 for p in parents)
    indexed={row["feature"]:row for row in ledger}
    assert indexed["missing"]["rejection_reason"]=="missing_feature_cache"
    assert indexed["c"]["rejection_reason"]=="redundancy_group_cap"


def test_parent_threshold_rejection_and_diversity_backfill(tmp_path):
    specs=[FeatureSpec("a","","one"),FeatureSpec("b","","two")];paths={s.name:_cache(tmp_path,s.name,np.arange(8)) for s in specs}
    rows=_results(["a","b"]);rows.loc[rows.feature.eq("a"),"primary_global_fdr"]=.20001;rows.loc[rows.feature.eq("b"),"top_bottom_spread"]=.000049
    parents,_=select_systematic_parents(rows,specs,paths,_config(include_top_per_family_when_threshold_not_met=1,include_top_per_target_family_when_threshold_not_met=0),None)
    assert {p.feature for p in parents}=={"a","b"} and all(p.diversity_backfill for p in parents)


def test_pair_generation_is_deterministic_and_filters_redundancy_and_correlation(tmp_path):
    cfg=_config();specs=[FeatureSpec("a","","x",redundancy_group="same"),FeatureSpec("b","","x",redundancy_group="same"),FeatureSpec("c","","y")]
    paths={"a":_cache(tmp_path,"a",np.arange(8)),"b":_cache(tmp_path,"b",np.arange(8)),"c":_cache(tmp_path,"c",[3,0,7,1,6,2,5,4])}
    parents,_=select_systematic_parents(_results(["a","b","c"]),specs,paths,cfg,None)
    pairs,ledger=generate_systematic_pairs(parents,paths,cfg);pairs2,_=generate_systematic_pairs(tuple(reversed(parents)),paths,cfg)
    assert pairs==pairs2 and len({(p.feature_a,p.feature_b) for p in pairs})==len(pairs)
    reasons={(r["feature_a"],r["feature_b"]):r["rejection_reason"] for r in ledger}
    assert reasons[("a","b")]=="same_redundancy_group"


def test_operator_plan_provenance_thresholds_and_ambiguous_orientation(tmp_path):
    specs=[FeatureSpec("a","","x"),FeatureSpec("b","","y")];paths={"a":_cache(tmp_path,"a",np.arange(8)),"b":_cache(tmp_path,"b",[3,0,7,1,6,2,5,4])}
    plan=compile_systematic_dual_plan(_results(["a","b"]),specs,paths,_config(),None)
    assert {item.spec.operator for item in plan.compiled_features}=={"aligned_rank_mean","intersection","gated_anchor"}
    assert all(item.spec.evidence_class=="systematic_generated_interaction" and item.spec.discovery_phase=="1B_systematic" for item in plan.compiled_features)
    intersection=next(item for item in plan.compiled_features if item.spec.operator=="intersection")
    assert sorted([intersection.definition.condition_a.threshold,intersection.definition.condition_b.threshold])==[.75,.90]
    conflicting=pd.concat([_results(["a","b"]),_results(["a"]).assign(target="fwd_return_15m",top_bottom_spread=-.00005)],ignore_index=True)
    plan2=compile_systematic_dual_plan(conflicting,specs,paths,_config(),None)
    assert all(item.spec.operator not in {"aligned_rank_mean","intersection"} for item in plan2.compiled_features)


def test_systematic_fdr_excludes_invalid_rows_and_is_reproducible():
    specs=[FeatureSpec("x","","dual",operator="aligned_rank_mean",parent_features=("a","b"),redundancy_group="pair")]
    raw=pd.DataFrame({"feature":["x","x"],"target":["fwd_return_5m","fwd_return_15m"],"target_family":["a","b"],"raw_p":[.01,np.nan]})
    first=finalize_systematic_results(raw,specs);second=finalize_systematic_results(raw.sample(frac=1,random_state=2),specs).sort_values("target").reset_index(drop=True)
    assert first.systematic_test_count.eq(1).all() and first.loc[0,"systematic_global_fdr"]==pytest.approx(.01)
    assert second.systematic_global_fdr.dropna().iloc[0]==pytest.approx(.01)
    assert first.target_tier.eq("exploratory").all()


def _candidate(**changes):
    row={"systematic_global_fdr":.05,"two_way_cluster_p":.05,"top_bottom_spread":.000125,"best_parent_absolute_effect":.0001,"historical_subperiod_positive_fold_pct":.60,"historical_subperiod_worst_signed_effect":.00001,"recent_to_full_effect_ratio":1.0,"eligible_symbols_expected_direction_pct":.55,"symbol_effect_hhi":.02,"top5_symbol_effect_pct":.25,"effect_remove_top5_symbols":.0000625}
    row.update(changes);return row


@pytest.mark.parametrize("changes,expected",[
    ({},"exploratory_generated_interaction_candidate"),
    ({"top_bottom_spread":.000099},"systematic_interaction_not_robust"),
    ({"top_bottom_spread":.000124},"significant_but_not_incremental_to_parent"),
    ({"historical_subperiod_positive_fold_pct":.59},"systematic_interaction_not_robust"),
    ({"eligible_symbols_expected_direction_pct":.54},"systematic_interaction_not_robust"),
    ({"symbol_effect_hhi":.02001},"systematic_interaction_not_robust"),
    ({"top5_symbol_effect_pct":.2501},"systematic_interaction_not_robust"),
    ({"effect_remove_top5_symbols":.00006},"systematic_interaction_not_robust"),
])
def test_systematic_promotion_boundaries(changes,expected):
    assert classify_systematic_candidate(_candidate(**changes),ScanConfig())==expected
