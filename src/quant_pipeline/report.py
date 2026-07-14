from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_reports(results: pd.DataFrame, quantiles: dict[tuple[str, str], pd.DataFrame], root: Path, frame: pd.DataFrame | None = None) -> None:
    """Write diagnostics only; never strategy-level performance metrics."""
    candidates = results.head(50) if not results.empty else results
    candidates.to_html(root / "ranked_candidates.html", index=False, float_format=lambda x: f"{x:.6g}")
    charts = root / "charts"; charts.mkdir(exist_ok=True)
    selected={(r.feature,r.target) for r in candidates.itertuples()} if not candidates.empty else set()
    for (feature, target), table in quantiles.items():
        if (feature,target) not in selected:
            continue
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        yerr=[table["mean"]-table.ci_low,table.ci_high-table["mean"]]
        ax.errorbar(table.bin, table["mean"], yerr=yerr, fmt="o-")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=f"{feature} -> {target}", xlabel="Feature quantile", ylabel="Mean forward return")
        fig.tight_layout(); fig.savefig(charts / f"{feature}__{target}.png", dpi=140); plt.close(fig)
    master=pd.read_csv(root/"master_results.csv") if (root/"master_results.csv").exists() else pd.DataFrame()
    feature_registry=pd.read_csv(root/"feature_registry.csv") if (root/"feature_registry.csv").exists() else pd.DataFrame()
    target_registry=pd.read_csv(root/"target_registry.csv") if (root/"target_registry.csv").exists() else pd.DataFrame()
    clusters=results.candidate_cluster.nunique() if "candidate_cluster" in results else 0; confirmed=int(results.status.eq("confirmed_anomaly_candidate").sum()) if "status" in results else 0
    text = ["# Phase 1.1 corrected anomaly scan", "", "Statistical relationships only; no result is a trading strategy candidate until later execution work.", "", "## Run accounting", "", f"- Features requested: {len(feature_registry)}",f"- Features successfully built: {len(set(master.feature)) if not master.empty else 0}",f"- Targets requested: {len(target_registry)}",f"- Broad-screen feature-target pairs: {len(master)}",f"- Pairs passing minimum coverage: {int(master.raw_p.notna().sum()) if not master.empty else 0}",f"- Globally FDR-significant primary pairs: {int(master.get('primary_global_fdr',pd.Series(dtype=float)).lt(.05).sum()) if not master.empty else 0}",f"- Exact diagnostic candidates: {len(results)}",f"- Redundancy clusters: {clusters}",f"- Internally confirmed anomalies: {confirmed}"]
    if not results.empty:
        columns=[c for c in ["feature","target","pearson","top_bottom_spread","raw_p","bh_fdr_p","year_consistency","symbol_breadth","status"] if c in candidates]
        markdown=["| "+" | ".join(columns)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
        markdown += ["| "+" | ".join(str(getattr(row,c)) for c in columns)+" |" for row in candidates[columns].itertuples(index=False)]
        text += ["", "## Top relationships", "", "\n".join(markdown)]
    (root / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if frame is not None and not candidates.empty:
        stable=root/"stability"; stable.mkdir(exist_ok=True)
        for row in candidates.head(25).itertuples():
            cols=["session_date","symbol","decision_ts",row.feature,row.target]
            z=frame[cols].dropna().copy(); z["year"]=pd.to_datetime(z.session_date).dt.year
            local=pd.to_datetime(z.decision_ts,utc=True).dt.tz_convert("America/New_York"); z["time_bucket"]=(local.dt.hour*60+local.dt.minute).floordiv(60)
            for key in ["year","symbol","time_bucket"]:
                tab=z.groupby(key).agg(observations=(row.target,"size"),mean_target=(row.target,"mean"),spearman=(row.feature,lambda s:s.corr(z.loc[s.index,row.target],method="spearman"))).reset_index()
                tab.to_csv(stable/f"{row.feature}__{row.target}__by_{key}.csv",index=False)
