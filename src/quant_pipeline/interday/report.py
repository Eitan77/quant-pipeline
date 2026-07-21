from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _markdown(frame: pd.DataFrame, *, index: bool = False) -> str:
    try:
        return frame.to_markdown(index=index)
    except ImportError:
        return frame.to_string(index=index)


def write_report(
    root: Path,
    *,
    scan: pd.DataFrame | None = None,
    candidates: pd.DataFrame | None = None,
    rejected_candidates: pd.DataFrame | None = None,
    horizon_profiles: pd.DataFrame | None = None,
    checkpoint_profiles: pd.DataFrame | None = None,
    feature_coverage: pd.DataFrame | None = None,
    target_coverage: pd.DataFrame | None = None,
    persistence_turnover: pd.DataFrame | None = None,
    diagnostics=None,
    metadata: dict,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    scan = pd.DataFrame() if scan is None else scan
    candidates = pd.DataFrame() if candidates is None else candidates
    rejected_candidates = pd.DataFrame() if rejected_candidates is None else rejected_candidates
    horizon_profiles = pd.DataFrame() if horizon_profiles is None else horizon_profiles
    checkpoint_profiles = pd.DataFrame() if checkpoint_profiles is None else checkpoint_profiles
    feature_coverage = pd.DataFrame() if feature_coverage is None else feature_coverage
    target_coverage = pd.DataFrame() if target_coverage is None else target_coverage
    persistence_turnover = pd.DataFrame() if persistence_turnover is None else persistence_turnover

    scan.to_parquet(root / "scan_results.parquet", index=False)
    candidates.to_parquet(root / "candidates.parquet", index=False)
    candidates.to_csv(root / "candidate_summary.csv", index=False)

    report_json = {
        "metadata": metadata,
        "hypotheses": {
            "planned": int(metadata.get("planned_hypotheses", len(scan))),
            "executed": int(len(scan)),
            "valid_pvalues": int(scan["raw_p"].notna().sum()) if "raw_p" in scan else 0,
        },
        "candidates": int(len(candidates)),
        "rejected_candidates": int(len(rejected_candidates)),
        "diagnostics_complete": bool(metadata.get("diagnostics_complete", False)),
    }
    json_path = root / "report.json"
    write_json_atomic(json_path, report_json)

    multiple_testing = (
        _markdown(scan.groupby(["fdr_family", "test_type"], dropna=False).size().rename("rows").to_frame())
        if {"fdr_family", "test_type"}.issubset(scan.columns)
        else "No scan rows."
    )
    sections = [
        "# Interday 2A Discovery Report", "",
        "## Run identity", "",
        f"- Experiment: `{metadata.get('experiment_id', '')}`",
        f"- Fingerprint: `{metadata.get('fingerprint', '')}`",
        f"- Git commit: `{metadata.get('git_commit', '')}`",
        f"- Analysis end: `{metadata.get('discovery_end', '')}`",
        f"- Sealed holdout begins: `{metadata.get('sealed_holdout_start', '')}`", "",
        "## Coverage", "",
        f"- Features built: {metadata.get('features_built', len(feature_coverage))}",
        f"- Targets built: {metadata.get('targets_built', len(target_coverage))}",
        f"- Hypotheses executed: {len(scan)}", "",
        "## Multiple testing", "", multiple_testing, "",
        "## Candidate summary", "",
        _markdown(candidates.head(50)) if not candidates.empty else "No candidates passed final gates.", "",
        "## Rejected candidates", "",
        _markdown(rejected_candidates.head(100)) if not rejected_candidates.empty else "No rejected candidate table was produced.", "",
        "## Horizon profiles", "",
        _markdown(horizon_profiles.head(100)) if not horizon_profiles.empty else "No horizon profiles.", "",
        "## Checkpoint profiles", "",
        _markdown(checkpoint_profiles.head(100)) if not checkpoint_profiles.empty else "No checkpoint profiles.", "",
        "## Turnover and persistence", "",
        _markdown(persistence_turnover.head(100)) if not persistence_turnover.empty else "No persistence or turnover table.", "",
        "## Diagnostics", "",
        _markdown(diagnostics.summary.head(100)) if diagnostics is not None and not diagnostics.summary.empty else "Exact diagnostics not complete.", "",
        "## Readiness", "",
        f"- Status: `{metadata.get('readiness', 'UNKNOWN')}`",
        f"- Full run permitted: `{metadata.get('full_run_permitted', False)}`",
    ]
    markdown_path = root / "report.md"
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(sections), encoding="utf-8")
    temporary.replace(markdown_path)
    return json_path, markdown_path
