from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import json
import hashlib
import importlib.metadata
import duckdb
import numpy as np
import pandas as pd

from .calendar import TradingCalendar
from .config import InterdayConfig

REQUIRED_BAR_COLUMNS = ("symbol","bar_start_ts","bar_end_ts","available_at_ts","open","high","low","close","vwap","volume","session_date","ingested_at")

@dataclass(frozen=True)
class SourceProvenance:
    catalog_path: str
    source_table: str
    source_start: str
    analysis_start: str
    discovery_end: str
    raw_rows_considered: int
    deduplicated_rows: int
    daily_rows: int
    stable_security_id_source: str
    source_schema_hash: str
    compact_file_sha256: str

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$.]*$")
def validate_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value): raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value

def resolved_source_sql(*, source_table: str, security_master_table: str | None, native_security_id: bool) -> str:
    source_table=validate_identifier(source_table)
    if native_security_id:
        identity_cte=f"""SELECT CAST(b.security_id AS VARCHAR) AS security_id, b.* FROM {source_table} AS b"""
    else:
        if not security_master_table: raise ValueError("Source lacks security_id and no security_master_table was configured")
        security_master_table=validate_identifier(security_master_table)
        identity_cte=f"""SELECT CAST(m.security_id AS VARCHAR) AS security_id, b.* FROM {source_table} AS b JOIN {security_master_table} AS m ON m.symbol=b.symbol AND CAST(b.session_date AS DATE)>=CAST(m.valid_from AS DATE) AND (m.valid_to IS NULL OR CAST(b.session_date AS DATE)<CAST(m.valid_to AS DATE)) AND (m.known_at_ts IS NULL OR CAST(m.known_at_ts AS TIMESTAMPTZ)<=CAST(b.available_at_ts AS TIMESTAMPTZ))"""
    return f"""WITH identity_resolved AS ({identity_cte}), filtered AS (SELECT security_id,symbol,CAST(session_date AS DATE) AS session_date,CAST(bar_start_ts AS TIMESTAMPTZ) AS bar_start_ts,CAST(bar_end_ts AS TIMESTAMPTZ) AS bar_end_ts,CAST(available_at_ts AS TIMESTAMPTZ) AS available_at_ts,CAST(ingested_at AS TIMESTAMPTZ) AS ingested_at,CAST(open AS DOUBLE) AS open,CAST(high AS DOUBLE) AS high,CAST(low AS DOUBLE) AS low,CAST(close AS DOUBLE) AS close,CAST(vwap AS DOUBLE) AS vwap,CAST(volume AS DOUBLE) AS volume FROM identity_resolved WHERE feed=? AND adjustment=? AND bar_complete AND CAST(session_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)), deduplicated AS (SELECT * FROM filtered QUALIFY ROW_NUMBER() OVER (PARTITION BY security_id,bar_start_ts ORDER BY ingested_at DESC)=1) SELECT * FROM deduplicated"""

def _security_ids(connection: duckdb.DuckDBPyConnection, symbols: list[str]) -> dict[str, str]:
    try:
        frame = connection.execute("select symbol, id from assets where symbol in (select unnest(?))", [symbols]).fetchdf()
        if not frame.empty and frame.symbol.is_unique and frame.id.notna().all():
            return dict(zip(frame.symbol, frame.id.astype(str)))
    except Exception:
        pass
    return {symbol: f"symbol:{symbol}" for symbol in symbols}

def load_projected_bars(config: InterdayConfig) -> tuple[pd.DataFrame, SourceProvenance]:
    config.validate()
    start = config.source_start
    con = duckdb.connect(config.catalog_path, read_only=True)
    try:
        tables = {row[0] for row in con.execute("show tables").fetchall()}
        validate_identifier(config.source_table); validate_identifier(config.membership_table)
        if config.source_table not in tables:
            raise ValueError(f"Missing source table: {config.source_table}")
        if config.require_membership and config.membership_table not in tables:
            raise ValueError(f"Required membership table is missing: {config.membership_table}")
        source_columns={r[0] for r in con.execute(f"describe {config.source_table}").fetchall()}
        missing = set(REQUIRED_BAR_COLUMNS) - source_columns
        if missing:
            raise ValueError(f"Source schema missing columns: {sorted(missing)}")
        if config.security_id_column not in source_columns and not config.security_master_table and config.require_stable_security_id:
            raise ValueError("Stable security_id is unavailable; configure a historical security master")
        native_security_id = config.security_id_column in source_columns
        query = resolved_source_sql(source_table=config.source_table, security_master_table=config.security_master_table, native_security_id=native_security_id)
        frame = con.execute(query, [config.feed, config.source_adjustment, start, config.discovery_end]).fetchdf()
        if frame.empty: raise ValueError("No five-minute bars matched Interday 2A source filters")
        symbols = sorted(frame.symbol.dropna().unique().tolist() + [config.benchmark_symbol])
        ids = _security_ids(con, sorted(set(symbols)))
    finally:
        con.close()
    for col in ("bar_start_ts","bar_end_ts","available_at_ts","ingested_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    frame["session_date"] = pd.to_datetime(frame.session_date).dt.normalize()
    frame = frame.sort_values(["security_id","bar_start_ts","ingested_at"], kind="stable").reset_index(drop=True)
    if (frame.available_at_ts < frame.bar_end_ts).any(): raise ValueError("Source contains bars unavailable before completion")
    if frame.duplicated(["security_id","bar_start_ts"]).any(): raise ValueError("Duplicate source bars")
    if (frame[["open","high","low","close"]].to_numpy(float) <= 0).any(): raise ValueError("Source contains nonpositive prices")
    if (frame.volume < 0).any(): raise ValueError("Source contains negative volume")
    if frame.session_date.max() >= pd.Timestamp(config.sealed_holdout_start): raise ValueError("Source reaches sealed holdout")
    cal = TradingCalendar(config.exchange_calendar)
    sessions = pd.DatetimeIndex(cal.sessions(start, config.discovery_end)).tz_localize(None).normalize()
    frame = frame.loc[frame.session_date.isin(sessions)].copy()
    if "security_id" not in frame or frame.security_id.isna().all():
        if config.require_stable_security_id: raise ValueError("Stable security_id could not be resolved")
        frame["security_id"] = frame.symbol.map(lambda s: f"symbol:{s}")
    else: frame["security_id"] = frame.security_id.astype(str)
    frame["stable_security_id"] = frame.security_id
    frame["source_has_native_security_id"] = config.security_id_column in frame.columns
    schema_hash=hashlib.sha256(json.dumps(sorted(source_columns)).encode()).hexdigest()
    provenance = SourceProvenance(config.catalog_path, config.source_table, config.source_start, config.analysis_start, config.discovery_end, len(frame), len(frame), 0, "native.security_id" if config.security_id_column in source_columns else "security_master", schema_hash, "")
    return frame, provenance

def load_compact_daily_inputs(config: InterdayConfig):
    """Aggregate five-minute bars in DuckDB and return only daily/checkpoint data."""
    from .panel import CHECKPOINTS
    start = config.source_start
    calendar = TradingCalendar(config.exchange_calendar)
    schedule = calendar.clocks(start, config.discovery_end)
    schedule = schedule.rename(columns={"session_date":"session_date"})[["session_date","open_ts","close_ts","shortened_session"]]
    checkpoint_rows=[]
    for row in schedule.itertuples(index=False):
        values={"session_date":row.session_date,"open5_ts":row.open_ts+pd.Timedelta(minutes=5),"open15_ts":row.open_ts+pd.Timedelta(minutes=15),"close5_ts":row.close_ts,"close15_ts":row.close_ts-pd.Timedelta(minutes=10)}
        for cp in CHECKPOINTS:
            try: values[f"cp_{cp.replace(':','_')}_ts"] = calendar.checkpoint_bar_end(__import__('quant_pipeline.interday.calendar',fromlist=['SessionClock']).SessionClock(row.session_date,row.open_ts,row.close_ts,bool(row.shortened_session)),cp)
            except ValueError: values[f"cp_{cp.replace(':','_')}_ts"] = pd.NaT
        checkpoint_rows.append(values)
    checkpoint_schedule=pd.DataFrame(checkpoint_rows)
    con=duckdb.connect(config.catalog_path,read_only=True)
    try:
        tables={r[0] for r in con.execute("show tables").fetchall()}
        validate_identifier(config.source_table); validate_identifier(config.membership_table)
        if config.source_table not in tables: raise ValueError(f"Missing source table: {config.source_table}")
        source_columns={r[0] for r in con.execute(f"describe {config.source_table}").fetchall()}
        if config.security_id_column not in source_columns and config.require_stable_security_id:
            if not config.security_master_table:
                raise ValueError("Stable security_id is unavailable for compact production aggregation")
        identity_join = ""
        identity_select = "" if config.security_id_column in source_columns else "m.security_id AS security_id"
        if config.security_id_column not in source_columns:
            validate_identifier(config.security_master_table)
            master_columns={r[0] for r in con.execute(f"describe {config.security_master_table}").fetchall()}
            if not {"symbol", "security_id", "valid_from"}.issubset(master_columns):
                raise ValueError("Historical security master must expose symbol, security_id, and valid_from")
            valid_to="m.valid_to" if "valid_to" in master_columns else "NULL"
            identity_join=f"JOIN {config.security_master_table} m ON m.symbol=b.symbol AND CAST(b.session_date AS DATE)>=CAST(m.valid_from AS DATE) AND ({valid_to} IS NULL OR CAST(b.session_date AS DATE)<CAST({valid_to} AS DATE))"
        identity_projection = f", {identity_select}" if identity_select else ""
        identity_key = "b.security_id" if config.security_id_column in source_columns else "m.security_id"
        if config.require_membership and config.membership_table not in tables: raise ValueError(f"Required membership table is missing: {config.membership_table}")
        con.register("interday_schedule",schedule)
        con.register("interday_checkpoint_schedule",checkpoint_schedule)
        cp_expr=[]
        for cp in CHECKPOINTS:
            col=cp.replace(':','_')
            if cp == "open15":
                cp_expr.append("sum(vwap*volume) FILTER (WHERE rn<=3)/NULLIF(sum(volume) FILTER (WHERE rn<=3),0) AS \"open15\"")
            elif cp == "close15":
                cp_expr.append("sum(vwap*volume) FILTER (WHERE rn>CAST(date_diff('minute',scheduled_open,scheduled_close)/5 AS BIGINT)-3)/NULLIF(sum(volume) FILTER (WHERE rn>CAST(date_diff('minute',scheduled_open,scheduled_close)/5 AS BIGINT)-3),0) AS \"close15\"")
            else:
                cp_expr.append(f"max(vwap) FILTER (WHERE bar_end_ts = cp_{col}_ts) AS \"{cp}\"")
        group_columns = ["security_id","symbol","session_day","scheduled_open","scheduled_close","shortened_session"] + [f"cp_{cp.replace(':','_')}_ts" for cp in CHECKPOINTS]
        query=f"""
            WITH filtered AS (
                SELECT b.*{identity_projection}, CAST(b.session_date AS DATE) AS session_day,
                       s.open_ts AS scheduled_open, s.close_ts AS scheduled_close,
                       s.shortened_session, cs.* EXCLUDE (session_date),
                       row_number() OVER (PARTITION BY {identity_key}, b.symbol, b.bar_start_ts ORDER BY b.ingested_at DESC) AS source_rn
                FROM {config.source_table} b {identity_join}
                JOIN interday_schedule s ON CAST(b.session_date AS DATE)=s.session_date
                JOIN interday_checkpoint_schedule cs ON CAST(b.session_date AS DATE)=cs.session_date
                WHERE b.feed=? AND b.adjustment=? AND b.bar_complete
                  AND CAST(b.session_date AS DATE)>=CAST(? AS DATE)
                  AND CAST(b.session_date AS DATE)<CAST(? AS DATE)
                  AND b.bar_end_ts<=s.close_ts AND b.bar_start_ts>=s.open_ts
            ), numbered AS (
                SELECT *, row_number() OVER (PARTITION BY security_id, symbol, session_day ORDER BY bar_start_ts) AS rn,
                       count(*) OVER (PARTITION BY security_id, symbol, session_day) AS bar_count
                FROM filtered WHERE source_rn=1
            )
            SELECT security_id, symbol, session_day AS session_date,
                   scheduled_open, scheduled_close, shortened_session,
                   count(*) = CAST(date_diff('minute',scheduled_open,scheduled_close)/5 AS BIGINT) AS session_complete,
                   max(available_at_ts) AS available_at_ts,
                   arg_min(open,bar_start_ts) AS open, max(high) AS high, min(low) AS low,
                   arg_max(close,bar_start_ts) AS close,
                   sum(vwap*volume)/NULLIF(sum(volume),0) AS session_vwap,
                   sum(volume) AS volume, sum(vwap*volume) AS dollar_volume,
                   arg_min(vwap,bar_start_ts) AS first_5m_vwap,
                   sum(vwap*volume) FILTER (WHERE rn<=3)/NULLIF(sum(volume) FILTER (WHERE rn<=3),0) AS first_15m_vwap,
                   arg_max(vwap,bar_start_ts) AS last_5m_vwap,
                   sum(vwap*volume) FILTER (WHERE rn>CAST(date_diff('minute',scheduled_open,scheduled_close)/5 AS BIGINT)-3)/NULLIF(sum(volume) FILTER (WHERE rn>CAST(date_diff('minute',scheduled_open,scheduled_close)/5 AS BIGINT)-3),0) AS last_15m_vwap,
                   sum(volume) FILTER (WHERE bar_end_ts<=scheduled_open+INTERVAL 30 MINUTE) AS open_30m_volume,
                   sum(volume) FILTER (WHERE bar_start_ts>=scheduled_close-INTERVAL 30 MINUTE) AS close_30m_volume,
                   sum(volume) FILTER (WHERE bar_end_ts<=scheduled_open+INTERVAL 60 MINUTE) AS first_60m_volume,
                   sum(volume) FILTER (WHERE bar_start_ts>=scheduled_close-INTERVAL 60 MINUTE) AS last_60m_volume,
                   max(volume) AS largest_5m_volume,
                   arg_max(vwap,bar_end_ts) FILTER (WHERE bar_end_ts<=scheduled_open+INTERVAL 150 MINUTE) AS midday,
                   arg_max(close,bar_end_ts) FILTER (WHERE bar_end_ts<=scheduled_open+INTERVAL 60 MINUTE)/NULLIF(arg_min(open,bar_start_ts),0)-1 AS first_60m_return,
                   arg_max(close,bar_end_ts) FILTER (WHERE bar_start_ts>=scheduled_close-INTERVAL 60 MINUTE)/NULLIF(arg_min(open,bar_start_ts) FILTER (WHERE bar_start_ts>=scheduled_close-INTERVAL 60 MINUTE),0)-1 AS last_60m_return,
                   {', '.join(cp_expr)}
            FROM numbered GROUP BY {', '.join(group_columns)} ORDER BY session_date,symbol
        """
        daily=con.execute(query,[config.feed,config.source_adjustment,start,config.sealed_holdout_start]).fetchdf()
    finally: con.close()
    for c in ("session_date",): daily[c]=pd.to_datetime(daily[c]).dt.normalize()
    if "security_id" not in daily or daily.security_id.isna().all():
        if config.require_stable_security_id: raise ValueError("Stable security_id could not be resolved")
        daily["security_id"] = "symbol:" + daily.symbol.astype(str)
    checkpoint_cols=["security_id","symbol","session_date"]+[c for c in CHECKPOINTS if c in daily]
    checkpoints=daily[checkpoint_cols].copy()
    coverage=daily[["security_id","symbol","session_date","session_complete"]].copy(); coverage["expected_bars"]=(daily.scheduled_close-daily.scheduled_open).dt.total_seconds().div(300).astype(int); coverage["actual_bars"]=np.where(daily.session_complete,coverage.expected_bars,np.nan); coverage["missing_checkpoints"]=daily[list(CHECKPOINTS)].isna().sum(axis=1)
    return daily,checkpoints,coverage,SourceProvenance(config.catalog_path,config.source_table,config.source_start,config.analysis_start,config.discovery_end,0,0,len(daily),"source.security_id", "", "")

def schema_check(config: InterdayConfig, output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config.catalog_path, read_only=True)
    try:
        tables = [r[0] for r in con.execute("show tables").fetchall()]
        schema = {config.source_table: con.execute(f"describe {config.source_table}").fetchdf().to_dict("records") if config.source_table in tables else []}
        schema[config.membership_table] = con.execute(f"describe {config.membership_table}").fetchdf().to_dict("records") if config.membership_table in tables else []
    finally:
        con.close()
    cal = TradingCalendar(config.exchange_calendar)
    clocks = cal.clocks(config.analysis_start, min(config.discovery_end, "2026-04-30"))
    deps = {name: importlib.metadata.version(name) for name in ("duckdb","exchange-calendars","numpy","pandas","pyarrow","scipy","torch") if _installed(name)}
    source_cols = {x["column_name"] for x in schema.get(config.source_table, [])}
    stable_available = config.security_id_column in source_cols or bool(config.security_master_table)
    action_report = {"path": config.corporate_actions_path, "exists": bool(config.corporate_actions_path and Path(config.corporate_actions_path).exists())}
    if action_report["exists"]:
        try:
            import pyarrow.parquet as pq
            action_schema = pq.read_schema(config.corporate_actions_path)
            action_report["columns"] = action_schema.names
            action_types = set(pd.read_parquet(config.corporate_actions_path, columns=["action_type"]).action_type.astype(str).str.lower()) if "action_type" in action_schema.names else set()
            action_report["has_cash_dividends"] = bool(action_types & {"cash_dividend", "cash_dividends", "dividend"})
            action_report["has_delisting_outcomes"] = bool(action_types & {"delisting", "delist", "cash_merger"}) or {"terminal_cash_per_share", "delisting_return"}.issubset(action_schema.names)
        except Exception as exc:
            action_report["error"] = str(exc)
    action_report["ready"] = bool(action_report["exists"] and (not config.require_cash_dividends or action_report.get("has_cash_dividends", False)) and (not config.require_delisting_outcomes or action_report.get("has_delisting_outcomes", False)))
    membership_cols={x["column_name"] for x in schema.get(config.membership_table, [])}
    report = {"tables": tables, "schema": schema, "required_bar_columns": list(REQUIRED_BAR_COLUMNS),
              "security_id_source_column": config.security_id_column if config.security_id_column in source_cols else None,
              "stable_security_id_available": stable_available,
              "membership_security_id_available": config.membership_security_id_column in membership_cols,
              "security_id_policy": "historical source/security-master identity is mandatory; symbol fallback is disabled in production",
              "corporate_actions": action_report, "corporate_actions_ready": action_report["ready"], "cuda_available": _cuda_available(), "bar_timestamp_contract": "bar_end_ts is the completed five-minute VWAP endpoint"}
    (output_root / "source_schema.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (output_root / "dependency_versions.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")
    (output_root / "calendar_contract.json").write_text(json.dumps({"calendar": config.exchange_calendar, "sessions": len(clocks), "first": str(clocks.session_date.min()), "last": str(clocks.session_date.max()), "timezone": "America/New_York"}, indent=2), encoding="utf-8")
    return report

def _installed(name: str) -> bool:
    try: importlib.metadata.version(name); return True
    except importlib.metadata.PackageNotFoundError: return False

def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
