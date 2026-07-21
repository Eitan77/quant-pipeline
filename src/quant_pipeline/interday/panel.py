from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .calendar import ET, SessionClock, TradingCalendar
from .config import InterdayConfig

CHECKPOINTS = ("open5","open15","09:40","09:45","10:00","10:15","10:30","11:00","12:00","13:00","14:00","15:00","close15","close5")

@dataclass
class DailyPanelBuild:
    daily: pd.DataFrame
    checkpoints: pd.DataFrame
    coverage: pd.DataFrame

def join_effective_dated_metadata(panel: pd.DataFrame, metadata: pd.DataFrame, *, entity_column: str, value_columns: list[str]) -> pd.DataFrame:
    if metadata.empty: return panel.assign(**{c: np.nan for c in value_columns})
    rows = []
    meta = metadata.copy()
    meta["valid_from"] = pd.to_datetime(meta["valid_from"]).dt.normalize()
    meta["valid_to"] = pd.to_datetime(meta.get("valid_to"), errors="coerce").dt.normalize()
    meta["known_at_ts"] = pd.to_datetime(meta["known_at_ts"], utc=True)
    for i, row in panel.iterrows():
        mask = meta[entity_column].eq(row[entity_column]) & meta.valid_from.le(row.session_date) & (meta.valid_to.isna() | meta.valid_to.gt(row.session_date)) & meta.known_at_ts.le(row.available_at_ts)
        matches = meta.loc[mask]
        if len(matches) > 1: raise ValueError(f"Overlapping metadata records for {row[entity_column]} on {row.session_date}")
        rows.append(matches.iloc[0][value_columns].to_dict() if len(matches) else {c: np.nan for c in value_columns})
    return pd.concat([panel.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

def build_daily_panel(bars: pd.DataFrame, calendar: TradingCalendar, config: InterdayConfig) -> DailyPanelBuild:
    schedule = calendar.clocks(config.start, config.discovery_end).set_index("session_date")
    rows, cps, coverage = [], [], []
    for (security_id, symbol, session_date), group in bars.groupby(["security_id","symbol","session_date"], sort=False):
        if session_date not in schedule.index: continue
        c = schedule.loc[session_date]; open_ts, close_ts = pd.Timestamp(c.open_ts), pd.Timestamp(c.close_ts)
        group = group.sort_values("bar_start_ts", kind="stable")
        expected = pd.date_range(open_ts + pd.Timedelta(minutes=5), close_ts, freq="5min")
        actual = pd.DatetimeIndex(group.bar_end_ts)
        complete = actual.equals(expected)
        vol = group.volume.astype(float); dv = vol * group.vwap.astype(float)
        first, last = group.iloc[0], group.iloc[-1]
        first_hour = group.loc[group.bar_end_ts.le(open_ts + pd.Timedelta(hours=1))]
        last_hour = group.loc[group.bar_start_ts.ge(close_ts - pd.Timedelta(hours=1))]
        rows.append({"security_id":security_id,"symbol":symbol,"session_date":session_date,"scheduled_open":open_ts,"scheduled_close":close_ts,"shortened_session":bool(c.shortened_session),"session_complete":complete,"available_at_ts":group.available_at_ts.max(),"open":float(first.open),"high":float(group.high.max()),"low":float(group.low.min()),"close":float(last.close),"session_vwap":float(dv.sum()/vol.sum()) if vol.sum() else np.nan,"volume":float(vol.sum()),"dollar_volume":float(dv.sum()),"first_5m_vwap":float(first.vwap),"first_15m_vwap":float(group.head(3).pipe(lambda x:(x.vwap*x.volume).sum()/x.volume.sum())) if len(group)>=3 else np.nan,"last_5m_vwap":float(last.vwap),"last_15m_vwap":float(group.tail(3).pipe(lambda x:(x.vwap*x.volume).sum()/x.volume.sum())) if len(group)>=3 else np.nan,"first_60m_return":float(first_hour.iloc[-1].close/first.open-1) if len(first_hour) else np.nan,"last_60m_return":float(last.close/last_hour.iloc[0].open-1) if len(last_hour) else np.nan,"open_30m_volume":float(group.loc[group.bar_end_ts.le(open_ts+pd.Timedelta(minutes=30)),"volume"].sum()),"close_30m_volume":float(group.loc[group.bar_start_ts.ge(close_ts-pd.Timedelta(minutes=30)),"volume"].sum())})
        record = {"security_id":security_id,"symbol":symbol,"session_date":session_date}; missing=0
        clock=SessionClock(pd.Timestamp(session_date), open_ts, close_ts, bool(c.shortened_session))
        for checkpoint in CHECKPOINTS:
            if checkpoint == "open15":
                first_three = group.head(3)
                record[checkpoint] = float((first_three.vwap * first_three.volume).sum() / first_three.volume.sum()) if len(first_three) == 3 and first_three.volume.sum() else np.nan
                missing += int(pd.isna(record[checkpoint])); continue
            if checkpoint == "close15":
                last_three = group.tail(3)
                record[checkpoint] = float((last_three.vwap * last_three.volume).sum() / last_three.volume.sum()) if len(last_three) == 3 and last_three.volume.sum() else np.nan
                missing += int(pd.isna(record[checkpoint])); continue
            try: end = calendar.checkpoint_bar_end(clock, checkpoint)
            except ValueError: record[checkpoint]=np.nan; missing+=1; continue
            found = group.loc[group.bar_end_ts.eq(end), "vwap"]
            record[checkpoint] = float(found.iloc[0]) if len(found)==1 else np.nan
            missing += int(pd.isna(record[checkpoint]))
        cps.append(record); coverage.append({"security_id":security_id,"symbol":symbol,"session_date":session_date,"expected_bars":len(expected),"actual_bars":len(actual),"session_complete":complete,"missing_checkpoints":missing})
    daily, checkpoints, cov = (pd.DataFrame(x).sort_values(["session_date","security_id"], kind="stable").reset_index(drop=True) for x in (rows,cps,coverage))
    if daily.duplicated(["security_id","session_date"]).any() or checkpoints.duplicated(["security_id","session_date"]).any(): raise ValueError("Duplicate daily panel keys")
    return DailyPanelBuild(daily, checkpoints, cov)

def attach_membership_and_eligibility(build: DailyPanelBuild, membership: pd.DataFrame, config: InterdayConfig) -> DailyPanelBuild:
    daily = build.daily.copy(); membership = membership.copy()
    if not membership.empty:
        membership["session_date"] = pd.to_datetime(membership["date"]).dt.normalize()
        daily = daily.merge(membership[["symbol","session_date","is_member"]].drop_duplicates(["symbol","session_date"]), on=["symbol","session_date"], how="left")
    else: daily["is_member"] = daily.symbol.eq(config.benchmark_symbol)
    daily["pit_member"] = daily.is_member.fillna(False) | daily.symbol.eq(config.benchmark_symbol)
    daily["sector_id"] = np.nan; daily["industry_id"] = np.nan
    daily["prior_20d_median_dollar_volume"] = daily.groupby("security_id", sort=False).dollar_volume.transform(lambda x: x.rolling(20, min_periods=20).median().shift(1))
    daily["analysis_eligible"] = daily.pit_member & daily.session_complete & daily.open.ge(config.minimum_price) & daily.close.ge(config.minimum_price) & daily.prior_20d_median_dollar_volume.ge(config.minimum_prior_20d_median_dollar_volume)
    daily["scan_eligible"] = daily.analysis_eligible
    daily["corporate_action_valid"] = True
    daily["available_at_ts"] = pd.to_datetime(daily.available_at_ts, utc=True)
    cps = build.checkpoints.merge(daily[["security_id","session_date","analysis_eligible"]], on=["security_id","session_date"], how="left")
    return DailyPanelBuild(daily, cps, build.coverage)
