from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class HoldingOutcome:
    total_return: float; price_return: float; split_multiplier: float; dividends_received: float; delisted: bool

ACTION_COLUMNS = ("security_id","symbol","effective_session","action_sequence","action_type","split_ratio","cash_dividend_per_share","terminal_cash_per_share","reference_price_before_delisting","delisting_return_from_reference","known_at_ts","source")

def normalize_actions(actions: pd.DataFrame | None) -> pd.DataFrame:
    if actions is None or actions.empty: return pd.DataFrame(columns=ACTION_COLUMNS)
    out = actions.copy()
    aliases = {"ex_date":"effective_session","cash_amount":"cash_dividend_per_share","cash_dividend_per_pre_action_share":"cash_dividend_per_share","split_factor":"split_ratio","cash_out_value":"terminal_cash_per_share","delisting_return":"delisting_return_from_reference"}
    out = out.rename(columns={k:v for k,v in aliases.items() if k in out})
    for c in ACTION_COLUMNS:
        if c not in out: out[c] = np.nan
    out["effective_session"] = pd.to_datetime(out.effective_session).dt.normalize()
    out["known_at_ts"] = pd.to_datetime(out.known_at_ts, utc=True, errors="coerce")
    if "action_sequence" not in out: out["action_sequence"] = 0
    out["action_sequence"] = pd.to_numeric(out.action_sequence,errors="coerce").fillna(0).astype(int)
    out["action_type"] = (out.action_type.astype(str).str.lower().str.strip().replace({
        "cash_dividends": "cash_dividend", "dividend": "cash_dividend",
        "splits": "split", "delist": "delisting", "cash_mergers": "cash_merger",
    }))
    if out["effective_session"].isna().any():
        raise ValueError("Corporate-action ledger contains rows without effective_session")
    return out.loc[:, ACTION_COLUMNS]

def calculate_holding_outcome(*, entry_price: float, exit_price: float | None, entry_session: pd.Timestamp, exit_session: pd.Timestamp, actions: pd.DataFrame) -> HoldingOutcome:
    if not np.isfinite(entry_price) or entry_price <= 0: return HoldingOutcome(np.nan,np.nan,np.nan,np.nan,False)
    relevant = actions.loc[actions.effective_session.gt(entry_session) & actions.effective_session.le(exit_session)].sort_values(["effective_session","action_sequence"], kind="stable")
    shares, dividends, delisted, terminal_cash = 1.0, 0.0, False, None
    for action in relevant.itertuples(index=False):
        kind = str(action.action_type)
        if kind == "split":
            ratio = float(action.split_ratio)
            if not np.isfinite(ratio) or ratio <= 0: raise ValueError("Invalid split ratio")
            shares *= ratio
        elif kind == "cash_dividend": dividends += shares * float(action.cash_dividend_per_share)
        elif kind in {"delisting","cash_merger"}:
            delisted = True
            if np.isfinite(action.terminal_cash_per_share): terminal_cash = shares * float(action.terminal_cash_per_share)
            elif np.isfinite(action.delisting_return_from_reference):
                reference=float(action.reference_price_before_delisting) if np.isfinite(action.reference_price_before_delisting) else np.nan
                if not np.isfinite(reference): return HoldingOutcome(np.nan,np.nan,shares,dividends,True)
                terminal_cash = shares * reference * (1 + float(action.delisting_return_from_reference))
            else: return HoldingOutcome(np.nan,np.nan,shares,dividends,True)
            break
    if terminal_cash is None:
        if exit_price is None or not np.isfinite(exit_price) or exit_price <= 0: return HoldingOutcome(np.nan,np.nan,shares,dividends,delisted)
        terminal_cash = shares * float(exit_price)
    total = (terminal_cash + dividends) / entry_price - 1
    price = shares * float(exit_price) / entry_price - 1 if exit_price is not None and np.isfinite(exit_price) else np.nan
    return HoldingOutcome(float(total),float(price),float(shares),float(dividends),delisted)

def vectorized_total_return(entry: np.ndarray, exit_: np.ndarray, entry_sessions: np.ndarray, exit_sessions: np.ndarray,
                            actions: pd.DataFrame | None = None, *, security_ids: np.ndarray | None = None,
                            symbols: np.ndarray | None = None) -> tuple[np.ndarray,np.ndarray]:
    entry=np.asarray(entry,float); exit_=np.asarray(exit_,float); total=np.full(entry.shape,np.nan,np.float32); price=np.full(entry.shape,np.nan,np.float32)
    if actions is None or actions.empty:
        price=np.divide(exit_,entry,out=np.full(entry.shape,np.nan),where=np.isfinite(entry)&np.isfinite(exit_)&(entry>0))-1; return price.astype(np.float32),price.astype(np.float32)
    normalized=normalize_actions(actions)
    for index in np.ndindex(entry.shape):
        scoped = normalized
        if security_ids is not None:
            sid = str(np.asarray(security_ids)[index[-1]])
            mask = normalized.security_id.astype(str).eq(sid)
            if symbols is not None:
                mask |= normalized.symbol.astype(str).eq(str(np.asarray(symbols)[index[-1]]))
            scoped = normalized.loc[mask]
        outcome=calculate_holding_outcome(entry_price=entry[index],exit_price=exit_[index],entry_session=pd.Timestamp(entry_sessions[index]),exit_session=pd.Timestamp(exit_sessions[index]),actions=scoped)
        total[index]=outcome.total_return; price[index]=outcome.price_return
    return total,price

def cumulative_action_factors(actions: pd.DataFrame, sessions: pd.DatetimeIndex, security_ids: np.ndarray):
    """Return split and dividend cumulative factors indexed [date, security]."""
    normalized=normalize_actions(actions); dates=pd.DatetimeIndex(sessions); splits=np.ones((len(dates),len(security_ids)),np.float64); dividends=np.zeros_like(splits)
    for j,sid in enumerate(security_ids):
        subset=normalized.loc[normalized.security_id.astype(str).eq(str(sid))].sort_values(["effective_session","action_sequence"]); cumulative_split=1.; cumulative_dividend=0.
        for i,date in enumerate(dates):
            for row in subset.loc[subset.effective_session.eq(date.normalize())].itertuples(index=False):
                if row.action_type=="split": cumulative_split*=float(row.split_ratio)
                elif row.action_type=="cash_dividend": cumulative_dividend+=cumulative_split*float(row.cash_dividend_per_share)
            splits[i,j]=cumulative_split; dividends[i,j]=cumulative_dividend
    return splits,dividends

def interval_total_return(*,entry_price,exit_price,entry_split_index,exit_split_index,entry_dividend_index,exit_dividend_index,terminal_cash=None):
    output=np.full(np.shape(entry_price),np.nan,np.float32); e=np.asarray(entry_price,float); x=np.asarray(exit_price,float); es=np.asarray(entry_split_index,float); xs=np.asarray(exit_split_index,float); ed=np.asarray(entry_dividend_index,float); xd=np.asarray(exit_dividend_index,float); valid=np.isfinite(e)&np.isfinite(x)&np.isfinite(es)&np.isfinite(xs)&np.isfinite(ed)&np.isfinite(xd)&(e>0)&(es>0)
    shares=xs[valid]/es[valid]; div=(xd[valid]-ed[valid])/es[valid]; value=shares*x[valid]+div
    if terminal_cash is not None: value=np.where(np.isfinite(np.asarray(terminal_cash)[valid]),np.asarray(terminal_cash)[valid]+div,value)
    output[valid]=(value/e[valid]-1).astype(np.float32); return output
