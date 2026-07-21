from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class HoldingOutcome:
    total_return: float; price_return: float; split_multiplier: float; dividends_received: float; delisted: bool

ACTION_COLUMNS = ("security_id","symbol","effective_session","action_type","split_ratio","cash_dividend_per_pre_action_share","delisting_return","cash_out_value","known_at_ts","source")

def normalize_actions(actions: pd.DataFrame | None) -> pd.DataFrame:
    if actions is None or actions.empty: return pd.DataFrame(columns=ACTION_COLUMNS)
    out = actions.copy()
    aliases = {"ex_date":"effective_session","cash_amount":"cash_dividend_per_pre_action_share","split_factor":"split_ratio"}
    out = out.rename(columns={k:v for k,v in aliases.items() if k in out})
    for c in ACTION_COLUMNS:
        if c not in out: out[c] = np.nan
    out["effective_session"] = pd.to_datetime(out.effective_session).dt.normalize()
    out["known_at_ts"] = pd.to_datetime(out.known_at_ts, utc=True, errors="coerce")
    return out.loc[:, ACTION_COLUMNS]

def calculate_holding_outcome(*, entry_price: float, exit_price: float | None, entry_session: pd.Timestamp, exit_session: pd.Timestamp, actions: pd.DataFrame) -> HoldingOutcome:
    if not np.isfinite(entry_price) or entry_price <= 0: return HoldingOutcome(np.nan,np.nan,np.nan,np.nan,False)
    relevant = actions.loc[actions.effective_session.gt(entry_session) & actions.effective_session.le(exit_session)].sort_values("effective_session", kind="stable")
    shares, dividends, delisted, terminal_cash = 1.0, 0.0, False, None
    for action in relevant.itertuples(index=False):
        kind = str(action.action_type)
        if kind == "split":
            ratio = float(action.split_ratio)
            if not np.isfinite(ratio) or ratio <= 0: raise ValueError("Invalid split ratio")
            shares *= ratio
        elif kind == "cash_dividend": dividends += shares * float(action.cash_dividend_per_pre_action_share)
        elif kind in {"delisting","cash_merger"}:
            delisted = True
            if np.isfinite(action.cash_out_value): terminal_cash = shares * float(action.cash_out_value)
            elif np.isfinite(action.delisting_return): terminal_cash = entry_price * (1 + float(action.delisting_return))
            else: return HoldingOutcome(np.nan,np.nan,shares,dividends,True)
            break
    if terminal_cash is None:
        if exit_price is None or not np.isfinite(exit_price) or exit_price <= 0: return HoldingOutcome(np.nan,np.nan,shares,dividends,delisted)
        terminal_cash = shares * float(exit_price)
    total = (terminal_cash + dividends) / entry_price - 1
    price = shares * float(exit_price) / entry_price - 1 if exit_price is not None and np.isfinite(exit_price) else np.nan
    return HoldingOutcome(float(total),float(price),float(shares),float(dividends),delisted)

def vectorized_total_return(entry: np.ndarray, exit_: np.ndarray, entry_sessions: np.ndarray, exit_sessions: np.ndarray, actions: pd.DataFrame | None = None) -> tuple[np.ndarray,np.ndarray]:
    total = np.divide(exit_, entry, out=np.full_like(entry, np.nan, dtype=float), where=np.isfinite(entry)&np.isfinite(exit_)&(entry>0)) - 1
    return total.astype(np.float32), total.astype(np.float32)
