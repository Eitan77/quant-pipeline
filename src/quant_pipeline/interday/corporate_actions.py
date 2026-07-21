
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class ActionStatus(IntEnum):
    VALID = 0
    UNRESOLVED_TERMINAL_VALUE = 1
    INVALID_SPLIT_RATIO = 2
    DUPLICATE_ACTION_SEQUENCE = 3


REQUIRED_ACTION_COLUMNS = {
    "security_id",
    "effective_session",
    "action_sequence",
    "action_type",
    "split_ratio",
    "cash_dividend_per_share",
    "terminal_cash_per_share",
    "reference_price_before_delisting",
    "delisting_return_from_reference",
    "known_at_ts",
}


@dataclass
class CorporateActionIndex:
    sessions: pd.DatetimeIndex
    security_ids: np.ndarray

    split_index: np.ndarray       # [date, security], cumulative A_t
    dividend_index: np.ndarray    # [date, security], cumulative C_t

    terminal_cash: np.ndarray     # [date, security]
    terminal_status: np.ndarray   # int8 [date, security]

    action_valid: np.ndarray      # [date, security]


def slice_action_index(
    action_index: CorporateActionIndex,
    security_index: int,
) -> CorporateActionIndex:
    """
    Return a one-security CorporateActionIndex.

    The result preserves the two-dimensional [date, security] contract so it
    can be passed to interval_total_return with benchmark prices shaped
    [date, 1].
    """
    if security_index < 0 or security_index >= len(action_index.security_ids):
        raise IndexError(
            f"security_index {security_index} is outside "
            f"0..{len(action_index.security_ids) - 1}"
        )

    column = slice(security_index, security_index + 1)

    return CorporateActionIndex(
        sessions=action_index.sessions,
        security_ids=np.asarray(
            [action_index.security_ids[security_index]],
            dtype=action_index.security_ids.dtype,
        ),
        split_index=np.ascontiguousarray(
            action_index.split_index[:, column]
        ),
        dividend_index=np.ascontiguousarray(
            action_index.dividend_index[:, column]
        ),
        terminal_cash=np.ascontiguousarray(
            action_index.terminal_cash[:, column]
        ),
        terminal_status=np.ascontiguousarray(
            action_index.terminal_status[:, column]
        ),
        action_valid=np.ascontiguousarray(
            action_index.action_valid[:, column]
        ),
    )


def normalize_actions(
    actions: pd.DataFrame,
    *,
    discovery_end: str,
) -> pd.DataFrame:
    frame = actions.copy()

    aliases = {
        "cash_amount": "cash_dividend_per_share",
        "cash_out_value": "terminal_cash_per_share",
    }
    frame = frame.rename(
        columns={k: v for k, v in aliases.items() if k in frame}
    )

    for column in REQUIRED_ACTION_COLUMNS:
        if column not in frame:
            frame[column] = np.nan

    frame["security_id"] = frame["security_id"].astype(str)
    frame["effective_session"] = pd.to_datetime(
        frame["effective_session"]
    ).dt.normalize()
    frame["known_at_ts"] = pd.to_datetime(
        frame["known_at_ts"],
        utc=True,
        errors="coerce",
    )
    frame["action_sequence"] = pd.to_numeric(
        frame["action_sequence"],
        errors="coerce",
    ).fillna(0).astype(int)

    frame = frame[
        frame["effective_session"] <= pd.Timestamp(discovery_end)
    ].copy()

    key = ["security_id", "effective_session", "action_sequence"]
    if frame.duplicated(key).any():
        raise ValueError("Duplicate corporate-action sequence rows")

    valid_types = {
        "split",
        "cash_dividend",
        "cash_merger",
        "delisting",
    }
    invalid = set(frame["action_type"].dropna()) - valid_types
    if invalid:
        raise ValueError(f"Unknown corporate-action types: {sorted(invalid)}")

    split = frame["action_type"].eq("split")
    invalid_split = split & (
        frame["split_ratio"].isna()
        | frame["split_ratio"].le(0)
    )
    if invalid_split.any():
        raise ValueError("Invalid split ratio")

    return frame.sort_values(key, kind="stable").reset_index(drop=True)

def build_corporate_action_index(
    actions: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    security_ids: np.ndarray,
) -> CorporateActionIndex:
    date_lookup = {
        pd.Timestamp(value).normalize(): index
        for index, value in enumerate(sessions)
    }
    security_lookup = {
        str(value): index
        for index, value in enumerate(security_ids)
    }

    dates = len(sessions)
    securities = len(security_ids)

    daily_split = np.ones((dates, securities), dtype=np.float64)
    daily_dividend = np.zeros((dates, securities), dtype=np.float64)
    terminal_cash = np.full((dates, securities), np.nan, dtype=np.float64)
    terminal_status = np.zeros((dates, securities), dtype=np.int8)
    action_valid = np.ones((dates, securities), dtype=bool)

    for row in actions.itertuples(index=False):
        security = security_lookup.get(str(row.security_id))
        date = date_lookup.get(pd.Timestamp(row.effective_session).normalize())
        if security is None or date is None:
            continue

        action_type = str(row.action_type)

        if action_type == "split":
            daily_split[date, security] *= float(row.split_ratio)

        elif action_type == "cash_dividend":
            amount = float(row.cash_dividend_per_share)
            if not np.isfinite(amount):
                action_valid[date:, security] = False
                continue
            # Sequence ordering is already deterministic. Dividend is per
            # post-lower-sequence share for that session.
            daily_dividend[date, security] += amount

        elif action_type in {"cash_merger", "delisting"}:
            cash = float(row.terminal_cash_per_share)
            if not np.isfinite(cash):
                reference = float(row.reference_price_before_delisting)
                delisting_return = float(row.delisting_return_from_reference)
                if np.isfinite(reference) and np.isfinite(delisting_return):
                    cash = reference * (1.0 + delisting_return)

            if not np.isfinite(cash):
                terminal_status[date, security] = (
                    ActionStatus.UNRESOLVED_TERMINAL_VALUE
                )
                action_valid[date:, security] = False
            else:
                terminal_cash[date, security] = cash

    split_index = np.cumprod(daily_split, axis=0)

    # C_t = cumulative sum(A_t * dividend_t)
    dividend_index = np.cumsum(
        split_index * daily_dividend,
        axis=0,
    )

    return CorporateActionIndex(
        sessions=sessions,
        security_ids=security_ids,
        split_index=split_index.astype(np.float32),
        dividend_index=dividend_index.astype(np.float32),
        terminal_cash=terminal_cash.astype(np.float32),
        terminal_status=terminal_status,
        action_valid=action_valid,
    )


def interval_total_return(
    *,
    entry_price: np.ndarray,
    exit_price: np.ndarray,
    entry_date_ids: np.ndarray,
    exit_date_ids: np.ndarray,
    action_index: CorporateActionIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    entry_price and exit_price: [date, security]
    entry_date_ids and exit_date_ids: [date], -1 invalid

    Returns:
        total_return [date, security]
        price_return [date, security]
        unresolved_terminal [date, security]
    """
    if entry_price.ndim != 2 or exit_price.ndim != 2:
        raise ValueError(
            "entry_price and exit_price must be [date, security]"
        )

    if entry_price.shape != exit_price.shape:
        raise ValueError(
            "entry_price and exit_price shapes must match: "
            f"{entry_price.shape} != {exit_price.shape}"
        )

    expected_security_count = entry_price.shape[1]

    action_arrays = {
        "split_index": action_index.split_index,
        "dividend_index": action_index.dividend_index,
        "terminal_cash": action_index.terminal_cash,
        "terminal_status": action_index.terminal_status,
        "action_valid": action_index.action_valid,
    }

    for name, array in action_arrays.items():
        if array.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if array.shape[1] != expected_security_count:
            raise ValueError(
                f"{name} has {array.shape[1]} securities but prices have "
                f"{expected_security_count}"
            )

    dates, securities = entry_price.shape

    total_return = np.full(
        (dates, securities),
        np.nan,
        dtype=np.float32,
    )
    price_return = np.full_like(total_return, np.nan)
    unresolved = np.zeros((dates, securities), dtype=bool)

    for decision_date in range(dates):
        entry_date = int(entry_date_ids[decision_date])
        exit_date = int(exit_date_ids[decision_date])

        if entry_date < 0 or exit_date < 0:
            continue

        entry = entry_price[decision_date]
        exit_ = exit_price[decision_date]

        entry_split = action_index.split_index[entry_date]
        exit_split = action_index.split_index[exit_date]
        entry_dividend = action_index.dividend_index[entry_date]
        exit_dividend = action_index.dividend_index[exit_date]

        valid = (
            np.isfinite(entry)
            & np.isfinite(exit_)
            & (entry > 0)
            & (exit_ > 0)
            & (entry_split > 0)
        )

        shares = np.full(securities, np.nan, dtype=np.float64)
        dividends = np.full(securities, np.nan, dtype=np.float64)

        shares[valid] = (
            exit_split[valid] / entry_split[valid]
        )
        dividends[valid] = (
            (exit_dividend[valid] - entry_dividend[valid])
            / entry_split[valid]
        )

        terminal_value = shares * exit_ + dividends

        # First terminal event in (entry_date, exit_date].
        terminal_slice = action_index.terminal_cash[
            entry_date + 1 : exit_date + 1
        ]
        status_slice = action_index.terminal_status[
            entry_date + 1 : exit_date + 1
        ]

        if len(terminal_slice):
            for security in range(securities):
                terminal_events = np.flatnonzero(
                    np.isfinite(terminal_slice[:, security])

                )
                unresolved_events = np.flatnonzero(
                    status_slice[:, security]
                    == ActionStatus.UNRESOLVED_TERMINAL_VALUE
                )

                if len(unresolved_events) and (
                    not len(terminal_events)
                    or unresolved_events[0] <= terminal_events[0]
                ):
                    unresolved[decision_date, security] = True
                    valid[security] = False
                    continue

                if len(terminal_events):
                    event_offset = int(terminal_events[0])
                    event_date = entry_date + 1 + event_offset
                    split_at_event = action_index.split_index[
                        event_date, security
                    ]
                    shares_at_event = (
                        split_at_event / entry_split[security]
                    )
                    cash = terminal_slice[
                        event_offset, security
                    ] * shares_at_event
                    dividends_to_event = (
                        action_index.dividend_index[
                            event_date, security
                        ] - entry_dividend[security]
                    ) / entry_split[security]
                    terminal_value[security] = (
                        cash + dividends_to_event
                    )

        price_return_row = np.full(securities, np.nan, dtype=np.float64)
        price_return_row[valid] = (
            shares[valid] * exit_[valid] / entry[valid] - 1.0
        )

        total_return_row = np.full(securities, np.nan, dtype=np.float64)
        total_return_row[valid] = (
            terminal_value[valid] / entry[valid] - 1.0
        )

        price_return[decision_date] = price_return_row.astype(np.float32)
        total_return[decision_date] = total_return_row.astype(np.float32)

    return total_return, price_return, unresolved


@dataclass
class AdjustedDailyPrices:
    split_adjusted_open: np.ndarray
    split_adjusted_high: np.ndarray
    split_adjusted_low: np.ndarray
    split_adjusted_close: np.ndarray
    split_adjusted_volume: np.ndarray

    daily_total_return: np.ndarray
    total_return_index: np.ndarray
    overnight_total_return: np.ndarray
    regular_session_return: np.ndarray
def build_adjusted_daily_prices(
    raw_open: np.ndarray,
    raw_high: np.ndarray,
    raw_low: np.ndarray,
    raw_close: np.ndarray,
    raw_volume: np.ndarray,
    action_index: CorporateActionIndex,
) -> AdjustedDailyPrices:
    final_split = action_index.split_index[-1]
    adjustment = (
        action_index.split_index
        / final_split[None, :]
    )

    split_open = raw_open * adjustment
    split_high = raw_high * adjustment
    split_low = raw_low * adjustment
    split_close = raw_close * adjustment

    # Share volume adjusts inversely to price.
    split_volume = np.divide(
        raw_volume,
        adjustment,
        out=np.full_like(raw_volume, np.nan, dtype=np.float32),
        where=np.isfinite(adjustment) & (adjustment > 0),
    )

    dates = raw_close.shape[0]
    entry_ids = np.arange(dates, dtype=np.int32) - 1
    exit_ids = np.arange(dates, dtype=np.int32)
    entry_ids[0] = -1

    prior_close = np.full_like(raw_close, np.nan)
    prior_close[1:] = raw_close[:-1]

    total_return, _, _ = interval_total_return(
        entry_price=prior_close,
        exit_price=raw_close,
        entry_date_ids=entry_ids,
        exit_date_ids=exit_ids,
        action_index=action_index,
    )

    total_return_index = np.full_like(total_return, np.nan)
    total_return_index[0] = 1.0

    for date in range(1, dates):
        previous = total_return_index[date - 1]
        current_return = total_return[date]
        valid = np.isfinite(previous) & np.isfinite(current_return)
        total_return_index[date, valid] = (
            previous[valid] * (1.0 + current_return[valid])
        )

    prior_raw_close = np.full_like(raw_close, np.nan)
    prior_raw_close[1:] = raw_close[:-1]

    overnight_total, _, _ = interval_total_return(
        entry_price=prior_raw_close,
        exit_price=raw_open,
        entry_date_ids=entry_ids,
        exit_date_ids=exit_ids,
        action_index=action_index,
    )

    regular = np.divide(
        raw_close,
        raw_open,
        out=np.full_like(raw_close, np.nan),
        where=np.isfinite(raw_open) & (raw_open > 0),
    ) - 1.0

    return AdjustedDailyPrices(
        split_adjusted_open=split_open.astype(np.float32),
        split_adjusted_high=split_high.astype(np.float32),
        split_adjusted_low=split_low.astype(np.float32),
        split_adjusted_close=split_close.astype(np.float32),
        split_adjusted_volume=split_volume.astype(np.float32),
        daily_total_return=total_return.astype(np.float32),
        total_return_index=total_return_index.astype(np.float32),
        overnight_total_return=overnight_total.astype(np.float32),
        regular_session_return=regular.astype(np.float32),
    )
