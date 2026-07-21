
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import exchange_calendars as xcals
import pandas as pd


ET = "America/New_York"


@dataclass(frozen=True)
class SessionClock:
    session_date: pd.Timestamp
    open_ts: pd.Timestamp
    close_ts: pd.Timestamp
    shortened_session: bool


class TradingCalendar:
    def __init__(self, name: str) -> None:
        self._calendar = xcals.get_calendar(name)

    def clocks(self, start: str, end: str) -> pd.DataFrame:
        sessions = self._calendar.sessions_in_range(start, end)
        rows: list[dict] = []

        for session in sessions:
            open_ts = pd.Timestamp(
                self._calendar.session_open(session)
            ).tz_convert("UTC")
            close_ts = pd.Timestamp(
                self._calendar.session_close(session)
            ).tz_convert("UTC")

            minutes = int((close_ts - open_ts) / pd.Timedelta(minutes=1))
            if minutes <= 0 or minutes % 5:
                raise ValueError(f"Invalid five-minute session: {session}")

            rows.append(
                {
                    "session_date": pd.Timestamp(session)
                    .tz_localize(None)
                    .normalize(),
                    "open_ts": open_ts,
                    "close_ts": close_ts,
                    "expected_bar_count": minutes // 5,
                    "shortened_session": minutes < 390,
                }
            )

        return pd.DataFrame(rows)

    def offset_session(
        self,
        session_date: pd.Timestamp,
        offset: int,
    ) -> pd.Timestamp:
        session = pd.Timestamp(session_date)
        if session.tzinfo is None:
            session = session.tz_localize("UTC")
        else:
            session = session.tz_convert("UTC")

        sessions = self._calendar.sessions
        location = sessions.get_indexer([session])[0]
        if location < 0:
            raise KeyError(f"Not an exchange session: {session_date}")

        target = location + offset
        if target < 0 or target >= len(sessions):
            raise IndexError("Session offset is outside calendar coverage")

        return pd.Timestamp(sessions[target]).tz_localize(None).normalize()

    @staticmethod
    def checkpoint_bar_end(
        clock: SessionClock,
        checkpoint: str,
    ) -> pd.Timestamp:
        if checkpoint == "open5":
            return clock.open_ts + pd.Timedelta(minutes=5)
        if checkpoint == "close5":
            return clock.close_ts

        if checkpoint in {"open15", "close15"}:
            raise ValueError(
                f"{checkpoint} is a multi-bar VWAP window, not a single bar end"
            )

        hour, minute = map(int, checkpoint.split(":"))
        local = pd.Timestamp.combine(
            clock.session_date.date(),
            time(hour, minute),
        ).tz_localize(ET)
        utc = local.tz_convert("UTC")

        if utc <= clock.open_ts or utc > clock.close_ts:
            raise ValueError(
                f"Checkpoint {checkpoint} is outside session "
                f"{clock.session_date.date()}"
            )
        return utc
