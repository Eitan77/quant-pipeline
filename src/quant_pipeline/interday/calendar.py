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
    def __init__(self, calendar_name: str) -> None:
        self.name = calendar_name
        self._calendar = xcals.get_calendar(calendar_name)

    def sessions(self, start: str, end: str) -> pd.DatetimeIndex:
        return self._calendar.sessions_in_range(start, end)

    def clocks(self, start: str, end: str) -> pd.DataFrame:
        schedule = self._calendar.schedule.loc[start:end].copy().rename(columns={"open": "open_ts", "close": "close_ts"})
        schedule["session_date"] = schedule.index.tz_localize(None).normalize()
        local_open = schedule.open_ts.dt.tz_convert(ET)
        local_close = schedule.close_ts.dt.tz_convert(ET)
        schedule["shortened_session"] = (local_close - local_open) < pd.Timedelta(hours=6, minutes=30)
        return schedule.reset_index(drop=True)

    def offset_session(self, session: pd.Timestamp, offset: int) -> pd.Timestamp:
        sessions = self._calendar.sessions
        location = sessions.get_indexer([pd.Timestamp(session)])[0]
        if location < 0 or not 0 <= location + offset < len(sessions):
            raise IndexError(f"Session offset outside calendar coverage: {session}, {offset}")
        return sessions[location + offset]

    @staticmethod
    def checkpoint_bar_end(clock: SessionClock, checkpoint: str) -> pd.Timestamp:
        if checkpoint == "open5": return clock.open_ts + pd.Timedelta(minutes=5)
        if checkpoint == "open15": return clock.open_ts + pd.Timedelta(minutes=15)
        if checkpoint == "close5": return clock.close_ts
        if checkpoint == "close15": return clock.close_ts - pd.Timedelta(minutes=10)
        hour, minute = map(int, checkpoint.split(":"))
        local = pd.Timestamp.combine(clock.session_date.date(), time(hour, minute)).tz_localize(ET).tz_convert("UTC")
        if local <= clock.open_ts or local > clock.close_ts:
            raise ValueError(f"Checkpoint {checkpoint} is outside session {clock.session_date.date()}")
        return local
