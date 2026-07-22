from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .corporate_actions import build_adjusted_daily_prices


@dataclass
class DensePanel:
    sessions: pd.DatetimeIndex
    security_ids: np.ndarray
    symbols: np.ndarray
    key_frame: pd.DataFrame
    arrays: dict[str, np.ndarray]
    valid: np.ndarray


@dataclass
class PrimitiveBundle:
    sessions: pd.DatetimeIndex
    security_ids: np.ndarray
    symbols: np.ndarray

    split_adjusted_open: np.ndarray
    split_adjusted_high: np.ndarray
    split_adjusted_low: np.ndarray
    split_adjusted_close: np.ndarray
    split_adjusted_session_vwap: np.ndarray

    split_adjusted_open5: np.ndarray
    split_adjusted_open15: np.ndarray
    split_adjusted_0940: np.ndarray
    split_adjusted_0945: np.ndarray
    split_adjusted_1000: np.ndarray
    split_adjusted_1015: np.ndarray
    split_adjusted_1030: np.ndarray
    split_adjusted_1100: np.ndarray
    split_adjusted_1200: np.ndarray
    split_adjusted_1300: np.ndarray
    split_adjusted_1400: np.ndarray
    split_adjusted_1500: np.ndarray
    split_adjusted_close15: np.ndarray
    split_adjusted_close5: np.ndarray

    raw_open5: np.ndarray
    raw_open15: np.ndarray
    raw_close15: np.ndarray
    raw_close5: np.ndarray

    daily_total_return: np.ndarray
    daily_log_total_return: np.ndarray
    total_return_index: np.ndarray
    total_return_log_index: np.ndarray
    overnight_total_return: np.ndarray
    regular_session_return: np.ndarray

    benchmark_daily_total_return: np.ndarray
    benchmark_daily_log_total_return: np.ndarray
    benchmark_total_return_index: np.ndarray
    benchmark_total_return_log_index: np.ndarray
    benchmark_overnight_total_return: np.ndarray

    volume: np.ndarray
    dollar_volume: np.ndarray
    first_60m_volume: np.ndarray
    last_60m_volume: np.ndarray
    open_30m_volume: np.ndarray
    close_30m_volume: np.ndarray
    largest_5m_volume: np.ndarray

    first_60m_return: np.ndarray
    last_60m_return: np.ndarray

    sector_codes: np.ndarray | None
    industry_codes: np.ndarray | None

    @property
    def close_log(self) -> np.ndarray:
        return np.log(self.split_adjusted_close).astype(np.float32)

    @property
    def close_return(self) -> np.ndarray:
        return self.daily_total_return

    @property
    def regular_return(self) -> np.ndarray:
        return self.regular_session_return

    @property
    def high(self) -> np.ndarray:
        return self.split_adjusted_high

    @property
    def low(self) -> np.ndarray:
        return self.split_adjusted_low

    @property
    def open(self) -> np.ndarray:
        return self.split_adjusted_open

    @property
    def close(self) -> np.ndarray:
        return self.split_adjusted_close

    @property
    def market_return(self) -> np.ndarray:
        return self.benchmark_daily_total_return

    @property
    def market_overnight_return(self) -> np.ndarray:
        return self.benchmark_overnight_total_return

    @property
    def session_vwap(self) -> np.ndarray:
        return self.split_adjusted_session_vwap

    @property
    def open5(self) -> np.ndarray:
        return self.split_adjusted_open5

    @property
    def close5(self) -> np.ndarray:
        return self.split_adjusted_close5

    @property
    def close15(self) -> np.ndarray:
        return self.split_adjusted_close15

    @property
    def checkpoint_30m(self) -> np.ndarray:
        return self.split_adjusted_1000

    @property
    def checkpoint_60m(self) -> np.ndarray:
        return self.split_adjusted_1030

    @property
    def midday(self) -> np.ndarray:
        return self.split_adjusted_1200


def shift_2d(values: np.ndarray, periods: int) -> np.ndarray:
    output = np.full_like(values, np.nan)
    if periods == 0:
        output[:] = values
    elif periods > 0:
        output[periods:] = values[:-periods]
    else:
        output[:periods] = values[-periods:]
    return output


shift = shift_2d


def shift_1d(values: np.ndarray, periods: int) -> np.ndarray:
    output = np.full_like(values, np.nan)
    if periods == 0:
        output[:] = values
    elif periods > 0:
        output[periods:] = values[:-periods]
    else:
        output[:periods] = values[-periods:]
    return output


def rolling_sum_2d(
    values: np.ndarray,
    window: int,
    minimum_observations: int | None = None,
) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("rolling_sum_2d expects [date, security]")
    if window <= 0:
        raise ValueError("window must be positive")
    minimum = window if minimum_observations is None else minimum_observations
    finite = np.isfinite(values)
    numeric = np.where(finite, values, 0.0).astype(np.float64)
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1])), np.cumsum(numeric, axis=0)]
    )
    counts = np.vstack(
        [np.zeros((1, values.shape[1])), np.cumsum(finite, axis=0)]
    )
    sums = cumulative[window:] - cumulative[:-window]
    observations = counts[window:] - counts[:-window]
    output = np.full(values.shape, np.nan, dtype=np.float32)
    valid = observations >= minimum
    output[window - 1 :][valid] = sums[valid].astype(np.float32)
    return output


rolling_sum = rolling_sum_2d


def rolling_mean_2d(
    values: np.ndarray,
    window: int,
    minimum_observations: int | None = None,
) -> np.ndarray:
    minimum = window if minimum_observations is None else minimum_observations
    sums = rolling_sum_2d(values, window, minimum_observations=1)
    counts = rolling_sum_2d(
        np.isfinite(values).astype(np.float32),
        window,
        minimum_observations=1,
    )
    output = np.full(values.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(sums) & (counts >= minimum)
    output[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return output


rolling_mean = rolling_mean_2d


def rolling_std_2d(
    values: np.ndarray,
    window: int,
    minimum_observations: int | None = None,
) -> np.ndarray:
    mean = rolling_mean_2d(values, window, minimum_observations)
    mean_square = rolling_mean_2d(values * values, window, minimum_observations)
    variance = mean_square.astype(np.float64) - mean.astype(np.float64) ** 2
    variance[(variance < 0) & (variance > -1e-12)] = 0.0
    variance[variance <= -1e-12] = np.nan
    return np.sqrt(variance).astype(np.float32)


rolling_std = rolling_std_2d


def rolling_max(matrix, window, min_periods=None):
    minimum = window if min_periods is None else min_periods
    values = np.asarray(matrix, float)
    output = np.full(values.shape, np.nan, np.float32)
    for i in range(window - 1, len(values)):
        segment = values[i - window + 1 : i + 1]
        valid = np.isfinite(segment)
        count = valid.sum(axis=0)
        output[i] = np.where(valid.any(axis=0), np.nanmax(segment, axis=0), np.nan)
        output[i, count < minimum] = np.nan
    return output


def rolling_min(matrix, window, min_periods=None):
    return -rolling_max(-np.asarray(matrix), window, min_periods)


def to_dense_panel(panel: pd.DataFrame, *, value_columns: list[str]) -> DensePanel:
    keys = panel[["security_id", "session_date"]]
    if keys.duplicated().any():
        raise ValueError("Dense panel contains duplicate security-date keys")

    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(panel.session_date).unique()))
    security_ids = np.array(sorted(panel.security_id.astype(str).unique()))
    date_index = {value: i for i, value in enumerate(sessions)}
    security_index = {value: i for i, value in enumerate(security_ids)}
    dates = panel.session_date.map(date_index).to_numpy()
    securities = panel.security_id.astype(str).map(security_index).to_numpy()
    shape = (len(sessions), len(security_ids))

    arrays: dict[str, np.ndarray] = {}
    for column in value_columns:
        values = np.full(shape, np.nan, np.float32)
        values[dates, securities] = pd.to_numeric(
            panel[column], errors="coerce"
        ).to_numpy(np.float32)
        arrays[column] = values

    valid = np.zeros(shape, bool)
    valid[dates, securities] = panel.analysis_eligible.to_numpy(bool)
    symbols_by_security = (
        panel[["security_id", "symbol"]]
        .drop_duplicates("security_id")
        .set_index("security_id")["symbol"]
    )
    symbols = np.array([symbols_by_security.get(value, value) for value in security_ids])
    key_frame = panel[["security_id", "symbol", "session_date"]].sort_values(
        ["session_date", "security_id"], kind="stable"
    ).reset_index(drop=True)
    return DensePanel(sessions, security_ids, symbols, key_frame, arrays, valid)


def adjust_intraday_price(
    raw_price: np.ndarray,
    split_index: np.ndarray,
) -> np.ndarray:
    """
    Convert a raw historical price matrix to the same final-share basis used
    by split-adjusted daily OHLC.

    raw_price and split_index are [date, security].
    """
    if raw_price.shape != split_index.shape:
        raise ValueError("raw_price and split_index shapes must match")

    final_split = split_index[-1]
    adjustment = np.divide(
        split_index,
        final_split[None, :],
        out=np.full_like(split_index, np.nan, dtype=np.float32),
        where=np.isfinite(split_index)
        & np.isfinite(final_split[None, :])
        & (final_split[None, :] > 0),
    )
    return (raw_price * adjustment).astype(np.float32)


def resolve_benchmark_index(
    security_ids: np.ndarray,
    symbols: np.ndarray,
    benchmark_symbol: str,
) -> int:
    matches = np.flatnonzero(symbols.astype(str) == str(benchmark_symbol))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one benchmark security, found {len(matches)}"
        )
    return int(matches[0])


def build_primitives(
    dense: DensePanel,
    benchmark_symbol="QQQ",
    sector_codes=None,
    industry_codes=None,
    action_index=None,
) -> PrimitiveBundle:
    required_checkpoints = (
        "open5", "open15", "09:40", "09:45", "10:00", "10:15",
        "10:30", "11:00", "12:00", "13:00", "14:00", "15:00",
        "close15", "close5",
    )
    missing = [name for name in required_checkpoints if name not in dense.arrays]
    if missing:
        raise ValueError("Dense panel is missing required checkpoints: " + ", ".join(missing))

    raw_checkpoint = {
        name: dense.arrays[name].astype(np.float32, copy=False)
        for name in required_checkpoints
    }
    raw_open = dense.arrays["open"]
    raw_high = dense.arrays["high"]
    raw_low = dense.arrays["low"]
    raw_close = dense.arrays["close"]
    raw_volume = dense.arrays["volume"]

    if action_index is not None:
        adjusted_prices = build_adjusted_daily_prices(
            raw_open, raw_high, raw_low, raw_close, raw_volume, action_index
        )
        adjusted_checkpoint = {
            name: adjust_intraday_price(values, action_index.split_index)
            for name, values in raw_checkpoint.items()
        }
        adjusted_session_vwap = adjust_intraday_price(
            dense.arrays["session_vwap"], action_index.split_index
        )
    else:
        adjusted_prices = None
        adjusted_checkpoint = raw_checkpoint
        adjusted_session_vwap = dense.arrays["session_vwap"]

    if adjusted_prices is None:
        split_open, split_high, split_low, split_close = (
            raw_open, raw_high, raw_low, raw_close
        )
        daily_total_return = np.expm1(
            np.log(split_close) - shift_2d(np.log(split_close), 1)
        ).astype(np.float32)
        total_return_index = np.cumprod(
            np.where(np.isfinite(daily_total_return), 1.0 + daily_total_return, 1.0),
            axis=0,
        ).astype(np.float32)
        overnight = split_open / shift_2d(split_close, 1) - 1.0
        regular = split_close / split_open - 1.0
    else:
        split_open = adjusted_prices.split_adjusted_open
        split_high = adjusted_prices.split_adjusted_high
        split_low = adjusted_prices.split_adjusted_low
        split_close = adjusted_prices.split_adjusted_close
        daily_total_return = adjusted_prices.daily_total_return
        total_return_index = adjusted_prices.total_return_index
        overnight = adjusted_prices.overnight_total_return
        regular = adjusted_prices.regular_session_return

    daily_log_total_return = np.where(
        np.isfinite(daily_total_return) & (daily_total_return > -1.0),
        np.log1p(daily_total_return),
        np.nan,
    ).astype(np.float32)

    benchmark_index = resolve_benchmark_index(
        dense.security_ids, dense.symbols, benchmark_symbol
    )
    benchmark_daily = daily_total_return[:, benchmark_index]
    benchmark_daily_log = daily_log_total_return[:, benchmark_index]
    benchmark_index_values = total_return_index[:, benchmark_index]
    benchmark_log_index = np.log(benchmark_index_values).astype(np.float32)
    benchmark_overnight = overnight[:, benchmark_index]

    return PrimitiveBundle(
        sessions=dense.sessions,
        security_ids=dense.security_ids,
        symbols=dense.symbols,
        split_adjusted_open=split_open,
        split_adjusted_high=split_high,
        split_adjusted_low=split_low,
        split_adjusted_close=split_close,
        split_adjusted_session_vwap=adjusted_session_vwap,
        split_adjusted_open5=adjusted_checkpoint["open5"],
        split_adjusted_open15=adjusted_checkpoint["open15"],
        split_adjusted_0940=adjusted_checkpoint["09:40"],
        split_adjusted_0945=adjusted_checkpoint["09:45"],
        split_adjusted_1000=adjusted_checkpoint["10:00"],
        split_adjusted_1015=adjusted_checkpoint["10:15"],
        split_adjusted_1030=adjusted_checkpoint["10:30"],
        split_adjusted_1100=adjusted_checkpoint["11:00"],
        split_adjusted_1200=adjusted_checkpoint["12:00"],
        split_adjusted_1300=adjusted_checkpoint["13:00"],
        split_adjusted_1400=adjusted_checkpoint["14:00"],
        split_adjusted_1500=adjusted_checkpoint["15:00"],
        split_adjusted_close15=adjusted_checkpoint["close15"],
        split_adjusted_close5=adjusted_checkpoint["close5"],
        raw_open5=raw_checkpoint["open5"],
        raw_open15=raw_checkpoint["open15"],
        raw_close15=raw_checkpoint["close15"],
        raw_close5=raw_checkpoint["close5"],
        daily_total_return=daily_total_return,
        daily_log_total_return=daily_log_total_return,
        total_return_index=total_return_index,
        total_return_log_index=np.log(total_return_index).astype(np.float32),
        overnight_total_return=overnight,
        regular_session_return=regular,
        benchmark_daily_total_return=benchmark_daily,
        benchmark_daily_log_total_return=benchmark_daily_log,
        benchmark_total_return_index=benchmark_index_values,
        benchmark_total_return_log_index=benchmark_log_index,
        benchmark_overnight_total_return=benchmark_overnight,
        volume=dense.arrays["volume"],
        dollar_volume=dense.arrays["dollar_volume"],
        first_60m_volume=dense.arrays["first_60m_volume"],
        last_60m_volume=dense.arrays["last_60m_volume"],
        open_30m_volume=dense.arrays["open_30m_volume"],
        close_30m_volume=dense.arrays["close_30m_volume"],
        largest_5m_volume=dense.arrays["largest_5m_volume"],
        first_60m_return=(
            adjusted_checkpoint["10:30"] / adjusted_checkpoint["open5"] - 1.0
        ).astype(np.float32),
        last_60m_return=(
            adjusted_checkpoint["close5"] / adjusted_checkpoint["15:00"] - 1.0
        ).astype(np.float32),
        sector_codes=sector_codes if sector_codes is not None else dense.arrays.get("sector_code"),
        industry_codes=industry_codes if industry_codes is not None else dense.arrays.get("industry_code"),
    )
