
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class MeanInference:
    mean: float
    median: float
    std: float
    hac_standard_error: float
    hac_t: float
    pvalue: float
    positive_fraction: float
    n: int

    @classmethod
    def invalid(cls, n: int) -> "MeanInference":
        return cls(
            mean=np.nan,
            median=np.nan,
            std=np.nan,
            hac_standard_error=np.nan,
            hac_t=np.nan,
            pvalue=np.nan,
            positive_fraction=np.nan,
            n=n,
        )


def newey_west_mean_inference(
    values: np.ndarray,
    *,
    lag: int,
) -> MeanInference:
    series = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(series)
    n = int(finite.sum())

    if n < max(20, lag + 5):
        return MeanInference.invalid(n)

    mean = float(series[finite].mean())
    centered = np.where(finite, series - mean, np.nan)

    gamma_zero = float(
        np.nanmean(centered * centered)
    )
    long_run_variance = gamma_zero

    maximum_lag = min(lag, len(series) - 1)

    for current_lag in range(1, maximum_lag + 1):
        left = centered[current_lag:]
        right = centered[:-current_lag]
        pairwise = np.isfinite(left) & np.isfinite(right)

        if not pairwise.any():
            continue

        covariance = float(
            np.mean(left[pairwise] * right[pairwise])
        )
        weight = 1.0 - current_lag / (maximum_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance

    if long_run_variance < -1e-12:
        return MeanInference.invalid(n)

    long_run_variance = max(long_run_variance, 0.0)
    standard_error = float(
        np.sqrt(long_run_variance / n)
    )

    if standard_error <= 0:
        t_statistic = np.nan
        pvalue = np.nan
    else:
        t_statistic = mean / standard_error
        pvalue = float(
            2.0 * norm.sf(abs(t_statistic))
        )

    return MeanInference(
        mean=mean,
        median=float(np.nanmedian(series)),
        std=float(np.nanstd(series, ddof=1)),
        hac_standard_error=standard_error,
        hac_t=float(t_statistic),
        pvalue=pvalue,
        positive_fraction=float(
            np.mean(series[finite] > 0)
        ),
        n=n,
    )



def benjamini_hochberg(
    pvalues: pd.Series,
) -> pd.Series:
    output = pd.Series(
        np.nan,
        index=pvalues.index,
        dtype=float,
    )

    valid = pvalues.notna()
    values = pvalues.loc[valid].astype(float)

    if ((values < 0) | (values > 1)).any():
        raise ValueError("P-values must be in [0, 1]")

    if values.empty:
        return output

    order = np.argsort(values.to_numpy())
    ordered = values.to_numpy()[order]
    n = len(ordered)

    adjusted = ordered * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    restored = np.empty(n, dtype=float)
    restored[order] = adjusted

    output.loc[valid] = restored
    return output
