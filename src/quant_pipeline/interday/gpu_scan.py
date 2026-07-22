from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import DailyPairSeries


@dataclass(frozen=True)
class CudaScanDiagnostics:
    backend: str
    device_name: str
    feature_count: int
    target_count: int
    date_count: int
    security_count: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def require_cuda(device_name: str, minimum_free_memory_bytes: int = 0):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Interday 2A full scan requires CUDA, but torch.cuda.is_available() is false. Use an explicit CPU smoke configuration only for small reference runs.")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < minimum_free_memory_bytes:
        raise RuntimeError(
            f"CUDA device has only {free_bytes} free bytes; "
            f"at least {minimum_free_memory_bytes} are required for the full scan"
        )
    return torch, device


def _masked_average_rank_2d(values, valid):
    import torch
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("values and valid must have identical [row, security] shape")
    rows, securities = values.shape
    if securities == 0:
        return torch.empty_like(values, dtype=torch.float64), torch.zeros(rows, device=values.device, dtype=torch.int64)
    work = torch.where(valid, values, torch.full_like(values, torch.inf))
    sorted_values, order = torch.sort(work, dim=1, stable=True)
    sorted_valid = torch.gather(valid, 1, order)
    new_group = torch.ones((rows, securities), device=values.device, dtype=torch.bool)
    if securities > 1:
        new_group[:, 1:] = (~sorted_valid[:, 1:] | ~sorted_valid[:, :-1] | (sorted_values[:, 1:] != sorted_values[:, :-1]))
    group_id = torch.cumsum(new_group.to(torch.int64), dim=1) - 1
    row_offset = torch.arange(rows, device=values.device, dtype=torch.int64)[:, None] * securities
    flat_group = (row_offset + group_id).reshape(-1)
    positions = torch.arange(1, securities + 1, device=values.device, dtype=torch.float64)[None, :].expand(rows, securities)
    weights = sorted_valid.to(torch.float64)
    group_position_sum = torch.zeros(rows * securities, device=values.device, dtype=torch.float64)
    group_count = torch.zeros_like(group_position_sum)
    group_position_sum.scatter_add_(0, flat_group, (positions * weights).reshape(-1))
    group_count.scatter_add_(0, flat_group, weights.reshape(-1))
    mean_rank = torch.full_like(group_position_sum, torch.nan)
    nonempty = group_count > 0
    mean_rank[nonempty] = group_position_sum[nonempty] / group_count[nonempty]
    sorted_rank = mean_rank[flat_group].reshape(rows, securities)
    sorted_rank = torch.where(sorted_valid, sorted_rank, torch.full_like(sorted_rank, torch.nan))
    ranks = torch.empty_like(sorted_rank)
    ranks.scatter_(1, order, sorted_rank)
    distinct = (new_group & sorted_valid).sum(dim=1, dtype=torch.int64)
    return ranks, distinct


def _rowwise_pearson(x, y, valid, minimum_count: int):
    import torch
    count = valid.sum(dim=1, dtype=torch.int64)
    x0 = torch.where(valid, x, torch.zeros_like(x))
    y0 = torch.where(valid, y, torch.zeros_like(y))
    count_float = count.to(torch.float64)
    sum_x = x0.sum(dim=1, dtype=torch.float64)
    sum_y = y0.sum(dim=1, dtype=torch.float64)
    sum_x2 = (x0 * x0).sum(dim=1, dtype=torch.float64)
    sum_y2 = (y0 * y0).sum(dim=1, dtype=torch.float64)
    sum_xy = (x0 * y0).sum(dim=1, dtype=torch.float64)
    safe_count = torch.clamp(count_float, min=1.0)
    covariance = sum_xy - sum_x * sum_y / safe_count
    variance_x = sum_x2 - sum_x * sum_x / safe_count
    variance_y = sum_y2 - sum_y * sum_y / safe_count
    denominator = torch.sqrt(torch.clamp(variance_x, min=0.0) * torch.clamp(variance_y, min=0.0))
    correlation = torch.full(count.shape, torch.nan, device=x.device, dtype=torch.float64)
    usable = (count >= minimum_count) & (variance_x > 0) & (variance_y > 0) & (denominator > 0)
    correlation[usable] = covariance[usable] / denominator[usable]
    return correlation, count


def _masked_bin_mean(target, finite_target, bin_mask):
    import torch
    assigned_count = bin_mask.sum(dim=1, dtype=torch.int64)
    selected = finite_target & bin_mask[None, :, :]
    valid_count = selected.sum(dim=2, dtype=torch.int64)
    total = torch.where(selected, target, torch.zeros_like(target)).sum(dim=2, dtype=torch.float64)
    mean = torch.full(valid_count.shape, torch.nan, device=target.device, dtype=torch.float64)
    nonempty = valid_count > 0
    mean[nonempty] = total[nonempty] / valid_count[nonempty].to(torch.float64)
    coverage = torch.full_like(mean, torch.nan)
    assigned = assigned_count[None, :].expand_as(valid_count)
    has_assignment = assigned > 0
    coverage[has_assignment] = valid_count[has_assignment].to(torch.float64) / assigned[has_assignment].to(torch.float64)
    return mean, valid_count, coverage, assigned_count


def _feature_target_daily_cuda(*, feature_rank, deciles, quintiles, target_block, minimum_ic_symbols: int, minimum_valid_extreme: int, minimum_bin_coverage: float, minimum_middle_coverage: float, minimum_quintile_extreme: int):
    import torch
    targets, dates, securities = target_block.shape
    finite_target = torch.isfinite(target_block)
    finite_feature = torch.isfinite(feature_rank)
    pair_valid = finite_target & finite_feature[None, :, :]
    flat_valid = pair_valid.reshape(targets * dates, securities)
    feature_expanded = feature_rank[None, :, :].expand(targets, dates, securities).reshape(targets * dates, securities)
    target_flat = target_block.reshape(targets * dates, securities)
    exact_feature_rank, feature_distinct = _masked_average_rank_2d(feature_expanded, flat_valid)
    exact_target_rank, target_distinct = _masked_average_rank_2d(target_flat, flat_valid)
    rank_ic, pair_count = _rowwise_pearson(exact_feature_rank, exact_target_rank, flat_valid, minimum_ic_symbols)
    rank_ic = rank_ic.reshape(targets, dates)
    pair_count = pair_count.reshape(targets, dates)
    feature_distinct = feature_distinct.reshape(targets, dates)
    target_distinct = target_distinct.reshape(targets, dates)
    rank_ic[(feature_distinct < 2) | (target_distinct < 2)] = torch.nan
    top_decile = deciles == 9
    bottom_decile = deciles == 0
    middle_decile = (deciles >= 1) & (deciles <= 8)
    top_mean, top_count, top_coverage, _ = _masked_bin_mean(target_block, finite_target, top_decile)
    bottom_mean, bottom_count, bottom_coverage, _ = _masked_bin_mean(target_block, finite_target, bottom_decile)
    middle_mean, middle_count, middle_coverage, _ = _masked_bin_mean(target_block, finite_target, middle_decile)
    top_minus_bottom = top_mean - bottom_mean
    top_minus_middle = top_mean - middle_mean
    middle_minus_bottom = middle_mean - bottom_mean
    top_bottom_valid = (top_count >= minimum_valid_extreme) & (bottom_count >= minimum_valid_extreme) & (top_coverage >= minimum_bin_coverage) & (bottom_coverage >= minimum_bin_coverage)
    top_middle_valid = (top_count >= minimum_valid_extreme) & (middle_count >= minimum_valid_extreme) & (top_coverage >= minimum_bin_coverage) & (middle_coverage >= minimum_middle_coverage)
    middle_bottom_valid = (middle_count >= minimum_valid_extreme) & (bottom_count >= minimum_valid_extreme) & (middle_coverage >= minimum_middle_coverage) & (bottom_coverage >= minimum_bin_coverage)
    top_minus_bottom[~top_bottom_valid] = torch.nan
    top_minus_middle[~top_middle_valid] = torch.nan
    middle_minus_bottom[~middle_bottom_valid] = torch.nan
    top_quintile = quintiles == 4
    bottom_quintile = quintiles == 0
    qtop_mean, qtop_count, qtop_coverage, _ = _masked_bin_mean(target_block, finite_target, top_quintile)
    qbottom_mean, qbottom_count, qbottom_coverage, _ = _masked_bin_mean(target_block, finite_target, bottom_quintile)
    quintile_spread = qtop_mean - qbottom_mean
    quintile_valid = (qtop_count >= minimum_quintile_extreme) & (qbottom_count >= minimum_quintile_extreme) & (qtop_coverage >= minimum_bin_coverage) & (qbottom_coverage >= minimum_bin_coverage)
    quintile_spread[~quintile_valid] = torch.nan
    target_coverage = torch.minimum(top_coverage, bottom_coverage)
    distinct_symbols = pair_valid.any(dim=1).sum(dim=1, dtype=torch.int64)
    return {"rank_ic": rank_ic, "top_minus_bottom": top_minus_bottom, "top_minus_middle": top_minus_middle, "middle_minus_bottom": middle_minus_bottom, "quintile_spread": quintile_spread, "target_coverage": target_coverage, "ic_cross_section_size": pair_count, "top_coverage": top_coverage, "bottom_coverage": bottom_coverage, "middle_coverage": middle_coverage, "quintile_top_coverage": qtop_coverage, "quintile_bottom_coverage": qbottom_coverage, "distinct_symbols": distinct_symbols}


def scan_feature_target_block_cuda(*, feature_ids: list[int], target_ids: list[int], rank_cache, target_values: np.ndarray, feature_specs, target_specs, config, retain_daily: bool = False):
    from .scan import summarize_pair
    torch, device = require_cuda(config.cuda_device, config.gpu_minimum_free_memory_bytes)
    torch.cuda.reset_peak_memory_stats(device)
    target_tensor = torch.as_tensor(np.asarray(target_values[target_ids], dtype=np.float32), device=device, dtype=torch.float32)
    rows: list[dict] = []
    retained: dict[tuple[int, int], DailyPairSeries] = {}
    for feature_id in feature_ids:
        feature_rank = torch.as_tensor(np.asarray(rank_cache.percentile_ranks[feature_id], dtype=np.float32), device=device, dtype=torch.float32)
        deciles = torch.as_tensor(np.asarray(rank_cache.deciles[feature_id], dtype=np.int8), device=device, dtype=torch.int8)
        quintiles = torch.as_tensor(np.asarray(rank_cache.quintiles[feature_id], dtype=np.int8), device=device, dtype=torch.int8)
        daily_gpu = _feature_target_daily_cuda(feature_rank=feature_rank, deciles=deciles, quintiles=quintiles, target_block=target_tensor, minimum_ic_symbols=config.minimum_rank_ic_cross_section_size, minimum_valid_extreme=config.minimum_valid_outcomes_per_extreme_decile, minimum_bin_coverage=config.minimum_target_coverage_fraction_per_bin, minimum_middle_coverage=config.minimum_middle_target_coverage_fraction, minimum_quintile_extreme=config.minimum_valid_outcomes_per_extreme_quintile)
        cpu = {key: value.detach().cpu().numpy() for key, value in daily_gpu.items()}
        for local_target, target_id in enumerate(target_ids):
            daily = DailyPairSeries(rank_ic=cpu["rank_ic"][local_target], top_minus_bottom=cpu["top_minus_bottom"][local_target], top_minus_middle=cpu["top_minus_middle"][local_target], middle_minus_bottom=cpu["middle_minus_bottom"][local_target], quintile_spread=cpu["quintile_spread"][local_target], target_coverage=cpu["target_coverage"][local_target], ic_cross_section_size=cpu["ic_cross_section_size"][local_target], top_coverage=cpu["top_coverage"][local_target], bottom_coverage=cpu["bottom_coverage"][local_target], middle_coverage=cpu["middle_coverage"][local_target], quintile_top_coverage=cpu["quintile_top_coverage"][local_target], quintile_bottom_coverage=cpu["quintile_bottom_coverage"][local_target])
            pair_rows = summarize_pair(daily, feature_spec=feature_specs[feature_id], target_spec=target_specs[target_id], distinct_symbols=int(cpu["distinct_symbols"][local_target]))
            for row in pair_rows: row["backend"] = "cuda_exact_cross_sectional"
            rows.extend(pair_rows)
            if retain_daily: retained[(feature_id, target_id)] = daily
    torch.cuda.synchronize(device)
    diagnostics = CudaScanDiagnostics(backend="cuda_exact_cross_sectional", device_name=torch.cuda.get_device_name(device), feature_count=len(feature_ids), target_count=len(target_ids), date_count=int(target_tensor.shape[1]), security_count=int(target_tensor.shape[2]), peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)), peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)))
    return rows, retained, diagnostics
