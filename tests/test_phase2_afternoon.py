from quant_pipeline.stage2_afternoon import AfternoonConfig, afternoon_specs


def test_afternoon_grid_is_bounded_and_unique() -> None:
    specs = afternoon_specs(AfternoonConfig("phase1", "phase2"))
    assert len(specs) == 50
    assert len({row[0].strategy_id for row in specs}) == 50
    assert sum(row[0].direction == "adaptive" for row in specs) == 2
