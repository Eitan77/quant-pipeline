
from __future__ import annotations

import argparse
from pathlib import Path

from quant_pipeline.interday.config import InterdayConfig
from quant_pipeline.interday.run import execute_interday_2a


STAGES = (
    "schema-check",
    "source",
    "panel",
    "features",
    "targets",
    "ranks",
    "scan",
    "finalize",
    "diagnostics",
    "report",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant-pipeline-interday-2a",
        description="Run sealed-holdout Interday 2A discovery.",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Invalidate the selected stage and its downstream dependents.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = InterdayConfig.from_yaml(args.config)
    output = execute_interday_2a(
        config,
        stage=args.stage,
        force_rebuild=args.force_rebuild,

    )
    print(output)


if __name__ == "__main__":
    main()
