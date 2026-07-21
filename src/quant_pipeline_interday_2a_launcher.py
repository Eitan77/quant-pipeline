from __future__ import annotations
import argparse
from pathlib import Path
from quant_pipeline.interday.config import InterdayConfig
from quant_pipeline.interday.run import execute_interday_2a

def build_parser():
    p=argparse.ArgumentParser(prog="quant-pipeline-interday-2a",description="Run sealed-holdout Interday 2A discovery")
    p.add_argument("config",type=Path); p.add_argument("--stage",choices=("schema-check","all","panel","features","targets","ranks","scan","finalize","diagnostics","report"),default="all"); p.add_argument("--force-rebuild",action="store_true"); return p

def main():
    args=build_parser().parse_args(); print(execute_interday_2a(InterdayConfig.from_yaml(args.config),stage=args.stage,force_rebuild=args.force_rebuild))

if __name__=="__main__": main()
