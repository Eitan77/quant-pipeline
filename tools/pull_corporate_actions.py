from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--symbols-file",required=True)
    parser.add_argument("--start",required=True)
    parser.add_argument("--end",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    load_dotenv(Path(os.environ.get("ALGO_RESEARCH_ENV",r"D:\AlgoResearch\.env")))
    from alpaca_research.client import AlpacaClient
    from alpaca_research.config import load_settings
    symbols=[s.strip().upper() for s in Path(args.symbols_file).read_text().replace("\n",",").split(",") if s.strip()]
    client=AlpacaClient(load_settings()); rows=[]
    for start in range(0,len(symbols),50):
        params={"start":args.start,"end":args.end,"symbols":",".join(symbols[start:start+50]),"limit":1000}
        for page in client.paged_data_get("/v1/corporate-actions",params):
            groups=page.get("corporate_actions") or {}
            if isinstance(groups,list): groups={"unknown":groups}
            for action_type,items in groups.items():
                for item in items or []:
                    row={"action_type":action_type,**item}
                    if action_type in {"forward_splits","reverse_splits","unit_splits"}:
                        old=float(item.get("old_rate") or 1); new=float(item.get("new_rate") or 1)
                        row["split_ratio"]=new/old if old else None
                    rows.append(row)
    frame=pd.DataFrame(rows)
    if frame.empty: raise RuntimeError("Corporate-action request returned no rows")
    for column in ["ex_date","process_date","effective_date"]:
        if column in frame: frame[column]=pd.to_datetime(frame[column],errors="coerce")
    split=frame.loc[frame.split_ratio.notna()&frame.symbol.notna(),["symbol","ex_date","process_date","action_type","split_ratio","id"]].copy()
    split=split.loc[split.ex_date.notna()].sort_values(["symbol","ex_date","id"]).drop_duplicates(["symbol","ex_date","id"])
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); split.to_parquet(output,index=False)
    print(f"wrote {len(split)} split actions to {output}")


if __name__=="__main__": main()
