from pathlib import Path
from src.core import read_records, save
from src.config import DATASET, DATA
import argparse

SUPPORTED={".csv",".tsv",".json",".jsonl",".txt"}
EXCLUDE={"processed","uploads","__pycache__"}
def main():
    ap=argparse.ArgumentParser(description="Import supported biomedical dataset exports into the evidence index.")
    ap.add_argument("--dataset-dir",default=str(DATASET))
    ap.add_argument("--replace",action="store_true",help="Replace the processed index instead of appending.")
    args=ap.parse_args()
    folder=Path(args.dataset_dir)
    if not folder.exists(): raise SystemExit(f"Dataset directory not found: {folder}")
    files=[p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED and not any(x in EXCLUDE for x in p.parts)]
    all_records=[]; summary=[]
    for p in sorted(files):
        try:
            records=read_records(p,source=p.stem)
            all_records.extend(records)
            summary.append((p.name,len(records),"OK"))
        except Exception as e:
            summary.append((p.name,0,f"ERROR: {e}"))
    if args.replace: existing=[]
    else:
        from src.core import load
        existing=load()
    total=save(existing+all_records)
    print("Import summary")
    for name,n,status in summary: print(f"{name:45} {n:8} {status}")
    print(f"\nNew records parsed: {len(all_records)}")
    print(f"Unique records indexed: {total}")
    print(f"Index: {DATA}")
    print("Note: parsing success does not establish biological validity; review source mappings and record samples.")
if __name__=="__main__": main()
