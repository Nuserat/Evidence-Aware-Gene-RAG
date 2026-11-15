import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/"dataset"
DATA=DATASET/"processed"/"evidence.jsonl"
UPLOADS=DATASET/"uploads"
DATA.parent.mkdir(parents=True,exist_ok=True)
UPLOADS.mkdir(parents=True,exist_ok=True)
API_KEY=os.getenv("OPENAI_API_KEY","").strip()
MODEL=os.getenv("OPENAI_MODEL","gpt-4o-mini")
MIN_EVIDENCE=int(os.getenv("MIN_EVIDENCE","2"))
MIN_SCORE=float(os.getenv("MIN_HYBRID_SCORE","0.05"))
TOP_K=int(os.getenv("TOP_K","8"))