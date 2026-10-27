
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI,UploadFile,File,HTTPException
from pydantic import BaseModel
from src.core import search,ingest,load
from src.config import UPLOADS
app=FastAPI(title="PharmaGraph-RAG API",version="0.1")
class Q(BaseModel):
 question:str
 top_k:int=8
@app.get("/")
def home():return {"status":"running","warning":"Research prototype; not clinical advice"}
@app.post("/query")
def ask(q:Q):
 hits,res=search(q.question,q.top_k)
 return {"answer":res["answer"],"abstained":res["abstained"],"indexed_records":len(load()),"used_evidence":res["used_evidence"],"hits":[{"id":h["record"].id,"text":h["record"].text,"score":h["score"],"source":h["record"].source} for h in hits]}
@app.post("/upload")
async def upload(file:UploadFile=File(...),source:str="api_upload"):
 p=Path(file.filename)
 if p.suffix.lower() not in {".csv",".tsv",".json",".jsonl",".txt"}:raise HTTPException(400,"Unsupported file type")
 target=UPLOADS/p.name;target.write_bytes(await file.read())
 try:n,total=ingest(target,source)
 except Exception as e:raise HTTPException(400,str(e))
 return {"imported":n,"total":total}
