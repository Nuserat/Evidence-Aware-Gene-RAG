import csv,json,re,hashlib
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from .config import DATA,API_KEY,MODEL,MIN_EVIDENCE,MIN_SCORE

class Record(BaseModel):
    id:str
    text:str
    gene:str=""
    chemical:str=""
    relation:str=""
    pmid:str=""
    source:str="uploaded"
    source_id:str=""
    split:str=""
    evidence_type:str="literature"
    url:str=""

ALIASES={
"text":["text","sentence","abstract","passage","evidence","description"],
"gene":["gene","gene_name","protein","protein_name","target"],
"chemical":["chemical","chemical_name","drug","drug_name","compound"],
"relation":["relation","relation_type","label","interaction","cpr"],
"pmid":["pmid","pubmed_id"],"source":["source","dataset","database","corpus"],
"source_id":["source_id","record_id","document_id","doc_id"],
"split":["split","partition"],"evidence_type":["evidence_type","record_type"],
"url":["url","link"]
}
def val(row,key):
    d={str(k).lower():v for k,v in row.items()}
    for alias in ALIASES[key]:
        if alias in d and d[alias] is not None: return re.sub(r"\s+"," ",str(d[alias])).strip()
    return ""
def convert(row,fallback):
    text=val(row,"text")
    if not text:return None
    gene,chem,rel=val(row,"gene"),val(row,"chemical"),val(row,"relation")
    src=val(row,"source") or fallback
    sid=val(row,"source_id") or hashlib.sha1((text+gene+chem+rel+src).encode()).hexdigest()[:16]
    rid=str(row.get("id") or f"{src}:{sid}")
    return Record(id=rid,text=text,gene=gene,chemical=chem,relation=rel,pmid=val(row,"pmid"),
      source=src,source_id=sid,split=val(row,"split"),
      evidence_type=val(row,"evidence_type") or "literature",url=val(row,"url"))
def read_records(path,source):
    p=Path(path); ext=p.suffix.lower()
    if ext in (".csv",".tsv"):
        with p.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f,delimiter="\t" if ext==".tsv" else ","))
    elif ext==".jsonl":
        rows=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    elif ext==".json":
        obj=json.loads(p.read_text(encoding="utf-8"))
        rows=obj if isinstance(obj,list) else next((obj[k] for k in ("records","data","documents","annotations") if isinstance(obj.get(k),list)),[obj])
    elif ext==".txt":
        rows=[{"text":x,"source_id":str(i)} for i,x in enumerate(p.read_text(encoding="utf-8").split("\n\n")) if x.strip()]
    else: raise ValueError("Supported: csv, tsv, json, jsonl, txt")
    return [r for row in rows if isinstance(row,dict) if (r:=convert(row,source))]
def load():
    if not DATA.exists():return []
    return [Record.model_validate_json(x) for x in DATA.read_text(encoding="utf-8").splitlines() if x.strip()]
def save(records):
    DATA.parent.mkdir(parents=True,exist_ok=True)
    seen=set(); clean=[]
    for r in records:
        key=(r.text.casefold(),r.gene.casefold(),r.chemical.casefold(),r.relation.casefold())
        if key not in seen:seen.add(key);clean.append(r)
    DATA.write_text("".join(x.model_dump_json()+"\n" for x in clean),encoding="utf-8")
    return len(clean)
def ingest(path,source):
    new=read_records(path,source)
    return len(new),save(load()+new)
def search(query,k=8):
    rs=load()
    if not rs:return [],{"abstained":True,"answer":"No evidence records indexed yet.","used_evidence":[]}
    docs=[" ".join([r.text,r.gene,r.chemical,r.relation,r.pmid,r.source]) for r in rs]
    tokens=[re.findall(r"[a-zA-Z0-9_*:-]+",x.lower()) for x in docs]
    qtok=re.findall(r"[a-zA-Z0-9_*:-]+",query.lower())
    bm=np.array(BM25Okapi(tokens).get_scores(qtok),dtype=float)
    bm=bm/bm.max() if bm.max()>0 else bm
    vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)
    mat=vec.fit_transform(docs)
    sem=cosine_similarity(vec.transform([query]),mat).ravel()
    sem=sem/sem.max() if sem.max()>0 else sem
    q=query.lower()
    graph=np.array([sum(1 for x in (r.gene,r.chemical,r.relation) if x and x.lower() in q) for r in rs],dtype=float)
    graph=graph/graph.max() if graph.max()>0 else graph
    score=.4*bm+.35*sem+.25*graph
    order=np.argsort(score)[::-1][:k]
    hits=[{"record":rs[i],"score":float(score[i]),"bm25":float(bm[i]),"tfidf":float(sem[i]),"graph":float(graph[i])} for i in order]
    chosen=[h for h in hits if h["score"]>=MIN_SCORE]
    if len(chosen)<MIN_EVIDENCE:
        return hits,{"abstained":True,"answer":"Insufficient indexed evidence above the configured threshold. Add relevant source records or narrow the research question.","used_evidence":[]}
    ctx="\n\n".join(f'[{h["record"].id}] {h["record"].text} (gene={h["record"].gene}; chemical={h["record"].chemical}; relation={h["record"].relation}; source={h["record"].source}; PMID={h["record"].pmid})' for h in chosen[:6])
    if API_KEY:
        try:
            from openai import OpenAI
            c=OpenAI(api_key=API_KEY)
            out=c.chat.completions.create(model=MODEL,temperature=0,messages=[
             {"role":"system","content":"You are a biomedical literature research assistant. Use only supplied evidence. Do not provide clinical advice. Distinguish chemical-protein benchmark relations from clinical pharmacogenomic evidence. Cite every factual claim using exact evidence IDs in square brackets. State uncertainty and conflicts. Never invent sources."},
             {"role":"user","content":f"Question: {query}\nEvidence:\n{ctx}\nWrite a cautious evidence synthesis."}])
            ans=out.choices[0].message.content or ""
            ids=[h["record"].id for h in chosen[:6]]
            if not any(i in ans for i in ids):return hits,{"abstained":True,"answer":"Generated response lacked a verifiable evidence ID. Inspect retrieved records instead.","used_evidence":[]}
            return hits,{"abstained":False,"answer":ans,"used_evidence":ids}
        except Exception:pass
    return hits,{"abstained":False,"answer":"Extractive evidence summary (LLM not configured):\n\n"+ctx+"\n\nVerify original records before interpretation.","used_evidence":[h["record"].id for h in chosen[:6]]}
