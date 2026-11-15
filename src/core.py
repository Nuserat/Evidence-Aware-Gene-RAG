from __future__ import annotations
import csv, hashlib, json, re
from pathlib import Path
from typing import Any
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import DATA, API_KEY, MODEL, MIN_EVIDENCE, MIN_SCORE

class Record(BaseModel):
    id: str
    text: str
    subject: str = ""
    subject_type: str = "UNKNOWN"
    subject_id: str = ""
    predicate: str = ""
    object: str = ""
    object_type: str = "UNKNOWN"
    object_id: str = ""
    source: str = "uploaded"
    source_id: str = ""
    pmid: str = ""
    doi: str = ""
    url: str = ""
    evidence_type: str = "literature"
    dataset_version: str = ""
    publication_year: int | None = None
    evidence_polarity: str = "NOT_ASSESSED"
    confidence: float | None = None
    statistical_significance: str = ""
    effect_direction: str = ""
    effect_size: str = ""
    gene: str = ""
    chemical: str = ""
    relation: str = ""
    entity_1: str = ""
    entity_2: str = ""
    split: str = ""
    original_record: dict = Field(default_factory=dict)

ALIASES = {
"text": ["text","evidence_text","sentence","sentence_text","abstract","passage","description","statement","context","document_text"],
"subject": ["subject","subject_name","subject_text","entity_1","entity1","head","head_text","source_entity","entity1_name","entity1name","variant/haplotypes","variant","variant_name","variant_id"],
"subject_type": ["subject_type","entity_1_type","entity1_type","head_type","entity1type"],
"subject_id": ["subject_id","entity_1_id","entity1_id","head_id","entity1_id"],
"predicate": ["predicate","relation","relation_type","interaction","label","cpr","relation_label","association"],
"object": ["object","object_name","object_text","entity_2","entity2","tail","tail_text","target_entity","entity2_name","entity2name","drug(s)","drug","chemical","phenotype","phenotype_category"],
"object_type": ["object_type","entity_2_type","entity2_type","tail_type","entity2type"],
"object_id": ["object_id","entity_2_id","entity2_id","tail_id"],
"gene": ["gene","gene_name","gene_symbol"],
"chemical": ["chemical","chemical_name","drug","drug_name","compound","drugbank_name"],
"pmid": ["pmid","pubmed_id","pubmedid"],
"source": ["source","dataset","database","corpus"],
"source_id": ["source_id","record_id","document_id","doc_id","id","annotation_id"],
"split": ["split","partition"],
"evidence_type": ["evidence_type","record_type","annotation_type","evidence"],
"url": ["url","link","source_url"],
"doi": ["doi","digital_object_identifier"],
"confidence": ["confidence","confidence_score","probability","score"],
"evidence_polarity": ["evidence_polarity","polarity","finding","direction_of_association"],
"publication_year": ["publication_year","year","pub_year"],
"dataset_version": ["dataset_version","version"],
"statistical_significance": ["statistical_significance","significance","p_value","pvalue"],
"effect_direction": ["effect_direction","direction","direction_of_effect"],
"effect_size": ["effect_size","odds_ratio","hazard_ratio"],
}

def _clean(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, (list, tuple)): return "; ".join(map(str, v))
    return re.sub(r"\s+", " ", str(v)).strip()

def val(row: dict, key: str) -> str:
    low = {str(k).strip().lower(): v for k,v in row.items()}
    for alias in ALIASES.get(key, [key]):
        if alias.lower() in low and low[alias.lower()] not in (None, ""):
            return _clean(low[alias.lower()])
    return ""

def infer_type(name: str, explicit: str = "") -> str:
    t = re.sub(r"[^A-Z]", "", (explicit or "").upper())
    mapping = {"GENE":"GENE","VARIANT":"VARIANT","SNP":"VARIANT","DRUG":"DRUG","CHEMICAL":"CHEMICAL","PROTEIN":"PROTEIN","DISEASE":"DISEASE","PHENOTYPE":"PHENOTYPE","PATHWAY":"PATHWAY"}
    if t in mapping: return mapping[t]
    n = (name or "").strip()
    if re.match(r"^rs\d+$", n, re.I): return "VARIANT"
    return "UNKNOWN"

def convert(row: dict, fallback: str = "uploaded", file_name: str = "") -> Record | None:
    text = val(row,"text")
    subject, obj, predicate = val(row,"subject"), val(row,"object"), val(row,"predicate")
    gene, chemical = val(row,"gene"), val(row,"chemical")
    # Explicit table-specific mappings for common PharmGKB annotation exports.
    # These relation labels describe the annotation table, not causal effects.
    table = (file_name or fallback).casefold()
    if "var_drug_ann" in table:
        subject = subject or val(row,"subject")
        obj = obj or val(row,"object")
        predicate = predicate or "VARIANT_ANNOTATED_FOR_DRUG_RESPONSE"
        if not val(row,"subject_type"): row = {**row, "subject_type": "VARIANT"}
        if not val(row,"object_type"): row = {**row, "object_type": "DRUG"}
    elif "var_fa_ann" in table:
        subject = subject or val(row,"subject")
        obj = obj or gene
        predicate = predicate or "VARIANT_HAS_FUNCTIONAL_ANNOTATION_FOR_GENE"
        if not val(row,"subject_type"): row = {**row, "subject_type": "VARIANT"}
        if not val(row,"object_type"): row = {**row, "object_type": "GENE"}
    elif "var_pheno_ann" in table:
        subject = subject or val(row,"subject")
        obj = obj or val(row,"object")
        predicate = predicate or "VARIANT_ANNOTATED_WITH_PHENOTYPE"
        if not val(row,"subject_type"): row = {**row, "subject_type": "VARIANT"}
        if not val(row,"object_type"): row = {**row, "object_type": "PHENOTYPE"}
    if not subject and gene: subject = gene
    if not obj and chemical: obj = chemical
    if not text:
        text = " ".join(x for x in (subject,predicate,obj) if x)
    if not text: return None
    source = val(row,"source") or fallback or Path(file_name).stem
    source_id = val(row,"source_id")
    stable = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    source_id = source_id or hashlib.sha256((source+"|"+stable).encode()).hexdigest()[:20]
    rid = f"{source}:{source_id}"
    conf = None
    raw_conf = val(row,"confidence")
    try:
        conf = float(raw_conf) if raw_conf else None
        if conf is not None and not 0 <= conf <= 1: conf = None
    except ValueError: pass
    year = None
    try: year = int(val(row,"publication_year")) if val(row,"publication_year") else None
    except ValueError: pass
    polarity_raw = val(row,"evidence_polarity").upper()
    polarity_map = {"POSITIVE":"POSITIVE","SUPPORTS":"POSITIVE","ASSOCIATED":"POSITIVE","NEGATIVE":"NEGATIVE","CONTRADICTS":"NEGATIVE","UNCERTAIN":"UNCERTAIN","MIXED":"UNCERTAIN"}
    polarity = polarity_map.get(polarity_raw, "NOT_ASSESSED")
    s_type = infer_type(subject, val(row,"subject_type"))
    o_type = infer_type(obj, val(row,"object_type"))
    return Record(
        id=rid, text=text, subject=subject, subject_type=s_type,
        subject_id=val(row,"subject_id"), predicate=predicate,
        object=obj, object_type=o_type, object_id=val(row,"object_id"),
        source=source, source_id=source_id, pmid=val(row,"pmid"),
        doi=val(row,"doi"), url=val(row,"url"),
        evidence_type=val(row,"evidence_type") or "unspecified",
        dataset_version=val(row,"dataset_version"), publication_year=year,
        evidence_polarity=polarity, confidence=conf,
        statistical_significance=val(row,"statistical_significance"),
        effect_direction=val(row,"effect_direction"), effect_size=val(row,"effect_size"),
        gene=gene or (subject if s_type=="GENE" else ""),
        chemical=chemical or (obj if o_type in {"DRUG","CHEMICAL"} else ""),
        relation=predicate, entity_1=subject, entity_2=obj,
        split=val(row,"split"), original_record=row
    )

def read_records(path, source=None):
    p=Path(path); ext=p.suffix.lower()
    if ext in {".csv",".tsv"}:
        with p.open(encoding="utf-8-sig", newline="") as f:
            rows=list(csv.DictReader(f, delimiter="\t" if ext==".tsv" else ","))
    elif ext==".jsonl":
        rows=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    elif ext==".json":
        obj=json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj,list): rows=obj
        elif isinstance(obj,dict):
            rows=next((obj[k] for k in ("records","data","documents","annotations","examples") if isinstance(obj.get(k),list)),[obj])
        else: rows=[]
    elif ext==".txt":
        rows=[{"text":x,"source_id":str(i)} for i,x in enumerate(p.read_text(encoding="utf-8").split("\n\n")) if x.strip()]
    else: raise ValueError(f"Unsupported file: {p.name}")
    label=source or p.stem
    out=[]
    for row in rows:
        if isinstance(row,dict):
            rec=convert(row,label,p.name)
            if rec: out.append(rec)
    return out

def load():
    if not DATA.exists(): return []
    records=[]
    for line in DATA.read_text(encoding="utf-8").splitlines():
        try:
            row=json.loads(line)
            if isinstance(row,dict):
                try: records.append(Record.model_validate(row))
                except Exception:
                    rec=convert(row,"legacy")
                    if rec: records.append(rec)
        except Exception: continue
    return deduplicate(records)

def deduplicate(records):
    seen=set(); out=[]
    for r in records:
        key=(r.source.casefold(),r.source_id or r.id, r.pmid, r.subject.casefold(),r.predicate.casefold(),r.object.casefold(),hashlib.sha1(r.text.encode()).hexdigest())
        if key not in seen: seen.add(key); out.append(r)
    return out

def save(records):
    DATA.parent.mkdir(parents=True,exist_ok=True)
    clean=deduplicate(records)
    DATA.write_text("".join(x.model_dump_json()+"\n" for x in clean),encoding="utf-8")
    return len(clean)

def ingest(path, source=None):
    new=read_records(path,source)
    return len(new), save(load()+new)

def _tokens(s): return re.findall(r"[a-zA-Z0-9_*:-]+", (s or "").casefold())

def search(query, k=8):
    records=load()
    if not records:
        return [], {"abstained":True,"answer":"No evidence has been indexed.","used_evidence":[]}
    docs=[]
    for r in records:
        docs.append(" ".join([r.text,r.subject,r.predicate,r.object,r.gene,r.chemical,r.pmid,r.source]))
    token_docs=[_tokens(d) or ["empty"] for d in docs]
    qtokens=_tokens(query)
    if not qtokens:
        return [], {"abstained":True,"answer":"Please enter a biomedical research question.","used_evidence":[]}
    bm=np.asarray(BM25Okapi(token_docs).get_scores(qtokens),dtype=float)
    if bm.max()>0: bm=bm/bm.max()
    try:
        vec=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True, min_df=1)
        matrix=vec.fit_transform(docs)
        sem=cosine_similarity(vec.transform([query]),matrix).ravel()
        if sem.max()>0: sem=sem/sem.max()
    except ValueError: sem=np.zeros(len(records))
    qnorm=query.casefold()
    # Structured entity match: exact normalized entity mention, not claimed as graph traversal.
    structured=np.array([sum(1 for x in (r.subject,r.object,r.gene,r.chemical) if x and x.casefold() in qnorm) for r in records],dtype=float)
    if structured.max()>0: structured=structured/structured.max()
    score=.40*bm+.35*sem+.25*structured
    order=np.argsort(score)[::-1]
    hits=[]; seen=set()
    for i in order:
        r=records[int(i)]
        assertion=(r.subject.casefold(),r.predicate.casefold(),r.object.casefold(),r.source.casefold(),r.pmid)
        if assertion in seen: continue
        seen.add(assertion)
        hits.append({"record":r,"score":float(score[i]),"bm25":float(bm[i]),"semantic_tfidf":float(sem[i]),"entity_match":float(structured[i])})
        if len(hits)>=max(1,int(k)): break
    chosen=[h for h in hits if h["score"]>=MIN_SCORE]
    if len(chosen)<MIN_EVIDENCE:
        return hits, {"abstained":True,"answer":"Insufficient evidence above the configured retrieval threshold. Treat retrieved records as leads, not verified conclusions.","used_evidence":[]}
    # Conservative answer: LLM is asked to synthesize only supplied evidence and cite record IDs.
    context="\n\n".join(f"[{h['record'].id}] {h['record'].text}\nTriple: {h['record'].subject} --{h['record'].predicate}--> {h['record'].object}\nSource: {h['record'].source}; PMID: {h['record'].pmid or 'not supplied'}" for h in chosen[:8])
    answer="Retrieved evidence is listed below. Review the source records before drawing a biological conclusion."
    if API_KEY:
        try:
            from openai import OpenAI
            client=OpenAI(api_key=API_KEY)
            response=client.chat.completions.create(
                model=MODEL,temperature=0,
                messages=[
                    {"role":"system","content":"You are a cautious biomedical research assistant, not a clinician. Answer only from the supplied records. Distinguish reported associations from causal claims. Do not infer missing evidence or treat chemical-protein interactions as clinical recommendations. Cite every factual claim with exact record IDs in square brackets. If records do not answer the question, say so. Explicitly mention when PMID or source metadata is missing."},
                    {"role":"user","content":f"Question: {query}\n\nEvidence records:\n{context}\n\nGive a concise answer and a short evidence table."}
                ])
            answer=response.choices[0].message.content or answer
        except Exception as e:
            answer += f"\nLLM synthesis unavailable ({type(e).__name__}); inspect the retrieved records."
    return hits, {"abstained":False,"answer":answer,"used_evidence":[h["record"].id for h in chosen[:8]]}
