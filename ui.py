from pathlib import Path
import json
import pandas as pd
import streamlit as st
from src.evidence_rag import EvidenceAwareRAG

st.set_page_config(page_title="Evidence-Aware Gene RAG", page_icon="🧬", layout="wide")
st.title("🧬 Evidence-Aware Gene RAG")
st.caption("Research prototype | TF-IDF retrieval | Optional Hugging Face generation and NLI verification")
st.warning("Research use only; not a clinical decision-support system. Verify biomedical claims against primary sources.")

def read_records(upload):
    name, raw = upload.name.lower(), upload.getvalue()
    if name.endswith((".jsonl", ".ndjson")):
        records = []
        for n, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if line.strip():
                try: item = json.loads(line)
                except json.JSONDecodeError as e: raise ValueError(f"Invalid JSONL line {n}: {e}") from e
                if isinstance(item, dict): records.append(item)
        return records
    if name.endswith(".json"):
        obj = json.loads(raw.decode("utf-8-sig"))
        if isinstance(obj, list): return obj
        if isinstance(obj, dict):
            for k in ("data", "documents", "corpus", "records", "examples"):
                if isinstance(obj.get(k), list): return obj[k]
            return [obj]
    if name.endswith(".tsv"): return pd.read_csv(upload, sep="\\t").fillna("").to_dict("records")
    if name.endswith(".csv"): return pd.read_csv(upload).fillna("").to_dict("records")
    if name.endswith(".txt"):
        return [{"id": upload.name, "title": upload.name, "text": raw.decode("utf-8-sig"), "source": upload.name, "url": ""}]
    raise ValueError("Unsupported file type. Use JSON, JSONL, TSV, CSV, or TXT.")

def normalize(records):
    docs = []
    for i, r in enumerate(records):
        if not isinstance(r, dict): continue
        text = next((r[k] for k in ("text", "abstract", "passage", "context", "content", "answer") if r.get(k)), "")
        if not text: text = " | ".join(f"{k}: {v}" for k, v in r.items() if str(v).strip())
        docs.append({
            "id": str(r.get("id", r.get("doc_id", r.get("pmid", f"doc_{i+1}")))),
            "title": str(r.get("title", r.get("document_title", r.get("question", "")))) ,
            "text": str(text),
            "source": str(r.get("source", r.get("dataset", r.get("journal", "uploaded_dataset")))),
            "url": str(r.get("url", r.get("source_url", r.get("link", "")))) ,
        })
    if not docs: raise ValueError("No usable records found in uploaded file.")
    return docs

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Evidence corpus", type=["json", "jsonl", "ndjson", "tsv", "csv", "txt"])
    top_k = st.slider("Top-k evidence", 1, 10, 5)
    min_score = st.slider("Minimum retrieval score", 0.0, 1.0, 0.08, 0.01)
    gen_model = st.text_input("Generator model (optional)", placeholder="Qwen/Qwen2.5-3B-Instruct")
    nli_model = st.text_input("NLI model (optional)", placeholder="cross-encoder/nli-deberta-v3-base")
    min_entail = st.slider("Minimum entailment", 0.0, 1.0, 0.50, 0.05)
    max_contra = st.slider("Maximum contradiction", 0.0, 1.0, 0.40, 0.05)

if uploaded:
    try:
        documents = normalize(read_records(uploaded))
        corpus_name = uploaded.name
    except Exception as e:
        st.error(f"Could not read corpus: {e}")
        st.stop()
else:
    demo = Path("data/example_corpus.jsonl")
    if not demo.exists():
        st.info("Upload a corpus file from the sidebar to begin.")
        st.stop()
    with demo.open(encoding="utf-8") as f:
        documents = [json.loads(line) for line in f if line.strip()]
    corpus_name = str(demo)

with st.expander(f"Corpus preview — {len(documents)} records ({corpus_name})"):
    st.dataframe(pd.DataFrame(documents)[["id", "title", "source", "text"]].head(25), use_container_width=True)

question = st.text_area("Biomedical question", placeholder="Ask a question answerable from the uploaded evidence...")
if st.button("Retrieve evidence and answer", type="primary", use_container_width=True):
    if not question.strip():
        st.error("Please enter a question.")
        st.stop()
    try:
        with st.spinner("Retrieving evidence and processing question..."):
            rag = EvidenceAwareRAG(
                documents, top_k=top_k, min_retrieval_score=min_score,
                generator_model=gen_model.strip() or None,
                nli_model=nli_model.strip() or None,
                min_entailment=min_entail, max_contradiction=max_contra,
            )
            result = rag.answer(question.strip())
        st.subheader("Answer")
        if result.get("abstained"): st.warning("Abstained: " + str(result.get("abstention_reason")))
        else: st.success("Generated answer — inspect supporting evidence.")
        st.write(result.get("answer", "No answer returned."))
        st.subheader("Evidence")
        for e in result.get("evidence", []):
            with st.expander(f"[E{e.get('rank', '?')}] {e.get('title') or e.get('id')} · score {e.get('retrieval_score', 0):.3f}", expanded=True):
                st.write(e.get("text", ""))
                st.caption(f"ID: {e.get('id')} | Source: {e.get('source', 'unknown')}")
                if e.get("url"): st.markdown(f"[Open source]({e['url']})")
        if result.get("verification"):
            st.subheader("NLI verification signal")
            st.json(result["verification"])
            st.caption("NLI is an imperfect research signal, not proof of factual correctness.")
        st.download_button("Download result JSON", json.dumps({"question": question, **result}, ensure_ascii=False, indent=2), "biomedical_rag_result.json", "application/json")
    except Exception as e:
        st.exception(e)

st.divider()
st.caption("Check column mapping in the corpus preview. Dataset-specific preprocessing may be required for gene/protein QA datasets.")
