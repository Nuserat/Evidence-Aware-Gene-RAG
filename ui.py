import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import streamlit as st
import pandas as pd
from src.core import load,ingest,search,save
from src.config import UPLOADS,TOP_K
from src.knowledge_graph import build_knowledge_graph,render_knowledge_graph,detect_evidence_conflicts,export_rdf_turtle

st.set_page_config(page_title="PharmaGraph-RAG",page_icon="🧬",layout="wide")
st.title("🧬 PharmaGraph-RAG")
st.caption("Evidence-aware pharmacogenomic research assistant | Research prototype — not clinical advice")
page=st.sidebar.radio("Navigate",["Research question","Knowledge Graph","Evidence conflicts","Upload data","Dataset overview","Export RDF","Methodology & limitations"])

if page=="Research question":
    st.warning("Research use only. Retrieved associations are not automatically causal or clinical recommendations.")
    q=st.text_area("Biomedical research question",placeholder="e.g. Which drugs are associated with CYP2C19 or rs4244285?")
    k=st.slider("Maximum evidence records",3,20,TOP_K)
    if st.button("Retrieve evidence",type="primary",disabled=not q.strip()):
        hits,res=search(q,k)
        st.metric("Indexed records",len(load()))
        if res["abstained"]: st.warning(res["answer"])
        else: st.markdown("### Evidence-grounded response"); st.write(res["answer"])
        st.subheader("Retrieved evidence")
        if not hits: st.info("No records retrieved.")
        for h in hits:
            r=h["record"]
            with st.expander(f"{r.subject or r.gene or 'Entity'} — {r.predicate or r.relation or 'text'} → {r.object or r.chemical or 'Entity'} | score {h['score']:.3f}"):
                st.write(r.text)
                st.json({"subject":r.subject,"subject_type":r.subject_type,"predicate":r.predicate,"object":r.object,"object_type":r.object_type,"source":r.source,"source_id":r.source_id,"PMID":r.pmid or None,"DOI":r.doi or None,"evidence_type":r.evidence_type,"polarity":r.evidence_polarity,"confidence":r.confidence,"BM25":round(h["bm25"],3),"TF-IDF (lexical semantic proxy)":round(h["semantic_tfidf"],3),"structured entity match":round(h["entity_match"],3)})
                if r.url: st.markdown(f"[Open source record]({r.url})")

elif page=="Knowledge Graph":
    records=load()
    graph,evidence=build_knowledge_graph(records)
    c1,c2,c3=st.columns(3)
    c1.metric("Indexed records",len(records)); c2.metric("Entities",graph.number_of_nodes()); c3.metric("Evidence edges",graph.number_of_edges())
    render_knowledge_graph(graph,evidence)

elif page=="Evidence conflicts":
    graph,_=build_knowledge_graph(load())
    conflicts=detect_evidence_conflicts(graph)
    st.subheader("Explicit polarity conflicts")
    st.caption("Only source-provided or independently validated POSITIVE/NEGATIVE labels are compared. Missing labels and non-significant findings are not treated as negative evidence.")
    st.metric("Potential conflicts",len(conflicts))
    if conflicts:
        for c in conflicts:
            with st.expander(f"{c['subject']} — {c['predicate']} → {c['object']}"):
                for a in c["assertions"]:
                    st.markdown(f"**{a['polarity']}** | {a['source']} | PMID: {a['pmid'] or 'not supplied'}")
                    st.write(a["evidence_text"])
    else: st.info("No explicit polarity conflicts found in the indexed data.")

elif page=="Upload data":
    st.write("Upload CSV, TSV, JSON, JSONL or TXT source exports. Original rows are preserved for auditing.")
    uploaded=st.file_uploader("Select dataset files",type=["csv","tsv","json","jsonl","txt"],accept_multiple_files=True)
    label=st.text_input("Source label (optional; filenames are used by default)","")
    if st.button("Import selected files",type="primary",disabled=not uploaded):
        total_new=0
        for f in uploaded:
            target=UPLOADS/Path(f.name).name
            target.write_bytes(f.getbuffer())
            n,total=ingest(target,label.strip() or Path(f.name).stem)
            total_new+=n
        st.success(f"Parsed {total_new} records. Total unique indexed records: {total}")

elif page=="Dataset overview":
    records=load()
    st.metric("Indexed evidence records",len(records))
    if records:
        df=pd.DataFrame([r.model_dump(exclude={"original_record"}) for r in records])
        st.dataframe(df,use_container_width=True)
        st.download_button("Download evidence JSONL","\n".join(r.model_dump_json() for r in records),"evidence.jsonl")

elif page=="Export RDF":
    st.write("Export structured subject–predicate–object assertions as RDF Turtle. Only records with explicit subject, predicate and object are exported.")
    if st.button("Generate RDF Turtle"):
        try:
            from tempfile import NamedTemporaryFile
            import tempfile
            path=Path(tempfile.gettempdir())/"pharmagraph_evidence.ttl"
            triples=export_rdf_turtle(load(),path)
            st.success(f"Exported RDF graph with {triples} triples.")
            st.download_button("Download RDF Turtle",path.read_text(encoding="utf-8"),"pharmagraph_evidence.ttl","text/turtle")
        except Exception as e: st.error(str(e))

elif page=="Methodology & limitations":
    st.markdown("""
### Current implemented components
- Dataset-specific field alias mapping and preservation of original input rows.
- Typed subject–predicate–object records where source fields support them.
- Deduplication at source-assertion level.
- BM25 + TF-IDF lexical retrieval + structured entity matching.
- Query-focused one-hop/two-hop graph visualization.
- Source-level provenance on each graph edge.
- Explicit polarity conflict grouping.
- RDF Turtle export for structured triples.

### Important limitations
- TF-IDF is not a biomedical dense embedding model.
- Entity normalization is conservative; authoritative HGNC, NCBI, PharmGKB/ClinPGx and drug identifier mapping still needs to be added and validated.
- Unknown entity types remain UNKNOWN rather than being guessed.
- Conflict detection only groups explicit POSITIVE/NEGATIVE labels; it is not a scientific contradiction classifier.
- Retrieval scores are relevance scores, not biological confidence.
- RDF export uses example.org URIs as local identifiers; replace with authoritative ontology URIs before publication.
- CHEMPROT-like chemical–protein interactions must not be interpreted as clinical pharmacogenomic recommendations.
- Dataset license, release, schema and provenance must be documented for each source.
""")
