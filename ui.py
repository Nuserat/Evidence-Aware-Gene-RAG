import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st,pandas as pd
from src.core import load,ingest,search,save
from src.config import UPLOADS,TOP_K
from src.knowledge_graph import (build_knowledge_graph,render_knowledge_graph,)
import networkx as nx

st.set_page_config(page_title="PharmaGraph-RAG",page_icon="🧬",layout="wide")
st.title("🧬 PharmaGraph-RAG")
st.caption("Hybrid pharmacogenomic evidence retrieval prototype for biomedical researchers")
page=st.sidebar.radio("Navigate",["Search evidence", "Knowledge Graph","Upload data","Dataset overview","Research notes"])

if page=="Search evidence":
 st.warning("Research prototype only. Not clinical interpretation, diagnosis or prescribing advice.")
 q=st.text_area("Research question",placeholder="e.g. CYP2C19 clopidogrel")
 k=st.slider("Top K",3,20,TOP_K)
 if st.button("Retrieve evidence",type="primary",disabled=not q.strip()):
  hits,res=search(q,k)
  st.caption(f"Indexed: {len(load())}; retrieved: {len(hits)}")
  if res["abstained"]:
    st.warning("Abstention: insufficient evidence")
  else:
    st.success("Evidence retrieved; verify source records")
  #st.markdown(res["answer"])
  st.subheader("Retrieved records")
  for h in hits:
   r=h["record"]
   with st.expander(f"{r.id} | score={h['score']:.3f} | {r.source}"):
    st.write(r.text);st.json({"gene":r.gene,"chemical":r.chemical,"relation":r.relation,"PMID":r.pmid,"split":r.split,"source_id":r.source_id,"url":r.url,"BM25":h["bm25"],"TFIDF":h["tfidf"],"graph":h["graph"]})
    if r.url:st.markdown(f"[Open source]({r.url})")

elif page == "Knowledge Graph":
    st.header("Biomedical Knowledge Graph")
    st.caption(
        "Explore entities, relationships, and their supporting evidence."
    )

    records = load()

    max_edges = st.slider(
        "Maximum relationships to display",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    graph, evidence = build_knowledge_graph(
        records,
        max_edges=max_edges,
    )

    if graph.number_of_nodes():
      high_confidence_spos = 0
      rdf_triples = set()

    for source, target, data in graph.edges(data=True):
        relation = str(data.get("label", "ASSOCIATED_WITH"))
        rdf_triples.add((source, relation, target))

        # Count high-confidence relationships only when
        # an actual confidence score is available.
        confidence = data.get("confidence")
        if confidence is not None:
            try:
                if float(confidence) >= 0.8:
                    high_confidence_spos += 1
            except (TypeError, ValueError):
                pass

    entities = graph.number_of_nodes()
    relationships = graph.number_of_edges()
    components = nx.number_weakly_connected_components(graph)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", entities)
    col2.metric("Edges", relationships)
    col3.metric("RDF Triples", len(rdf_triples))
    col4.metric("Connected components", components)

    col5, col6 = st.columns(2)
    col5.metric("High-confidence SPOs", high_confidence_spos)
    col6.metric("Entities", entities)

    render_knowledge_graph(graph, evidence)

elif page=="Upload data":
 st.write("CSV, TSV, JSON, JSONL and TXT accepted. Recommended fields: text, gene, chemical, relation, pmid, source, source_id, split, evidence_type, url.")
 ups=st.file_uploader("Upload public dataset exports",type=["csv","tsv","json","jsonl","txt"],accept_multiple_files=True)
 source=st.text_input("Source label","public_dataset")
 if st.button("Index files",type="primary",disabled=not ups):
  for f in ups:
   p=UPLOADS/Path(f.name).name;p.write_bytes(f.getbuffer());n,total=ingest(p,source)
  st.success(f"Imported {n} records; total indexed: {total}")
elif page=="Dataset overview":
 rs=load();st.metric("Indexed records",len(rs))
 if rs:
  df=pd.DataFrame([x.model_dump() for x in rs]);st.dataframe(df,use_container_width=True)
  st.download_button("Download indexed JSONL","\n".join(x.model_dump_json() for x in rs),"evidence.jsonl")
else:
 st.markdown("""
**Current baseline:** BM25 lexical retrieval + TF-IDF cosine retrieval + exact gene/chemical/relation matching.

**Required  extensions:** biomedical sentence-transformer embeddings; official CHEMPROT and verified GM-CIHT adapters; multi-hop graph retrieval; source-aware reranking; contradiction classification; independent claim-level entailment verification; calibrated abstention; expert-reviewed evaluation.

CHEMPROT chemical-protein relations are not automatically clinical pharmacogenomic recommendations. Verify GM-CIHT's authoritative release, version, schema and licence before reporting benchmark results.
""")
