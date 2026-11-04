
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd

from src.core import load, ingest, search
from src.config import UPLOADS, TOP_K
from src.knowledge_graph import (
    build_knowledge_graph,
    render_knowledge_graph,
    detect_evidence_conflicts,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PharmaGraph-RAG",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 PharmaGraph-RAG")
st.caption(
    "Hybrid pharmacogenomic evidence retrieval prototype "
    "for biomedical researchers"
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Search evidence",
        "Knowledge Graph",
        "Upload data",
        "Dataset overview",
        "Research notes"
    ]
)

# --------------------------------------------------
# 1. Search evidence
# --------------------------------------------------

if page == "Search evidence":

    st.warning(
        "Research prototype only. Not clinical interpretation, "
        "diagnosis or prescribing advice."
    )

    q = st.text_area(
        "Research question",
        placeholder="e.g. CYP2C19 clopidogrel"
    )

    k = st.slider("Top K", 3, 20, TOP_K)

    if st.button(
        "Retrieve evidence",
        type="primary",
        disabled=not q.strip()
    ):

        hits, res = search(q, k)

        st.caption(
            f"Indexed: {len(load())}; retrieved: {len(hits)}"
        )

        if res["abstained"]:
            st.warning("Abstention: insufficient evidence")
        else:
            st.success(
                "Evidence retrieved; verify source records"
            )

        st.subheader("Retrieved records")

        for h in hits:
            r = h["record"]

            with st.expander(
                f"{r.id} | score={h['score']:.3f} | {r.source}"
            ):

                st.write(r.text)

                st.json({
                    "gene": r.gene,
                    "chemical": r.chemical,
                    "relation": r.relation,
                    "PMID": r.pmid,
                    "split": r.split,
                    "source_id": r.source_id,
                    "url": r.url,
                    "BM25": h["bm25"],
                    "TFIDF": h["tfidf"],
                    "graph": h["graph"]
                })

                if r.url:
                    st.markdown(f"[Open source]({r.url})")


# --------------------------------------------------
# 2. Knowledge Graph
# --------------------------------------------------

elif page == "Knowledge Graph":

    records = load()

    st.subheader("Provenance-aware Genomic Knowledge Graph")

    if not records:
        st.info("No records indexed yet. Upload evidence records first.")

    else:

        # Build graph before using it
        graph, evidence = build_knowledge_graph(records)

        # Detect contradictions after graph creation
        conflicts = detect_evidence_conflicts(graph)

        # Graph statistics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Entities",
            graph.number_of_nodes()
        )

        col2.metric(
            "Evidence assertions",
            graph.number_of_edges()
        )

        col3.metric(
            "Conflicting relationships",
            len(conflicts)
        )

        # Render graph
        st.subheader("Knowledge Graph Visualization")

        render_knowledge_graph(graph, evidence)

        # Contradictory evidence
        st.subheader("Contradictory evidence")

        if conflicts:

            for item in conflicts:

                with st.expander(
                    f"{item['subject']} — "
                    f"{item['predicate']} → "
                    f"{item['object']}"
                ):

                    for assertion in item["assertions"]:

                        st.markdown(
                            f"**{assertion.get('polarity', 'NOT_ASSESSED')}** "
                            f"| Source: "
                            f"{assertion.get('source_name', 'Unknown')} "
                            f"| PMID: "
                            f"{assertion.get('pmid') or 'Not supplied'}"
                        )

                        if assertion.get("evidence_text"):
                            st.write(assertion["evidence_text"])

                        if assertion.get("url"):
                            st.markdown(
                                f"[Open original source]({assertion['url']})"
                            )

        else:
            st.info(
                "No explicit positive-vs-negative conflicts detected."
            )


# --------------------------------------------------
# 3. Upload data
# --------------------------------------------------

elif page == "Upload data":

    st.write(
        "CSV, TSV, JSON, JSONL and TXT accepted. "
        "Recommended fields: text, gene, chemical, relation, "
        "pmid, source, source_id, split, evidence_type, url."
    )

    ups = st.file_uploader(
        "Upload public dataset exports",
        type=["csv", "tsv", "json", "jsonl", "txt"],
        accept_multiple_files=True
    )

    source = st.text_input(
        "Source label",
        "public_dataset"
    )

    if st.button(
        "Index files",
        type="primary",
        disabled=not ups
    ):

        imported_total = 0
        total = len(load())

        for f in ups:

            p = UPLOADS / Path(f.name).name
            p.write_bytes(f.getbuffer())

            n, total = ingest(p, source)
            imported_total += n

        st.success(
            f"Imported {imported_total} records; "
            f"total indexed: {total}"
        )


# --------------------------------------------------
# 4. Dataset overview
# --------------------------------------------------

elif page == "Dataset overview":

    rs = load()

    st.metric("Indexed records", len(rs))

    if rs:

        df = pd.DataFrame([
            x.model_dump() for x in rs
        ])

        st.dataframe(
            df,
            use_container_width=True
        )

        st.download_button(
            "Download indexed JSONL",
            "\n".join(
                x.model_dump_json() for x in rs
            ),
            "evidence.jsonl"
        )


# --------------------------------------------------
# 5. Research notes
# --------------------------------------------------

else:

    st.markdown("""
    **Current baseline:** BM25 lexical retrieval + TF-IDF cosine
    retrieval + exact gene/chemical/relation matching.

    **Required extensions:** biomedical sentence-transformer
    embeddings; official CHEMPROT and verified GM-CIHT adapters;
    multi-hop graph retrieval; source-aware reranking;
    contradiction classification; independent claim-level
    entailment verification; calibrated abstention; expert-reviewed
    evaluation.

    CHEMPROT chemical-protein relations are not automatically
    clinical pharmacogenomic recommendations. Verify GM-CIHT's
    authoritative release, version, schema and licence before
    reporting benchmark results.
    """)
