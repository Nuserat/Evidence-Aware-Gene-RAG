from __future__ import annotations
import re
import networkx as nx
import streamlit as st
from pyvis.network import Network


def _clean(value):
    return str(value or "").strip()


def _key(name, kind, identifier=""):
    identity = identifier.strip().upper() if identifier else re.sub(r"\s+", " ", name.casefold().strip())
    return f"{kind.upper()}::{identity}"



def build_knowledge_graph(records):
    """Build a provenance-aware graph from explicit SPO and supported annotations."""

    graph = nx.MultiDiGraph()
    evidence_index = {}

    def add_edge(
        subject,
        subject_type,
        predicate,
        obj,
        object_type,
        record,
        subject_id="",
        object_id=""
    ):
        subject = _clean(subject)
        obj = _clean(obj)
        predicate = _clean(predicate)

        # Require all three components of an explicit assertion.
        if not subject or not obj or not predicate:
            return

        subject_type = _clean(subject_type) or "UNKNOWN"
        object_type = _clean(object_type) or "UNKNOWN"

        s_key = _key(subject, subject_type, subject_id)
        o_key = _key(obj, object_type, object_id)

        graph.add_node(
            s_key,
            label=subject,
            entity_type=subject_type,
            canonical_id=subject_id
        )

        graph.add_node(
            o_key,
            label=obj,
            entity_type=object_type,
            canonical_id=object_id
        )

        assertion = {
            "assertion_id": getattr(record, "id", ""),
            "evidence_id": getattr(record, "id", ""),
            "predicate": predicate,
            "label": predicate,
            "evidence_text": getattr(record, "text", ""),
            "evidence_polarity": getattr(
                record, "evidence_polarity", "NOT_ASSESSED"
            ),
            "confidence": getattr(record, "confidence", None),
            "source": getattr(record, "source", ""),
            "source_id": getattr(record, "source_id", ""),
            "pmid": getattr(record, "pmid", ""),
            "doi": getattr(record, "doi", ""),
            "url": getattr(record, "url", ""),
            "evidence_type": getattr(record, "evidence_type", ""),
            "dataset_version": getattr(record, "dataset_version", ""),
            "publication_year": getattr(record, "publication_year", None),
            "statistical_significance": getattr(
                record, "statistical_significance", None
            ),
            "effect_direction": getattr(record, "effect_direction", ""),
            "effect_size": getattr(record, "effect_size", None),
        }

        edge_key = (
            f"{assertion['evidence_id']}:{predicate}:"
            f"{graph.number_of_edges()}"
        )

        graph.add_edge(
            s_key,
            o_key,
            key=edge_key,
            **assertion
        )

        evidence_index.setdefault(
            (s_key, predicate, o_key), []
        ).append(assertion)

    for r in records:

        subject = getattr(r, "subject", "")
        obj = getattr(r, "object", "")
        predicate = (
            getattr(r, "predicate", "")
            or getattr(r, "relation", "")
        )

        # Explicit subject-predicate-object records.
        if _clean(subject) and _clean(predicate) and _clean(obj):
            add_edge(
                subject,
                getattr(r, "subject_type", "UNKNOWN"),
                predicate,
                obj,
                getattr(r, "object_type", "UNKNOWN"),
                r,
                getattr(r, "subject_id", ""),
                getattr(r, "object_id", "")
            )
            continue

        # PharmGKB variant-to-gene annotation.
        variant = getattr(r, "variant", "")
        gene = getattr(r, "gene", "")
        chemical = getattr(r, "chemical", "")

        if _clean(variant) and _clean(gene):
            add_edge(
                variant, "Variant",
                "VARIANT_ANNOTATED_FOR_GENE",
                gene, "Gene", r
            )

        # Variant-drug annotation.
        if _clean(variant) and _clean(chemical):
            add_edge(
                variant, "Variant",
                "VARIANT_DRUG_ANNOTATION",
                chemical, "Drug", r
            )

        # Gene-drug association, only when no variant relation exists.
        elif _clean(gene) and _clean(chemical):
            add_edge(
                gene, "Gene",
                predicate or "GENE_DRUG_ASSOCIATION",
                chemical, "Drug", r
            )

        # PharmGKB relationships table.
        entity_1 = getattr(r, "entity_1", "")
        entity_2 = getattr(r, "entity_2", "")

        if _clean(entity_1) and _clean(entity_2):
            add_edge(
                entity_1, "Entity",
                predicate or "ASSOCIATED_WITH",
                entity_2, "Entity", r
            )

    return graph, evidence_index



def render_knowledge_graph(graph, evidence=None, max_edges=350):
    if graph.number_of_edges() == 0:
        st.warning("No graph assertions were created. Rebuild the index from the original TSV/JSON/JSONL source exports; an old evidence index may not contain the structured source columns.")
        return

    edges = list(graph.edges(keys=True, data=True))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entities", f"{graph.number_of_nodes():,}")
    c2.metric("Evidence assertions", f"{graph.number_of_edges():,}")
    c3.metric("Unique relationships", f"{len({(u, d.get('predicate'), v) for u, v, _, d in edges}):,}")
    c4.metric("Records represented", f"{len({d.get('evidence_id') for _, _, _, d in edges}):,}")

    st.caption("Edges represent source-record annotations or extracted relations. They are not automatically causal claims or clinical recommendations.")
    shown = edges[:max_edges]
    visible_nodes = {n for u, v, _, _ in shown for n in (u, v)}
    net = Network(height="620px", width="100%", directed=True, bgcolor="#FFFFFF", font_color="#202020", cdn_resources="in_line")
    palette = {"GENE": "#4C78A8", "VARIANT": "#54A24B", "DRUG": "#F58518", "CHEMICAL": "#F58518", "DISEASE": "#B279A2", "PHENOTYPE": "#E45756", "PROTEIN": "#72B7B2"}
    for node in visible_nodes:
        d = graph.nodes[node]
        kind = str(d.get("entity_type", "UNKNOWN")).upper()
        net.add_node(node, label=d.get("label", node), title=f"Type: {kind}", color=palette.get(kind, "#9D9D9D"), size=22)
    for u, v, key, d in shown:
        title = (f"Relation: {d.get('predicate')}\nEvidence: {d.get('evidence_id')}\n"
                 f"Source: {d.get('source')}\nPMID: {d.get('pmid') or 'Not supplied'}\n"
                 f"Polarity: {d.get('evidence_polarity')}")
        net.add_edge(u, v, label=d.get("predicate", ""), title=title, arrows="to")
    net.set_options('''{"interaction":{"hover":true,"navigationButtons":true,"multiselect":true},"physics":{"enabled":true,"stabilization":{"iterations":120}},"edges":{"smooth":{"type":"dynamic"},"font":{"size":11,"align":"middle"}}}''')
    st.caption(f"Showing {len(shown):,} of {len(edges):,} assertions for display performance.")
    st.components.v1.html(net.generate_html(), height=650, scrolling=True)

    labels = {f"{graph.nodes[n].get('entity_type')}: {graph.nodes[n].get('label')}": n for n in graph.nodes}
    selected_label = st.selectbox("Inspect entity and evidence", sorted(labels))
    if selected_label:
        node = labels[selected_label]
        incident = list(graph.in_edges(node, keys=True, data=True)) + list(graph.out_edges(node, keys=True, data=True))
        for u, v, key, d in incident[:100]:
            st.markdown(f"**{graph.nodes[u]['label']}** — *{d.get('predicate')}* → **{graph.nodes[v]['label']}**")
            st.caption(f"Source: {d.get('source')} | Record: {d.get('evidence_id')} | PMID: {d.get('pmid') or 'Not supplied'}")
            if d.get("evidence_text"):
                with st.expander("View source evidence"):
                    st.write(d["evidence_text"])
            if d.get("url"):
                st.markdown(f"[Open source]({d['url']})")


def detect_evidence_conflicts(graph):
    """Return only explicitly polarity-labelled positive-vs-negative pairs."""
    groups = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        polarity = str(data.get("evidence_polarity", "NOT_ASSESSED")).upper()
        pred = str(data.get("predicate", "ASSOCIATED_WITH")).upper()
        groups.setdefault((u, pred, v), []).append({
            "polarity": polarity, "source": data.get("source", ""),
            "source_id": data.get("source_id", ""), "pmid": data.get("pmid", ""),
            "url": data.get("url", ""), "evidence_text": data.get("evidence_text", ""),
            "evidence_id": data.get("evidence_id", ""),
        })
    conflicts = []
    for (u, pred, v), assertions in groups.items():
        polarities = {a["polarity"] for a in assertions}
        if "POSITIVE" in polarities and "NEGATIVE" in polarities:
            conflicts.append({"subject": graph.nodes[u]["label"], "predicate": pred,
                              "object": graph.nodes[v]["label"], "assertions": assertions})
    return conflicts



def export_rdf_turtle(graph):
    """Export graph assertions and provenance as RDF Turtle."""

    import hashlib
    import json
    from urllib.parse import quote

    entity_ns = "https://w3id.org/pharmagraph/entity/"
    relation_ns = "https://w3id.org/pharmagraph/relation/"
    provenance_ns = "https://w3id.org/pharmagraph/provenance/"
    type_ns = "https://w3id.org/pharmagraph/type/"

    def make_uri(namespace, value):
        raw = str(value)
        slug = quote(raw, safe="._-:")[:100]
        digest = hashlib.sha1(
            raw.encode("utf-8")
        ).hexdigest()[:10]

        return f"<{namespace}{slug}_{digest}>"

    def literal(value):
        return json.dumps(str(value), ensure_ascii=False)

    lines = [
        '@prefix pg: <https://w3id.org/pharmagraph/entity/> .',
        '@prefix rel: <https://w3id.org/pharmagraph/relation/> .',
        '@prefix prov: <https://w3id.org/pharmagraph/provenance/> .',
        '@prefix pgtype: <https://w3id.org/pharmagraph/type/> .',
        '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
        '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
        ''
    ]

    # Export entities.
    for node, data in graph.nodes(data=True):
        node_uri = make_uri(entity_ns, node)
        label = literal(data.get("label", node))
        kind = quote(
            str(data.get("entity_type", "UNKNOWN")),
            safe="._-"
        )

        lines.append(
            f'{node_uri} rdfs:label {label} ; '
            f'rdf:type pgtype:{kind} .'
        )

    lines.append("")

    # Export relationships and their provenance.
    for subject, obj, key, data in graph.edges(
        keys=True, data=True
    ):
        s_uri = make_uri(entity_ns, subject)
        o_uri = make_uri(entity_ns, obj)

        predicate = quote(
            str(data.get("predicate", "ASSOCIATED_WITH")),
            safe="._-"
        )

        lines.append(
            f'{s_uri} rel:{predicate} {o_uri} .'
        )

        evidence_id = str(
            data.get("evidence_id") or key
        )

        assertion_uri = make_uri(
            provenance_ns,
            evidence_id + "|" + str(key)
        )

        properties = [
            "a prov:EvidenceAssertion",
            f"prov:subject {s_uri}",
            f"prov:predicate {literal(data.get('predicate', ''))}",
            f"prov:object {o_uri}"
        ]

        fields = {
            "source": "source",
            "sourceId": "source_id",
            "pmid": "pmid",
            "doi": "doi",
            "evidenceText": "evidence_text",
            "evidencePolarity": "evidence_polarity"
        }

        for rdf_field, record_field in fields.items():
            value = data.get(record_field)

            if value not in (None, ""):
                properties.append(
                    f"prov:{rdf_field} {literal(value)}"
                )

        lines.append(
            assertion_uri + " "
            + " ;\n    ".join(properties) + " ."
        )

    return "\n".join(lines) + "\n"
