import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
import re

def get_value(record, key, default=""):
    value = getattr(record, key, None)
    if value is None:
        extras = getattr(record, "model_extra", None) or {}
        value = extras.get(key, default)
    return value if value is not None else default


def clean(value):
    return str(value or "").strip()

def normalize_name(name):
    return re.sub(
        r"\s+",
        " ",
        str(name or "").strip().casefold()
    )


def entity_key(name, entity_type, canonical_id=""):
    entity_type = entity_type or "UNKNOWN"

    # Prefer an authoritative identifier when one is supplied.
    if canonical_id:
        return f"{entity_type}:{canonical_id.strip().upper()}"

    # Do not automatically merge different names without evidence
    # that they represent the same biological entity.
    return f"{entity_type}:{normalize_name(name)}"


def build_knowledge_graph(records):
    graph = nx.MultiDiGraph()
    evidence = {}

    for record in records:
        subject = record.subject or record.gene or record.entity_1
        object_ = record.object or record.chemical or record.entity_2

        predicate = (
            record.predicate
            or record.relation
            or "ASSOCIATED"
        )

        if not subject or not object_:
            continue

        subject_type = record.subject_type or "UNKNOWN"
        object_type = record.object_type or "UNKNOWN"

        source_key = entity_key(
            subject,
            subject_type,
            record.subject_id,
        )

        target_key = entity_key(
            object_,
            object_type,
            record.object_id,
        )

        graph.add_node(
            source_key,
            label=subject,
            entity_type=subject_type,
            canonical_id=record.subject_id,
        )

        graph.add_node(
            target_key,
            label=object_,
            entity_type=object_type,
            canonical_id=record.object_id,
        )

        assertion = {
            "assertion_id": record.id,
            "predicate": predicate,
            "evidence_id": record.id,
            "evidence_text": record.text,
            "evidence_polarity": record.evidence_polarity,
            "confidence": record.confidence,
            "source": record.source,
            "source_id": record.source_id,
            "pmid": record.pmid,
            "doi": record.doi,
            "url": record.url,
            "evidence_type": record.evidence_type,
            "dataset_version": record.dataset_version,
            "publication_year": record.publication_year,
            "statistical_significance": (
                record.statistical_significance
            ),
            "effect_direction": record.effect_direction,
            "effect_size": record.effect_size,
        }

        # Every assertion becomes a separate edge.
        graph.add_edge(
            source_key,
            target_key,
            key=record.id,
            **assertion,
        )

        triple = (source_key, predicate, target_key)

        evidence.setdefault(triple, []).append(assertion)

    return graph, evidence


def render_knowledge_graph(graph, evidence, max_edges=300):
    if graph.number_of_nodes() == 0:
        st.info(
            "No explicit relationships found. Records need gene + chemical "
            "or entity_1 + entity_2 fields."
        )
        return

    # Graph statistics are calculated from the full graph.
    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    self_loops = nx.number_of_selfloops(graph)
    isolated = sum(
        1 for node in graph.nodes()
        if graph.degree(node) == 0
    )

    components = list(nx.weakly_connected_components(graph))
    largest = max((len(c) for c in components), default=0)
    smallest = min((len(c) for c in components), default=0)

    high_confidence = sum(
        1
        for _, _, data in graph.edges(data=True)
        if data.get("confidence") is not None
        and data["confidence"] >= 0.8
    )

    unique_triples = {
        (source, data.get("label", "ASSOCIATED"), target)
        for source, target, data in graph.edges(data=True)
    }

    st.subheader("Graph overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{nodes:,}")
    c2.metric("Edges", f"{edges:,}")
    c3.metric("Unique RDF-style triples", f"{len(unique_triples):,}")
    c4.metric("High-confidence edges", f"{high_confidence:,}")

    st.subheader("Graph quality measures")
    c1, c2, c3 = st.columns(3)
    c1.metric("Self-loops", f"{self_loops:,}")
    c2.metric("Isolated nodes", f"{isolated:,}")
    c3.metric("Weakly connected components", f"{len(components):,}")

    c1, c2 = st.columns(2)
    c1.metric("Largest component", f"{largest:,} nodes")
    c2.metric("Smallest component", f"{smallest:,} node(s)")

    st.caption(
        "High-confidence threshold: 0.8, using supplied confidence values. "
        "Missing confidence values are not counted as high-confidence."
    )

    # Render only a subset for browser performance; keep full graph metrics.
    displayed_edges = list(graph.edges(keys=True, data=True))[:max_edges]
    visible_nodes = {
        node
        for source, target, _, _ in displayed_edges
        for node in (source, target)
    }

    net = Network(
        height="620px",
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#222222",
        cdn_resources="in_line",
    )

    colors = {
        "Gene": "#4C78A8",
        "Chemical": "#F58518",
        "Entity": "#999999",
    }

    for node in visible_nodes:
        data = graph.nodes[node]
        net.add_node(
            node,
            label=data["label"],
            title=f"Type: {data['entity_type']}",
            color=colors.get(data["entity_type"], "#999999"),
            size=22,
        )

    for source, target, key, data in displayed_edges:
        title = (
            f"Relation: {data.get('label', '')}\n"
            f"Evidence ID: {data.get('evidence_id', '')}\n"
            f"PMID: {data.get('pmid', '')}\n"
            f"Source: {data.get('source', '')}"
        )
        net.add_edge(
            source,
            target,
            label=data.get("label", ""),
            title=title,
            arrows="to",
        )

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "multiselect": true
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      },
      "edges": {
        "smooth": {"type": "dynamic"},
        "font": {"size": 12, "align": "middle"}
      }
    }
    """)

    st.caption(
        f"Displaying {len(displayed_edges):,} of {edges:,} edges. "
        "Metrics above use the full graph."
    )
    st.components.v1.html(
    net.generate_html(),
    height=650,
    scrolling=True,
)

    node_map = {
        f"{data['entity_type']}: {data['label']}": node
        for node, data in graph.nodes(data=True)
    }

    selected = st.selectbox(
        "Inspect entity and supporting evidence",
        sorted(node_map.keys()),
    )

    if selected:
        node = node_map[selected]
        st.markdown("#### Supporting relationships")

        found = False
        for source, target, key, data in graph.edges(
            node, keys=True, data=True
        ):
            found = True
            relation = data.get("label", "ASSOCIATED")
            st.markdown(
                f"**{graph.nodes[source]['label']}** "
                f"— *{relation}* → "
                f"**{graph.nodes[target]['label']}**"
            )
            st.caption(
                f"Evidence ID: {data.get('evidence_id') or 'Not supplied'} "
                f"| PMID: {data.get('pmid') or 'Not supplied'}"
            )
            if data.get("url"):
                st.markdown(f"[Open source evidence]({data['url']})")
            if data.get("evidence_text"):
                with st.expander("View evidence text"):
                    st.write(data["evidence_text"])

        # Also show incoming relationships for the selected entity.
        for source, target, key, data in graph.in_edges(
            node, keys=True, data=True
        ):
            found = True
            relation = data.get("label", "ASSOCIATED")
            st.markdown(
                f"**{graph.nodes[source]['label']}** "
                f"— *{relation}* → "
                f"**{graph.nodes[target]['label']}**"
            )
            st.caption(
                f"Evidence ID: {data.get('evidence_id') or 'Not supplied'} "
                f"| PMID: {data.get('pmid') or 'Not supplied'}"
            )
            if data.get("url"):
                st.markdown(f"[Open source evidence]({data['url']})")
            if data.get("evidence_text"):
                with st.expander("View evidence text"):
                    st.write(data["evidence_text"])

        if not found:
            st.info("This entity has no relationships in the graph.")


def detect_evidence_conflicts(graph):
    """
    Detect explicit positive-vs-negative evidence for the same
    normalized subject-predicate-object relationship.

    Non-significant findings are not automatically treated as negative.
    """
    grouped = {}

    for source, target, key, data in graph.edges(
        keys=True, data=True
    ):
        predicate = str(
            data.get("predicate") or data.get("label") or "ASSOCIATED"
        ).strip().upper()

        polarity = str(
            data.get("evidence_polarity") or "NOT_ASSESSED"
        ).strip().upper()

        triple = (source, predicate, target)
        grouped.setdefault(triple, []).append({
            "source": source,
            "target": target,
            "predicate": predicate,
            "polarity": polarity,
            "assertion_id": data.get("assertion_id", str(key)),
            "source_name": data.get("source", ""),
            "source_id": data.get("source_id", ""),
            "pmid": data.get("pmid", ""),
            "doi": data.get("doi", ""),
            "url": data.get("url", ""),
            "evidence_text": data.get("evidence_text", ""),
        })

    conflicts = []

    for (source, predicate, target), assertions in grouped.items():
        polarities = {item["polarity"] for item in assertions}

        if "POSITIVE" in polarities and "NEGATIVE" in polarities:
            conflicts.append({
                "subject": source,
                "predicate": predicate,
                "object": target,
                "assertions": assertions,
            })

    return conflicts

