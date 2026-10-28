import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network


def get_value(record, key, default=""):
    value = getattr(record, key, None)
    if value is None:
        extras = getattr(record, "model_extra", None) or {}
        value = extras.get(key, default)
    return value if value is not None else default


def clean(value):
    return str(value or "").strip()


def build_knowledge_graph(records, max_edges=300):
    # Keep the complete graph for accurate graph metrics.
    graph = nx.MultiDiGraph()
    evidence = {}
    for record in records:
        gene = clean(get_value(record, "gene"))
        chemical = clean(get_value(record, "chemical"))
        entity_1 = clean(get_value(record, "entity_1"))
        entity_2 = clean(get_value(record, "entity_2"))
        relation = clean(get_value(record, "relation")) or "ASSOCIATED"

        if gene and chemical:
            subject = gene
            object_ = chemical
            subject_type = "Gene"
            object_type = "Chemical"

        elif entity_1 and entity_2:
            subject = entity_1
            object_ = entity_2
            subject_type = "Entity"
            object_type = "Entity"

        else:
            # Do not invent a relationship from text alone.
            continue

        subject_key = f"{subject_type}: {subject}"
        object_key = f"{object_type}: {object_}"

        graph.add_node(
            subject_key,
            label=subject,
            entity_type=subject_type,
        )
        graph.add_node(
            object_key,
            label=object_,
            entity_type=object_type,
        )

        confidence = get_value(record, "confidence", None)
        try:
            confidence = (
                float(confidence) if confidence is not None else None
            )
        except (TypeError, ValueError):
            confidence = None

        edge_key = graph.add_edge(
            subject_key,
            object_key,
            label=relation,
            confidence=confidence,
            evidence_id=clean(get_value(record, "id")),
            evidence_text=clean(get_value(record, "text")),
            source=clean(get_value(record, "source")),
            source_id=clean(get_value(record, "source_id")),
            pmid=clean(get_value(record, "pmid")),
            url=clean(get_value(record, "url")),
        )

        evidence.setdefault(
            (subject_key, object_key, relation), []
        ).append(record)

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
