
from collections import Counter, defaultdict
from html import escape

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network


def field(record, name, default=""):
    """Read fields from Pydantic records, including optional extra fields."""
    value = getattr(record, name, None)

    if value is None:
        extra = getattr(record, "model_extra", None) or {}
        value = extra.get(name, default)

    if value is None:
        return default

    return str(value).strip()


def build_knowledge_graph(records, max_edges=300):
    """
    Build a biomedical graph from existing indexed evidence records.

    Supported relationships:
      gene --relation--> chemical
      entity_1 --relation--> entity_2

    Missing entities are not invented.
    """
    graph = nx.MultiDiGraph()
    evidence = defaultdict(list)

    # Keep the strongest available relationships first.
    candidates = []

    for record in records:
        gene = field(record, "gene")
        chemical = field(record, "chemical")
        relation = field(record, "relation")
        entity_1 = field(record, "entity_1")
        entity_2 = field(record, "entity_2")

        if gene and chemical:
            source = f"gene::{gene}"
            target = f"chemical::{chemical}"
            source_type = "Gene"
            target_type = "Chemical"

        elif entity_1 and entity_2:
            # Use explicit entity types when available.
            type_1 = field(record, "entity_1_type", "Entity")
            type_2 = field(record, "entity_2_type", "Entity")

            source = f"{type_1}::{entity_1}"
            target = f"{type_2}::{entity_2}"
            source_type = type_1
            target_type = type_2

        else:
            continue

        if source == target:
            continue

        relation = relation or "ASSOCIATED_WITH"

        candidates.append(
            (
                source,
                target,
                source_type,
                target_type,
                relation,
                record,
            )
        )

    for (
        source,
        target,
        source_type,
        target_type,
        relation,
        record,
    ) in candidates[:max_edges]:

        graph.add_node(
            source,
            label=source.split("::", 1)[1],
            entity_type=source_type,
        )

        graph.add_node(
            target,
            label=target.split("::", 1)[1],
            entity_type=target_type,
        )

        graph.add_edge(
            source,
            target,
            label=relation,
            title=relation,
        )

        evidence[(source, target, relation)].append(record)

    return graph, evidence


def render_knowledge_graph(graph, evidence):
    if graph.number_of_nodes() == 0:
        st.info(
            "No graphable relationships found. "
            "Check whether the uploaded records contain both "
            "gene and chemical fields, or entity_1 and entity_2."
        )
        return

    st.caption(
        f"{graph.number_of_nodes()} entities · "
        f"{graph.number_of_edges()} relationships"
    )

    # Entity types determine node colors.
    colors = {
        "Gene": "#4C78A8",
        "Chemical": "#F58518",
        "Protein": "#54A24B",
        "Disease": "#E45756",
        "Drug": "#F58518",
        "Entity": "#9D9DA0",
    }

    net = Network(
        height="650px",
        width="100%",
        directed=True,
        bgcolor="#FFFFFF",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )

    for node_id, data in graph.nodes(data=True):
        entity_type = data.get("entity_type", "Entity")
        label = data.get("label", node_id)

        net.add_node(
            node_id,
            label=label,
            title=f"Type: {escape(entity_type)}",
            color=colors.get(entity_type, colors["Entity"]),
            shape="dot",
            size=22,
            font={"size": 16},
        )

    for source, target, data in graph.edges(data=True):
        relation = data.get("label", "ASSOCIATED_WITH")

        net.add_edge(
            source,
            target,
            label=relation,
            title=escape(relation),
            arrows="to",
            font={"size": 11, "align": "middle"},
        )

    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "multiselect": true
      },
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 150},
        "barnesHut": {
          "gravitationalConstant": -2500,
          "centralGravity": 0.25,
          "springLength": 160,
          "springConstant": 0.04,
          "damping": 0.15
        }
      },
      "edges": {
        "smooth": {"type": "dynamic"},
        "color": {"color": "#999999", "highlight": "#222222"}
      }
    }
    """)

    html = net.generate_html()
    components.html(html, height=680, scrolling=True)

    # Streamlit controls for inspecting the graph and its evidence.
    nodes = list(graph.nodes(data=True))
    node_options = {
        f"{data.get('entity_type', 'Entity')}: "
        f"{data.get('label', node_id)}": node_id
        for node_id, data in nodes
    }

    selected_label = st.selectbox(
        "Inspect an entity",
        options=list(node_options.keys()),
        key="kg_selected_entity",
    )

    if selected_label:
        selected_node = node_options[selected_label]

        st.subheader("Entity details")
        st.write(f"**Entity:** {graph.nodes[selected_node]['label']}")
        st.write(
            f"**Type:** "
            f"{graph.nodes[selected_node].get('entity_type', 'Entity')}"
        )

        related = []

        for source, target, data in graph.edges(data=True):
            if source == selected_node or target == selected_node:
                related.append(
                    {
                        "source": graph.nodes[source]["label"],
                        "relation": data.get("label", ""),
                        "target": graph.nodes[target]["label"],
                        "evidence_key": (source, target, data.get("label", "")),
                    }
                )

        st.subheader("Relationships and supporting evidence")

        if not related:
            st.info("No connected relationships for this entity.")
        else:
            for item in related:
                st.markdown(
                    f"**{item['source']}** "
                    f"— *{item['relation']}* → "
                    f"**{item['target']}**"
                )

                matching_records = evidence.get(
                    item["evidence_key"], []
                )

                for record in matching_records:
                    with st.expander(
                        f"Evidence: {field(record, 'id', 'record')} "
                        f"| {field(record, 'source', 'unknown source')}"
                    ):
                        st.write(field(record, "text", "No evidence text"))
                        st.json(
                            {
                                "PMID": field(record, "pmid"),
                                "source_id": field(record, "source_id"),
                                "gene": field(record, "gene"),
                                "chemical": field(record, "chemical"),
                                "relation": field(record, "relation"),
                                "url": field(record, "url"),
                            }
                        )

                        url = field(record, "url")
                        if url.startswith(("https://", "http://")):
                            st.markdown(f"[Open source]({url})")
