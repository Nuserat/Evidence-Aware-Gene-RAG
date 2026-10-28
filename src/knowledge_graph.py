
import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network


def get_value(record, key, default=""):
    value = getattr(record, key, None)

    if value is None:
        extras = getattr(record, "model_extra", None) or {}
        value = extras.get(key, default)

    return str(value or "").strip()


def build_knowledge_graph(records, max_edges=300):
    graph = nx.MultiDiGraph()
    evidence = {}

    for record in records:
        gene = get_value(record, "gene")
        chemical = get_value(record, "chemical")
        relation = get_value(record, "relation") or "ASSOCIATED_WITH"

        entity_1 = get_value(record, "entity_1")
        entity_2 = get_value(record, "entity_2")

        if gene and chemical:
            source = f"Gene: {gene}"
            target = f"Chemical: {chemical}"
            source_type = "Gene"
            target_type = "Chemical"

        elif entity_1 and entity_2:
            source = f"Entity: {entity_1}"
            target = f"Entity: {entity_2}"
            source_type = target_type = "Entity"

        else:
            continue

        if source == target:
            continue

        if graph.number_of_edges() >= max_edges:
            break

        graph.add_node(
            source,
            label=source.split(": ", 1)[1],
            entity_type=source_type,
        )
        graph.add_node(
            target,
            label=target.split(": ", 1)[1],
            entity_type=target_type,
        )
        graph.add_edge(source, target, label=relation)

        evidence.setdefault((source, target, relation), []).append(record)

    return graph, evidence


def render_knowledge_graph(graph, evidence):
    if graph.number_of_nodes() == 0:
        st.info(
            "No graphable relationships found. "
            "Records need gene + chemical or entity_1 + entity_2."
        )
        return

    st.caption(
        f"Entities: {graph.number_of_nodes()} | "
        f"Relationships: {graph.number_of_edges()}"
    )

    net = Network(
        height="600px",
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

    for node, data in graph.nodes(data=True):
        net.add_node(
            node,
            label=data["label"],
            title=data["entity_type"],
            color=colors.get(data["entity_type"], "#999999"),
            size=22,
        )

    for source, target, data in graph.edges(data=True):
        net.add_edge(
            source,
            target,
            label=data["label"],
            title=data["label"],
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

    components.html(net.generate_html(), height=630, scrolling=True)

    node_map = {
        f"{data['entity_type']}: {data['label']}": node
        for node, data in graph.nodes(data=True)
    }

    selected = st.selectbox(
        "Inspect entity and supporting evidence",
        list(node_map.keys()),
    )

    if selected:
        node = node_map[selected]

        for source, target, data in graph.edges(data=True):
            if node not in (source, target):
                continue

            st.markdown(
                f"**{graph.nodes[source]['label']}** "
                f"— *{data['label']}* → "
                f"**{graph.nodes[target]['label']}**"
            )

            for record in evidence.get(
                (source, target, data["label"]), []
            ):
                with st.expander(
                    f"Evidence: {get_value(record, 'id', 'record')}"
                ):
                    st.write(get_value(record, "text", ""))
                    st.json({
                        "gene": get_value(record, "gene"),
                        "chemical": get_value(record, "chemical"),
                        "relation": get_value(record, "relation"),
                        "PMID": get_value(record, "pmid"),
                        "source": get_value(record, "source"),
                        "source_id": get_value(record, "source_id"),
                    })
