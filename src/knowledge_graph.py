import re
import networkx as nx
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components

def _norm(s): return re.sub(r"\s+"," ",str(s or "").strip().casefold())
def entity_key(name, typ, ident=""):
    if ident: return f"{typ}:{ident.strip().upper()}"
    return f"{typ}:{_norm(name)}"

def build_knowledge_graph(records):
    graph=nx.MultiDiGraph()
    evidence={}
    for r in records:
        subject=r.subject or r.gene or r.entity_1
        obj=r.object or r.chemical or r.entity_2
        predicate=r.predicate or r.relation
        if not subject or not obj or not predicate: continue
        stype=r.subject_type or "UNKNOWN"; otype=r.object_type or "UNKNOWN"
        sk=entity_key(subject,stype,r.subject_id); ok=entity_key(obj,otype,r.object_id)
        graph.add_node(sk,label=subject,entity_type=stype,canonical_id=r.subject_id)
        graph.add_node(ok,label=obj,entity_type=otype,canonical_id=r.object_id)
        assertion={
            "assertion_id":r.id,"predicate":predicate,"evidence_id":r.id,
            "evidence_text":r.text,"evidence_polarity":r.evidence_polarity,
            "confidence":r.confidence,"source":r.source,"source_id":r.source_id,
            "pmid":r.pmid,"doi":r.doi,"url":r.url,"evidence_type":r.evidence_type,
            "dataset_version":r.dataset_version,"publication_year":r.publication_year,
            "statistical_significance":r.statistical_significance,
            "effect_direction":r.effect_direction,"effect_size":r.effect_size
        }
        graph.add_edge(sk,ok,key=r.id,**assertion)
        evidence.setdefault((sk,predicate,ok),[]).append(assertion)
    return graph,evidence

def detect_evidence_conflicts(graph):
    grouped={}
    for s,o,k,d in graph.edges(keys=True,data=True):
        pol=(d.get("evidence_polarity") or "NOT_ASSESSED").upper()
        if pol not in {"POSITIVE","NEGATIVE"}: continue
        key=(s,(d.get("predicate") or "").casefold(),o)
        grouped.setdefault(key,[]).append({
            "polarity":pol,"source":d.get("source"),"source_id":d.get("source_id"),
            "pmid":d.get("pmid"),"doi":d.get("doi"),"url":d.get("url"),
            "evidence_text":d.get("evidence_text"),"assertion_id":d.get("assertion_id")
        })
    return [{"subject":s,"predicate":p,"object":o,"assertions":a}
            for (s,p,o),a in grouped.items()
            if {x["polarity"] for x in a}=={"POSITIVE","NEGATIVE"}]

def render_knowledge_graph(graph, evidence, max_edges=250):
    if not graph.number_of_edges():
        st.info("No explicit subject–predicate–object relationships are available. Check imported schemas.")
        return
    st.caption("Graph counts are based on imported structured triples. Unknown entity types are not automatically guessed.")
    # Search/filter to display a focused subgraph rather than a crowded global graph.
    labels={n:d.get("label",n) for n,d in graph.nodes(data=True)}
    options=sorted(labels, key=lambda n: labels[n].casefold())
    selected=st.selectbox("Focus on an entity",["All entities"]+options,format_func=lambda x: x if x=="All entities" else f"{graph.nodes[x]['entity_type']}: {labels[x]}")
    hops=st.radio("Neighbourhood",["1 hop","2 hops"],horizontal=True)
    if selected=="All entities":
        sub=graph.copy()
    else:
        und=graph.to_undirected()
        nodes={selected}
        frontier={selected}
        for _ in range(1 if hops=="1 hop" else 2):
            frontier={v for n in frontier for v in und.neighbors(n)}-nodes
            nodes |= frontier
        sub=graph.subgraph(nodes).copy()
    edges=list(sub.edges(keys=True,data=True))[:max_edges]
    visible={x for s,o,k,d in edges for x in (s,o)}
    color={"GENE":"#4C78A8","VARIANT":"#9467bd","DRUG":"#2ca02c","CHEMICAL":"#ff7f0e","PROTEIN":"#17becf","DISEASE":"#d62728","PHENOTYPE":"#e377c2"}
    net=Network(height="600px",width="100%",directed=True,bgcolor="#101820",font_color="#ffffff",cdn_resources="in_line")
    for n in visible:
        d=graph.nodes[n]
        net.add_node(n,label=d["label"],title=f"Type: {d['entity_type']}\nID: {d.get('canonical_id') or 'not supplied'}",color=color.get(d["entity_type"],"#999999"),size=24)
    for s,o,k,d in edges:
        title=f"Predicate: {d['predicate']}\nSource: {d['source']}\nPMID: {d['pmid'] or 'not supplied'}\nEvidence ID: {d['evidence_id']}"
        net.add_edge(s,o,label=d["predicate"],title=title,arrows="to")
    net.set_options('{"physics":{"stabilization":{"iterations":120}},"interaction":{"hover":true,"navigationButtons":true},"edges":{"font":{"size":10,"align":"middle"}}}')
    components.html(net.generate_html(),height=620,scrolling=True)
    st.caption(f"Showing {len(edges)} of {sub.number_of_edges()} edges in the selected subgraph (display limit {max_edges}).")
    st.subheader("Evidence behind relationships")
    for s,o,k,d in edges:
        with st.expander(f"{labels[s]} — {d['predicate']} → {labels[o]} | {d['source']}"):
            st.write(d.get("evidence_text") or "No evidence sentence supplied.")
            st.write({"PMID":d.get("pmid") or "Not supplied","DOI":d.get("doi") or "Not supplied","Source record":d.get("source_id"),"Evidence type":d.get("evidence_type"),"Polarity":d.get("evidence_polarity"),"Confidence":d.get("confidence")})
            if d.get("url"): st.markdown(f"[Open source]({d['url']})")

def export_rdf_turtle(records, path):
    try:
        from rdflib import Graph, URIRef, Literal, Namespace
        from rdflib.namespace import RDF, RDFS, XSD
    except ImportError as e:
        raise RuntimeError("Install rdflib to export RDF: pip install rdflib") from e
    g=Graph()
    EX=Namespace("https://example.org/pharmagraph/")
    g.bind("pg",EX); g.bind("rdfs",RDFS)
    for r in records:
        if not (r.subject and r.object and r.predicate): continue
        def safe_uri(prefix, value):
            import hashlib
            return URIRef(str(EX)+prefix+"/"+hashlib.sha1(str(value).encode()).hexdigest())
        s=safe_uri("entity",(r.subject_type,r.subject_id or r.subject))
        o=safe_uri("entity",(r.object_type,r.object_id or r.object))
        pred=URIRef(str(EX)+"relation/"+re.sub(r"[^A-Za-z0-9_-]","_",r.predicate))
        ev=safe_uri("evidence",r.id)
        g.add((s,RDFS.label,Literal(r.subject))); g.add((o,RDFS.label,Literal(r.object)))
        g.add((s,pred,o)); g.add((ev,RDF.type,EX.EvidenceAssertion))
        g.add((ev,EX.subject,s)); g.add((ev,EX.predicate,Literal(r.predicate))); g.add((ev,EX.object,o))
        g.add((ev,EX.source,Literal(r.source))); g.add((ev,EX.sourceRecord,Literal(r.source_id)))
        if r.pmid: g.add((ev,EX.pmid,Literal(r.pmid)))
        if r.text: g.add((ev,EX.evidenceText,Literal(r.text)))
    g.serialize(destination=str(path),format="turtle")
    return len(g)
