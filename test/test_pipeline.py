from src.core import convert, Record, save, search
from src.knowledge_graph import build_knowledge_graph, detect_evidence_conflicts

def test_pharmgkb_style_mapping():
    r=convert({"Entity1_name":"CYP2C19","Entity1_type":"Gene","Entity2_name":"clopidogrel","Entity2_type":"Chemical","Association":"associated","PMIDs":"12345","Evidence":"clinical annotation"},"PharmGKB")
    assert r.subject=="CYP2C19"
    assert r.subject_type=="GENE"
    assert r.object=="clopidogrel"
    assert r.pmid=="12345"

def test_triple_mapping():
    r=convert({"SUBJECT_TEXT":"CYP2C19","PREDICATE":"ASSOCIATED_WITH","OBJECT_TEXT":"clopidogrel","SENTENCE":"Example sentence"},"triples")
    assert r.subject=="CYP2C19" and r.object=="clopidogrel"
    assert r.text=="Example sentence"

def test_graph_preserves_assertion():
    r=Record(id="x",text="evidence sentence",subject="GENE1",subject_type="GENE",predicate="ASSOCIATED_WITH",object="DRUG1",object_type="DRUG",source="test",pmid="111")
    g,e=build_knowledge_graph([r])
    assert g.number_of_nodes()==2 and g.number_of_edges()==1
    edge=list(g.edges(data=True))[0][2]
    assert edge["pmid"]=="111"

def test_conflict_requires_explicit_polarity():
    a=Record(id="a",text="supports",subject="G",predicate="ASSOCIATED",object="D",evidence_polarity="POSITIVE")
    b=Record(id="b",text="opposes",subject="G",predicate="ASSOCIATED",object="D",evidence_polarity="NEGATIVE")
    g,_=build_knowledge_graph([a,b])
    assert len(detect_evidence_conflicts(g))==1
