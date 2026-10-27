
from src.core import Record,save,search
def test_index_search():
 save([Record(id="a",text="CYP2C19 clopidogrel synthetic test",gene="CYP2C19",chemical="clopidogrel",source="test"),
       Record(id="b",text="unrelated synthetic text",gene="OTHER",chemical="OTHER",source="test")])
 hits,res=search("CYP2C19 clopidogrel",2)
 assert hits[0]["record"].id=="a"
