
from pathlib import Path

from src.config import DATASET, DATA
from src.core import read_records, rebuild_from_dataset, load
from src.knowledge_graph import build_knowledge_graph

print("\n========== DATASET PATH ==========")
print("Dataset:", DATASET.resolve())
print("Index:", DATA.resolve())

print("\n========== SOURCE FILES ==========")

for p in DATASET.rglob("train_triples.json"):
    print("Found:", p.resolve())

train_file = next(
    DATASET.rglob("train_triples.json"),
    None
)

if train_file is None:
    print("ERROR: train_triples.json NOT FOUND")
    print("Copy the original source file into dataset/")
    raise SystemExit(1)

print("\n========== PARSER TEST ==========")

parsed = read_records(train_file, train_file.name)

print("Parsed records:", len(parsed))

valid = [
    r for r in parsed
    if r.subject and r.predicate and r.object
]

print("Valid SPO records:", len(valid))

target = next(
    (
        r for r in parsed
        if "5-Fluorouracil" in r.text
        and "thymidylate synthase" in r.text
    ),
    None
)

if target:
    print("\n5-FLUOROURACIL TEST:")
    print("Subject:", repr(target.subject))
    print("Predicate:", repr(target.predicate))
    print("Object:", repr(target.object))
    print("Source:", target.source)
else:
    print("5-Fluorouracil record not found")

print("\n========== REBUILD INDEX ==========")

total, counts = rebuild_from_dataset(DATASET)

print("Rebuilt total:", total)
print("Source file counts:")

for name, count in counts.items():
    print(name, ":", count)

print("\n========== GRAPH TEST ==========")

records = load()

print("Loaded records:", len(records))

graph, evidence = build_knowledge_graph(records)

print("Entities:", graph.number_of_nodes())
print("Evidence edges:", graph.number_of_edges())

print("\n========== FINAL CHECK ==========")

if graph.number_of_edges() > 0:
    print("SUCCESS: Graph relationships created.")
else:
    print("FAILED: No graph relationships created.")
