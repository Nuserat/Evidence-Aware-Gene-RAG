import argparse, json
from pathlib import Path
from .evidence_rag import read_jsonl

def main():
    p = argparse.ArgumentParser(description="Evaluate evidence-aware RAG outputs")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = read_jsonl(args.predictions)
    n = len(rows)
    abstained = [bool(r.get("abstained")) for r in rows]
    answerable = [r.get("gold", {}).get("answerable") for r in rows]
    known = [(i, x) for i, x in enumerate(answerable) if x is not None]
    insufficient_idx = [i for i, x in known if x is False]
    sufficient_idx = [i for i, x in known if x is True]
    retrieved_recall = []
    citation_coverage = []
    for r in rows:
        gold_ids = r.get("gold", {}).get("gold_doc_ids")
        if gold_ids:
            got = {str(e.get("id")) for e in r.get("evidence", [])}
            retrieved_recall.append(len(got & set(map(str, gold_ids))) / len(set(map(str, gold_ids))))
        if not r.get("abstained"):
            citation_coverage.append(bool(__import__("re").search(r"\[E\d+\]", r.get("answer", ""))))
    metrics = {
        "n": n,
        "abstention_rate": sum(abstained) / n if n else None,
        "over_answer_rate_on_unanswerable": (
            sum(not abstained[i] for i in insufficient_idx) / len(insufficient_idx)
            if insufficient_idx else None
        ),
        "abstention_rate_on_answerable": (
            sum(abstained[i] for i in sufficient_idx) / len(sufficient_idx)
            if sufficient_idx else None
        ),
        "mean_retrieval_recall_at_k": sum(retrieved_recall) / len(retrieved_recall) if retrieved_recall else None,
        "citation_coverage_on_non_abstained": sum(citation_coverage) / len(citation_coverage) if citation_coverage else None,
        "note": "This evaluator does not compute semantic answer accuracy. Add a task-appropriate, validated scorer for your dataset."
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
