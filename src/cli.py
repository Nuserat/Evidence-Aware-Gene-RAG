import argparse
from .evidence_rag import read_jsonl, write_jsonl, EvidenceAwareRAG

def main():
    p = argparse.ArgumentParser(description="Evidence-aware Gene RAG research starter")
    p.add_argument("--corpus", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-retrieval-score", type=float, default=0.08)
    p.add_argument("--generator-model", default=None)
    p.add_argument("--nli-model", default=None)
    p.add_argument("--min-entailment", type=float, default=0.50)
    p.add_argument("--max-contradiction", type=float, default=0.40)
    p.add_argument("--no-generation", action="store_true", help="Retrieval-only smoke test")
    args = p.parse_args()

    corpus = read_jsonl(args.corpus)
    questions = read_jsonl(args.questions)
    rag = EvidenceAwareRAG(
        corpus, top_k=args.top_k, min_retrieval_score=args.min_retrieval_score,
        generator_model=None if args.no_generation else args.generator_model,
        nli_model=args.nli_model, min_entailment=args.min_entailment,
        max_contradiction=args.max_contradiction
    )
    outputs = []
    for i, q in enumerate(questions, 1):
        question = str(q.get("question", "")).strip()
        if not question:
            continue
        result = rag.answer(question)
        result.update({
            "question_id": q.get("id", f"q_{i}"),
            "question": question,
            "gold": {
                "reference_answer": q.get("reference_answer"),
                "gold_doc_ids": q.get("gold_doc_ids"),
                "answerable": q.get("answerable"),
            },
            "config": {
                "top_k": args.top_k,
                "min_retrieval_score": args.min_retrieval_score,
                "generator_model": None if args.no_generation else args.generator_model,
                "nli_model": args.nli_model,
            },
        })
        outputs.append(result)
    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} predictions to {args.output}")

if __name__ == "__main__":
    main()
