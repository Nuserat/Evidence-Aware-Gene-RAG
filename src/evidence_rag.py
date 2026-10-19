from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class Evidence:
    id: str
    title: str
    source: str
    url: str
    text: str
    retrieval_score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TfidfRetriever:
    """Transparent sparse retrieval baseline. Fit only on the supplied corpus."""
    def __init__(self, documents: list[dict[str, Any]], ngram_range=(1, 2)):
        if not documents:
            raise ValueError("Corpus is empty.")
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=ngram_range,
            max_features=500_000, sublinear_tf=True
        )
        texts = [self._document_text(d) for d in documents]
        self.matrix = self.vectorizer.fit_transform(texts)

    @staticmethod
    def _document_text(d: dict[str, Any]) -> str:
        return f"{d.get('title', '')}. {d.get('text', '')}".strip()

    def retrieve(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not query.strip():
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = np.argsort(scores)[::-1][:max(0, top_k)]
        results = []
        for rank, idx in enumerate(order, 1):
            d = self.documents[int(idx)]
            results.append(Evidence(
                id=str(d.get("id", f"doc_{idx}")),
                title=str(d.get("title", "")),
                source=str(d.get("source", "unknown")),
                url=str(d.get("url", "")),
                text=str(d.get("text", "")),
                retrieval_score=float(scores[idx]),
                rank=rank,
            ))
        return results


class NliVerifier:
    """Optional NLI signal: premise=evidence, hypothesis=generated answer."""
    def __init__(self, model_name: str):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_name, top_k=None)
        self.labels = self._label_map()

    def _label_map(self):
        cfg = getattr(self.pipe.model, "config", None)
        id2label = getattr(cfg, "id2label", {}) or {}
        result = {}
        for idx, label in id2label.items():
            name = str(label).lower()
            if "entail" in name:
                result["entailment"] = int(idx)
            elif "contrad" in name:
                result["contradiction"] = int(idx)
            elif "neutral" in name:
                result["neutral"] = int(idx)
        return result

    def score(self, answer: str, evidence: list[Evidence]) -> dict[str, Any]:
        if not answer.strip() or not evidence:
            return {"max_entailment": 0.0, "max_contradiction": 0.0, "per_evidence": []}
        pairs = [(e.text[:3000], answer[:1500]) for e in evidence]
        raw = self.pipe(pairs, truncation=True)
        rows = []
        for e, predictions in zip(evidence, raw):
            # Transformers pipeline versions can return dict or list[dict].
            if isinstance(predictions, dict):
                predictions = [predictions]
            scores = {str(x["label"]).lower(): float(x["score"]) for x in predictions}
            ent = max([v for k, v in scores.items() if "entail" in k] or [0.0])
            con = max([v for k, v in scores.items() if "contrad" in k] or [0.0])
            rows.append({"evidence_id": e.id, "entailment": ent, "contradiction": con})
        return {
            "max_entailment": max((r["entailment"] for r in rows), default=0.0),
            "max_contradiction": max((r["contradiction"] for r in rows), default=0.0),
            "per_evidence": rows,
        }


class EvidenceAwareRAG:
    def __init__(
        self,
        documents: list[dict[str, Any]],
        top_k: int = 5,
        min_retrieval_score: float = 0.08,
        generator_model: Optional[str] = None,
        nli_model: Optional[str] = None,
        min_entailment: float = 0.50,
        max_contradiction: float = 0.40,
        max_context_chars: int = 12000,
    ):
        self.retriever = TfidfRetriever(documents)
        self.top_k = top_k
        self.min_retrieval_score = min_retrieval_score
        self.min_entailment = min_entailment
        self.max_contradiction = max_contradiction
        self.max_context_chars = max_context_chars
        self.generator = None
        if generator_model:
            from transformers import pipeline
            self.generator = pipeline("text-generation", model=generator_model, device_map="auto")
        self.verifier = NliVerifier(nli_model) if nli_model else None

    @staticmethod
    def _abstention(reason: str, evidence: list[Evidence], verification=None) -> dict[str, Any]:
        return {
            "answer": "Insufficient evidence in the retrieved sources to answer reliably.",
            "abstained": True,
            "abstention_reason": reason,
            "evidence": [e.to_dict() for e in evidence],
            "verification": verification,
        }

    def _make_prompt(self, question: str, evidence: list[Evidence]) -> str:
        blocks, used = [], 0
        for i, e in enumerate(evidence, 1):
            block = f"[E{i}] source={e.source}; title={e.title}; id={e.id}\n{e.text}\n"
            if used + len(block) > self.max_context_chars:
                break
            blocks.append(block)
            used += len(block)
        context = "\n".join(blocks)
        return (
            "You are a biomedical research assistant. Use ONLY the evidence below. "
            "Do not use unsupported prior knowledge. If evidence is insufficient or conflicting, "
            "say exactly that the evidence is insufficient and do not guess. Cite each factual "
            "claim using evidence IDs such as [E1]. Distinguish evidence from inference. "
            "This is research assistance, not medical advice.\n\n"
            f"QUESTION:\n{question}\n\nEVIDENCE:\n{context}\n\n"
            "Answer concisely, with inline evidence citations. If unsupported, abstain.\nANSWER:"
        )

    def answer(self, question: str) -> dict[str, Any]:
        evidence = self.retriever.retrieve(question, self.top_k)
        eligible = [e for e in evidence if e.retrieval_score >= self.min_retrieval_score]
        if not eligible:
            return self._abstention("no_evidence_above_retrieval_threshold", evidence)

        if self.generator is None:
            return {
                "answer": "Generation disabled. Inspect the retrieved evidence and provide a validated answer manually.",
                "abstained": True,
                "abstention_reason": "generation_disabled",
                "evidence": [e.to_dict() for e in eligible],
                "verification": None,
            }

        prompt = self._make_prompt(question, eligible)
        result = self.generator(prompt, max_new_tokens=384, do_sample=False, return_full_text=False)
        answer = str(result[0].get("generated_text", "")).strip()
        verification = self.verifier.score(answer, eligible) if self.verifier else None

        # Conservative post-generation gate: do not automatically trust a fluent answer.
        if self.verifier:
            if verification["max_entailment"] < self.min_entailment:
                return self._abstention("insufficient_nli_support", eligible, verification)
            if verification["max_contradiction"] >= self.max_contradiction:
                return self._abstention("possible_evidence_contradiction", eligible, verification)

        # Citation presence is a traceability check, not proof of correctness.
        citation_ids = set(re.findall(r"\[(E\d+)\]", answer))
        if not citation_ids:
            return self._abstention("answer_has_no_evidence_citations", eligible, verification)

        return {
            "answer": answer,
            "abstained": False,
            "abstention_reason": None,
            "evidence": [e.to_dict() for e in eligible],
            "verification": verification,
        }
