"""Dense retrieval + cross-encoder rerank (sentence-transformers; optional dependency)."""
EMB = "sentence-transformers/all-MiniLM-L6-v2"
CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI = "cross-encoder/nli-deberta-v3-xsmall"

class Dense:
    def __init__(self, docs, model=EMB):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, device="cpu")
        self.E = self.m.encode(docs, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    def search(self, q, k=10):
        import numpy as np
        s = self.E @ self.m.encode([q], normalize_embeddings=True)[0]
        return [int(i) for i in np.argsort(-s)[:k]]

class Reranker:
    def __init__(self, model=CE):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model, device="cpu")
    def rerank(self, q, ids, texts, k):
        s = self.m.predict([(q, texts[i]) for i in ids])
        return [i for _, i in sorted(zip(-s, ids))][:k]

class NLIJudge:
    """P(entailment) of a claim given premise text, via a small NLI cross-encoder."""
    def __init__(self, model=NLI):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model, device="cpu")
        lab = {v.lower(): k for k, v in self.m.model.config.id2label.items()}
        self.ent = lab["entailment"]
    def entail_windows(self, passage, claim):
        """Max P(entailment) over 2-sentence windows of the passage. Whole 120-word noisy PDF chunks
        make the small NLI model say 'neutral' even when the fact is there, so we score windows."""
        import re
        s = re.split(r"(?<=[.!?])\s+", passage)
        return max(self.entail(" ".join(s[i:i + 2]), claim) for i in range(max(1, len(s) - 1)))

    def entail(self, premise, claim):
        import numpy as np
        logits = self.m.predict([(premise, claim)])[0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        return float(p[self.ent])
