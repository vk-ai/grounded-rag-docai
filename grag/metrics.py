from .text import norm

def recall_at_k(ranked, gold, k): return float(any(c in gold for c in ranked[:k]))

def mrr(ranked, gold, k=10):
    for i, c in enumerate(ranked[:k], 1):
        if c in gold: return 1.0 / i
    return 0.0

def answer_correct(answer, accepted):
    a = norm(answer).lower()
    return float(any(norm(x).lower() in a for x in accepted))

def citation_precision(cited_ids, gold):
    """Share of cited passages that contain the gold evidence. None if nothing was cited."""
    return None if not cited_ids else sum(c in gold for c in cited_ids) / len(cited_ids)
