def rrf(rankings, k=60, top=None):
    """Reciprocal rank fusion (Cormack et al. 2009). rankings: lists of ids, best first."""
    s = {}
    for r in rankings:
        for rank, d in enumerate(r, 1):
            s[d] = s.get(d, 0.0) + 1.0 / (k + rank)
    out = sorted(s, key=lambda d: (-s[d], str(d)))
    return out[:top] if top else out
