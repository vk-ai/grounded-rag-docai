import math
from collections import Counter
from .text import tokens

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter(w for t in self.tf for w in t)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, q):
        qt = tokens(q); out = []
        for tf, L in zip(self.tf, self.len):
            s = 0.0
            for w in qt:
                f = tf.get(w)
                if f: s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out

    def search(self, q, k=10):
        s = self.scores(q)
        return sorted(range(len(s)), key=lambda i: -s[i])[:k]
