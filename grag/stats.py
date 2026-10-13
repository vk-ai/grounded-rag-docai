import random

def mean(x): return sum(x) / len(x) if x else float("nan")

def bootstrap_ci(x, n=2000, seed=0, alpha=0.05):
    if not x: return (float("nan"),) * 2
    rng = random.Random(seed); m = len(x)
    bs = sorted(mean([x[rng.randrange(m)] for _ in range(m)]) for _ in range(n))
    return bs[int(alpha / 2 * n)], bs[int((1 - alpha / 2) * n) - 1]

def paired_diff_ci(a, b, n=2000, seed=0):
    """Bootstrap CI of mean(a - b) over paired per-question values."""
    return bootstrap_ci([x - y for x, y in zip(a, b)], n, seed)

def percentile(x, p):
    s = sorted(x); return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))] if s else float("nan")
