"""Text utils: normalisation, tokenising, chunking, OCR error rates. Pure Python."""
import re, unicodedata

STOP = set("a an the of to in and or for on with by is are was were be been it this that as at from which what how why when who does do did into their its than".split())

def norm(s):
    s = unicodedata.normalize("NFKC", s).replace("\u00ad", "")
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    return re.sub(r"\s+", " ", s).strip()

def tokens(s, stop=True):
    t = re.findall(r"[a-z0-9]+(?:[.@][a-z0-9]+)*", norm(s).lower())
    return [w for w in t if not (stop and w in STOP)]

def chunk_pages(doc_id, pages, size=120, stride=80):
    """Overlapping word windows over the whole document; each chunk keeps its start page.
    With overlap = size - stride = 40 words, any evidence span of <= 40 words lies wholly in some chunk."""
    words = []
    for pno, txt in enumerate(pages, 1):
        words += [(w, pno) for w in norm(txt).split(" ") if w]
    out, i = [], 0
    while i < len(words):
        win = words[i:i + size]
        out.append({"id": f"{doc_id}#{len(out)}", "doc": doc_id, "page": win[0][1],
                    "text": " ".join(w for w, _ in win)})
        if i + size >= len(words): break
        i += stride
    return out

def contains(chunk_text, evidence):
    return norm(evidence).lower() in norm(chunk_text).lower()

def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def error_rate(ref, hyp):
    """Word error rate of hyp vs ref (both normalised, lowercased)."""
    r, h = norm(ref).lower().split(), norm(hyp).lower().split()
    return _lev(r, h) / max(1, len(r))
