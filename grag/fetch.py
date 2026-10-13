"""Download the checksum-pinned public corpus."""
import hashlib, json, os, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
def manifest():
    return json.loads((ROOT / "data/corpus.json").read_text())

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def fetch(dest):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for d in manifest():
        p = dest / f"{d['id']}.pdf"
        if not p.exists() or sha256(p) != d["sha256"]:
            req = urllib.request.Request(d["url"], headers={"User-Agent": "grounded-rag-docai/0.1"})
            p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        got = sha256(p)
        if got != d["sha256"]:
            sys.exit(f"checksum mismatch for {d['id']}: {got}")
    return dest

if __name__ == "__main__":
    print(fetch(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GRAG_DATA", ".data/pdf")))
