"""Fetch PDFs, extract text, OCR the 'scanned' pages, chunk. Writes .data/chunks.json + results/ocr.json."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grag.fetch import fetch, manifest
from grag.extract import pdf_pages, scan_and_ocr
from grag.text import chunk_pages, error_rate

SCANNED = {d: [2, 3] for d in [m["id"] for m in manifest()]}  # 12 pages treated as scans
data = Path(".data"); pdfdir = fetch(data / "pdf"); scans = data / "scans"; scans.mkdir(exist_ok=True)
chunks, ocr = [], []
for m in manifest():
    pages = pdf_pages(pdfdir / f"{m['id']}.pdf")
    for p in SCANNED[m["id"]]:
        t = time.time(); txt = scan_and_ocr(pdfdir / f"{m['id']}.pdf", p, seed=p * 7 + len(m["id"]), keep_dir=scans)
        ocr.append({"doc": m["id"], "page": p, "wer_vs_pdftotext": round(error_rate(pages[p - 1], txt), 4),
                    "ref_words": len(pages[p - 1].split()), "ocr_s": round(time.time() - t, 2)})
        pages[p - 1] = txt  # the index sees the OCR text, not the digital text, for these pages
        print(ocr[-1], flush=True)
    chunks += chunk_pages(m["id"], pages)
(data / "chunks.json").write_text(json.dumps(chunks))
Path("results").mkdir(exist_ok=True)
Path("results/ocr.json").write_text(json.dumps(ocr, indent=1))
print(len(chunks), "chunks")
