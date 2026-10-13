"""Compute the README tables from recorded runs (results/runs.jsonl). Pure Python, used by CI."""
import json, sys
from pathlib import Path
from .metrics import recall_at_k, mrr, answer_correct, citation_precision
from .stats import mean, bootstrap_ci, paired_diff_ci, percentile

SYSTEMS = ["bm25", "vector", "hybrid", "hybrid_rerank"]
THRESH = 0.5

def load(path):
    return [json.loads(l) for l in open(path)]

def per_question(runs, questions):
    acc = {q["id"]: q["answer"] for q in questions}
    out = {s: {} for s in SYSTEMS}
    for r in runs:
        g = set(r["gold"])
        faith = [float(p > THRESH) for p in r.get("claim_entail_ctx", [])]
        out[r["system"]][r["qid"]] = {
            "r1": recall_at_k(r["ranked"], g, 1), "r5": recall_at_k(r["ranked"], g, 5),
            "ctx_hit": recall_at_k(r["ctx"], g, len(r["ctx"])), "mrr": mrr(r["ranked"], g, 10),
            "acc": answer_correct(r["answer"], acc[r["qid"]]), "cited": float(bool(r["cited"])),
            "cprec": citation_precision(r["cited"], g), "abstain": float(r["abstain"]),
            "faith": mean(faith) if faith else None, "ret_s": r["retrieval_s"], "gen_s": r["gen_s"]}
    return out

def fmt(x, ci):
    return f"{x:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"

def col(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    return v, (fmt(mean(v), bootstrap_ci(v)) if v else "n/a")

def tables(runs, questions):
    pq = per_question(runs, questions); qids = [q["id"] for q in questions if q["id"] in pq["bm25"]]
    lines = ["| system | n | Recall@1 | Recall@5 | MRR@10 | answer acc. | cites given | citation precision | faithfulness | retrieval p50/p95 ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        rows = [pq[s][q] for q in qids if q in pq[s]]
        if not rows: continue
        c = {k: col(rows, k)[1] for k in ["r1", "r5", "mrr", "acc", "cited", "cprec", "faith"]}
        rt = [r["ret_s"] * 1000 for r in rows]
        lines.append(f"| {s} | {len(rows)} | {c['r1']} | {c['r5']} | {c['mrr']} | {c['acc']} | {c['cited']} | "
                     f"{c['cprec']} (n={len(col(rows,'cprec')[0])}) | {c['faith']} (n={len(col(rows,'faith')[0])}) | "
                     f"{percentile(rt,50):.1f} / {percentile(rt,95):.1f} |")
    lines += ["", "| paired difference (per question) | MRR@10 diff [95% CI] | answer acc. diff [95% CI] |", "|---|---|---|"]
    for a, b in [("vector", "bm25"), ("hybrid", "bm25"), ("hybrid", "vector"), ("hybrid_rerank", "hybrid"), ("hybrid_rerank", "bm25")]:
        common = [q for q in qids if q in pq[a] and q in pq[b]]
        cells = []
        for k in ["mrr", "acc"]:
            x = [pq[a][q][k] for q in common]; y = [pq[b][q][k] for q in common]
            lo, hi = paired_diff_ci(x, y); cells.append(f"{mean(x)-mean(y):+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append(f"| {a} − {b} | {cells[0]} | {cells[1]} |")
    gen = [r["gen_s"] for s in SYSTEMS for r in pq[s].values()]
    nogold = [q for q in qids if not runs_gold(runs, q)]
    lines += ["", f"Generation latency (all {len(gen)} calls): p50 {percentile(gen,50):.1f} s, p95 {percentile(gen,95):.1f} s. "
              f"Questions whose gold evidence is absent from the indexed text: {len(nogold)} {nogold}."]
    return "\n".join(lines)

def runs_gold(runs, qid):
    return next((r["gold"] for r in runs if r["qid"] == qid), [])

def ocr_table(ocr):
    w = [o["wer_vs_pdftotext"] for o in ocr]
    lo, hi = bootstrap_ci(w)
    return (f"OCR on {len(ocr)} simulated-scan pages: mean WER vs. pdftotext {mean(w):.3f} [95% CI {lo:.3f}, {hi:.3f}], "
            f"median {percentile(w,50):.3f}, worst {max(w):.3f} ({max(ocr, key=lambda o: o['wer_vs_pdftotext'])['doc']} "
            f"p.{max(ocr, key=lambda o: o['wer_vs_pdftotext'])['page']}); mean {mean([o['ocr_s'] for o in ocr]):.1f} s/page.")

def render(root="."):
    root = Path(root)
    qs = [json.loads(l) for l in open(root / "data/questions.jsonl")]
    return ocr_table(json.loads((root / "results/ocr.json").read_text())) + "\n\n" + tables(load(root / "results/runs.jsonl"), qs)

def update_readme(root="."):
    p = Path(root) / "README.md"; s = p.read_text()
    a, b = s.index("<!-- RESULTS -->") + len("<!-- RESULTS -->"), s.index("<!-- /RESULTS -->")
    p.write_text(s[:a] + "\n" + render(root) + "\n" + s[b:])

if __name__ == "__main__":
    update_readme() if "--write" in sys.argv else print(render())
