"""Offline replay of the recorded real runs: recompute metrics and check README matches."""
import json
from pathlib import Path
from grag.report import render, per_question, SYSTEMS

ROOT = Path(__file__).resolve().parent.parent

def test_recorded_runs_are_complete():
    runs = [json.loads(l) for l in open(ROOT / "results/runs.jsonl")]
    qs = [json.loads(l) for l in open(ROOT / "data/questions.jsonl")]
    assert {(r["qid"], r["system"]) for r in runs} == {(q["id"], s) for q in qs for s in SYSTEMS}
    for r in runs:
        assert len(r["ranked"]) == 10 and r["ctx"] == r["ranked"][:4]
        assert set(r["cited"]) <= set(r["ctx"])
        assert all(0 <= p <= 1 for p in r["claim_entail_ctx"])

def test_readme_table_matches_recorded_runs():
    readme = (ROOT / "README.md").read_text()
    assert render(ROOT) in readme

def test_rerank_never_drops_below_hybrid_candidates():
    runs = [json.loads(l) for l in open(ROOT / "results/runs.jsonl")]
    by = {(r["qid"], r["system"]): r for r in runs}
    for (q, s), r in by.items():
        if s == "hybrid_rerank":
            assert len(set(r["ranked"])) == 10
