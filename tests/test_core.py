import json
from pathlib import Path
import pytest
from grag.bm25 import BM25
from grag.fusion import rrf
from grag.text import chunk_pages, contains, error_rate, norm, tokens
from grag.metrics import recall_at_k, mrr, answer_correct, citation_precision
from grag.generate import parse_citations, claims, build_prompt
from grag.stats import bootstrap_ci, paired_diff_ci, percentile

ROOT = Path(__file__).resolve().parent.parent

def test_bm25_ranks_exact_term_first():
    docs = ["the cat sat on the mat", "dogs chase cats", "late interaction with MaxSim operator"]
    assert BM25(docs).search("MaxSim late interaction", 1) == [2]

def test_bm25_idf_downweights_common_terms():
    bm = BM25(["model alpha", "model beta", "model gamma"])
    s = bm.scores("model beta")
    assert s[1] > s[0] == s[2]

def test_rrf_rewards_agreement():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] == ["a", "b"] or rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] == ["b", "a"]
    assert rrf([["x", "b"], ["y", "b"]])[0] == "b"

def test_chunk_overlap_guarantees_evidence_inside_one_chunk():
    words = [f"w{i}" for i in range(500)]
    pages = [" ".join(words[:250]), " ".join(words[250:])]
    ch = chunk_pages("d", pages, size=120, stride=80)
    for start in range(0, 490, 7):
        ev = " ".join(words[start:start + 10])
        assert any(contains(c["text"], ev) for c in ch), start
    assert ch[0]["page"] == 1 and ch[-1]["page"] == 2

def test_norm_and_tokens():
    assert norm("re-\ntrieval  x") == "retrieval x"
    assert "mrr@10" in tokens("MRR@10 on MS MARCO")
    assert "the" not in tokens("the model")

def test_error_rate():
    assert error_rate("a b c d", "a b c d") == 0
    assert error_rate("a b c d", "a x c d") == 0.25

def test_metrics():
    assert recall_at_k(["x", "g"], {"g"}, 1) == 0 and recall_at_k(["x", "g"], {"g"}, 2) == 1
    assert mrr(["x", "g"], {"g"}) == 0.5 and mrr(["x"], {"g"}) == 0
    assert answer_correct("It uses 8 heads.", ["8", "eight"]) == 1
    assert citation_precision([], {"g"}) is None
    assert citation_precision(["g", "x"], {"g"}) == 0.5

def test_citation_parsing_ignores_out_of_range():
    assert parse_citations("A [1]. B [2, 3]. C [9].", 4) == [1, 2, 3]
    assert claims("Six layers [1]. It uses Adam [2].") == ["Six layers .", "It uses Adam ."] or len(claims("Six layers [1]. It uses Adam [2].")) == 2

def test_prompt_numbers_passages():
    m = build_prompt("q?", [{"doc": "d", "page": 3, "text": "t1"}, {"doc": "d", "page": 4, "text": "t2"}])
    assert "[1] (d p.3) t1" in m[1]["content"] and "[2] (d p.4) t2" in m[1]["content"]

def test_bootstrap():
    lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < 0.5 < hi
    assert paired_diff_ci([1] * 10, [1] * 10) == (0, 0)
    assert percentile([1, 2, 3, 4, 5], 50) == 3

def test_question_set_well_formed():
    qs = [json.loads(l) for l in open(ROOT / "data/questions.jsonl")]
    assert len(qs) == 40 and len({q["id"] for q in qs}) == 40
    docs = {d["id"] for d in json.loads((ROOT / "data/corpus.json").read_text())}
    for q in qs:
        assert q["doc"] in docs and q["answer"] and q["evidence"]

def test_corpus_manifest_is_pinned():
    for d in json.loads((ROOT / "data/corpus.json").read_text()):
        assert len(d["sha256"]) == 64 and d["url"].startswith("https://")

def test_json_answer_parsing_and_fallback():
    from grag.generate import parse_json_answer
    assert parse_json_answer('{"answer": "Six [1].", "citations": [1, 1, 9, 2]}', 4) == ("Six [1].", [1, 2])
    assert parse_json_answer("Six layers [3].", 4) == ("Six layers [3].", [3])

def test_short_answers_become_checkable_hypotheses():
    from grag.generate import hypotheses
    assert hypotheses("How many heads?", "8") == ['The answer to the question "How many heads?" is 8.']
    assert hypotheses("q", "The model uses eight heads [1].") == ["The model uses eight heads ."]
