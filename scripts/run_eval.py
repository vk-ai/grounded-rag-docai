"""Real run: 4 retrievers x 40 questions -> top-k, cited answer from llama.cpp, NLI faithfulness.
Writes results/runs.jsonl (chunk ids + answers + judgements; no corpus text)."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grag.bm25 import BM25
from grag.dense import Dense, Reranker, NLIJudge
from grag.fusion import rrf
from grag.generate import build_prompt_json, chat, parse_json_answer, hypotheses, SCHEMA
from grag.text import contains

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8092"); ap.add_argument("--model", default="qwen2.5-1.5b-instruct-q4_k_m")
ap.add_argument("--k-ctx", type=int, default=4); ap.add_argument("--out", default="results/runs.jsonl")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
chunks = json.loads(Path(".data/chunks.json").read_text()); texts = [c["text"] for c in chunks]
qs = [json.loads(l) for l in open("data/questions.jsonl")][: a.limit or None]
t = time.time(); bm = BM25(texts); dn = Dense(texts); rr = Reranker(); nli = NLIJudge("cross-encoder/nli-deberta-v3-base")
print("index built", round(time.time() - t, 1), "s", flush=True)
done = set()
if Path(a.out).exists():
    done = {(r["qid"], r["system"]) for r in map(json.loads, open(a.out))}
out = open(a.out, "a")
for q in qs:
    gold = [chunks[i]["id"] for i, c in enumerate(texts) if contains(c, q["evidence"])]
    lat = {}
    t = time.perf_counter(); b = bm.search(q["question"], 30); lat["bm25"] = time.perf_counter() - t
    t = time.perf_counter(); d = dn.search(q["question"], 30); lat["vector"] = time.perf_counter() - t
    t = time.perf_counter(); h = rrf([b, d], top=30); lat["hybrid"] = lat["bm25"] + lat["vector"] + time.perf_counter() - t
    t = time.perf_counter(); hr = rr.rerank(q["question"], h, texts, 10); lat["hybrid_rerank"] = lat["hybrid"] + time.perf_counter() - t
    for name, ranked in [("bm25", b), ("vector", d), ("hybrid", h), ("hybrid_rerank", hr)]:
        if (q["id"], name) in done: continue
        ranked = ranked[:10]; ctx = [chunks[i] for i in ranked[: a.k_ctx]]
        raw, gen_s, usage = chat(a.base_url, build_prompt_json(q["question"], ctx), schema=SCHEMA)
        ans, cites = parse_json_answer(raw, len(ctx))
        cl = hypotheses(q["question"], ans)
        abstain = ans.lower().startswith("i don't know") or not ans
        ent = [round(max(nli.entail_windows(ctx[i - 1]["text"], c) for i in cites), 4) for c in cl] if cites and not abstain else []
        ent_ctx = [round(max(nli.entail_windows(p["text"], c) for p in ctx), 4) for c in cl] if not abstain else []
        rec = {"qid": q["id"], "system": name, "gold": gold, "ranked": [chunks[i]["id"] for i in ranked],
               "ctx": [c["id"] for c in ctx], "answer": ans, "cited": [ctx[i - 1]["id"] for i in cites],
               "claim_entail": ent, "claim_entail_ctx": ent_ctx, "n_claims": len(cl), "abstain": abstain, "retrieval_s": round(lat[name], 4),
               "gen_s": round(gen_s, 2), "usage": usage, "model": a.model}
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(q["id"], name, round(gen_s, 1), ans[:90].replace("\n", " "), flush=True)
