# grounded-rag-docai

**Learning project** (personal OSS, not production, no employer work). It runs a small document-AI RAG pipeline end to end on real public PDFs:
OCR → BM25 / vector / hybrid (RRF) retrieval → cross-encoder re-ranking → answers with citations from a **real local LLM** → a
grounding eval with bootstrap CIs. Every number below comes from real runs on CPU. The runs are recorded in `results/` and replayed offline in CI.

## Pipeline (`grag/`)
| stage | what is used |
|---|---|
| corpus (`data/corpus.json`) | 6 public PDFs pinned by SHA-256: 5 arXiv papers (Transformer, BERT, Sentence-BERT, ColBERT, RAG) and NIST AI RMF 1.0 (US government work), 119 pages |
| extraction (`extract.py`) | `pdftotext` for born-digital pages. For **12 pages (pp. 2–3 of each PDF) treated as scans**: rasterised at 200 dpi, degraded (rotation ±1.5°, blur, speckle noise, JPEG q35), then OCR'd with **Tesseract 5.5**. The index contains the OCR text for those pages. |
| chunking (`text.py`) | 120-word windows with a stride of 80 (713 chunks). The 40-word overlap means any short evidence span lies wholly in one chunk. |
| BM25 (`bm25.py`) | pure-Python Okapi BM25 (k1 = 1.5, b = 0.75) |
| vector (`dense.py`) | `sentence-transformers/all-MiniLM-L6-v2`, cosine |
| hybrid (`fusion.py`) | reciprocal rank fusion (k = 60) of the BM25 and vector top-30 |
| re-rank | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the hybrid top-30 → top-10 |
| generation (`generate.py`) | **Qwen2.5-1.5B-Instruct Q4_K_M** in llama.cpp `llama-server` (b11552), temperature 0. Top-4 passages, output **grammar-constrained to JSON** `{answer, citations[]}` |
| faithfulness judge | `cross-encoder/nli-deberta-v3-base`: P(entailment) of each answer sentence, max over 2-sentence windows of the context passages, threshold 0.5 |

## Eval set
`data/questions.jsonl` has 40 hand-written questions, each with accepted answer strings and a verbatim **evidence span**. Gold passages are the chunks
whose indexed text contains that span. 10 of the 40 questions have gold evidence on an OCR'd page, and every evidence span survived OCR.
Metrics: Recall@1/5 and MRR@10 against gold chunks; answer accuracy (an accepted string appears in the answer); citation precision
(the share of cited passages that are gold); faithfulness (the share of answer sentences entailed by the context). CIs are percentile bootstrap
(2,000 resamples) over questions. Differences use a **paired** bootstrap over the same questions.

## Results (real run, 13 Oct 2026)
<!-- RESULTS -->
OCR on 12 simulated-scan pages: mean WER vs. pdftotext 0.146 [95% CI 0.089, 0.224], median 0.118, worst 0.491 (1706.03762 p.3); mean 3.2 s/page.

| system | n | Recall@1 | Recall@5 | MRR@10 | answer acc. | cites given | citation precision | faithfulness | retrieval p50/p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 40 | 0.40 [0.25, 0.55] | 0.82 [0.70, 0.93] | 0.56 [0.44, 0.68] | 0.78 [0.65, 0.90] | 1.00 [1.00, 1.00] | 0.47 [0.33, 0.62] (n=40) | 0.74 [0.60, 0.86] (n=40) | 1.4 / 2.3 |
| vector | 40 | 0.28 [0.15, 0.42] | 0.55 [0.40, 0.70] | 0.38 [0.26, 0.51] | 0.65 [0.50, 0.80] | 1.00 [1.00, 1.00] | 0.28 [0.16, 0.41] (n=40) | 0.62 [0.46, 0.77] (n=39) | 16.5 / 302.4 |
| hybrid | 40 | 0.30 [0.15, 0.45] | 0.62 [0.47, 0.78] | 0.44 [0.31, 0.57] | 0.70 [0.55, 0.85] | 1.00 [1.00, 1.00] | 0.33 [0.19, 0.47] (n=40) | 0.74 [0.59, 0.87] (n=39) | 18.6 / 303.8 |
| hybrid_rerank | 40 | 0.53 [0.38, 0.68] | 0.80 [0.68, 0.93] | 0.65 [0.53, 0.76] | 0.75 [0.62, 0.88] | 1.00 [1.00, 1.00] | 0.60 [0.45, 0.74] (n=40) | 0.72 [0.56, 0.85] (n=39) | 705.1 / 1261.4 |

| paired difference (per question) | MRR@10 diff [95% CI] | answer acc. diff [95% CI] |
|---|---|---|
| vector − bm25 | -0.18 [-0.33, -0.03] | -0.12 [-0.33, +0.05] |
| hybrid − bm25 | -0.12 [-0.23, -0.02] | -0.08 [-0.23, +0.07] |
| hybrid − vector | +0.06 [-0.04, +0.15] | +0.05 [-0.07, +0.17] |
| hybrid_rerank − hybrid | +0.21 [+0.08, +0.33] | +0.05 [-0.05, +0.17] |
| hybrid_rerank − bm25 | +0.09 [-0.03, +0.20] | -0.03 [-0.17, +0.12] |

Generation latency (all 160 calls): p50 3.5 s, p95 5.6 s. Questions whose gold evidence is absent from the indexed text: 0 [].
<!-- /RESULTS -->

**What the numbers say**
- **The cross-encoder re-ranker is the one clear, significant win.** It improved MRR@10 by +0.21 [+0.08, +0.33] over un-reranked hybrid, and the paired CI excludes 0.
  It also gave the best citation precision.
- **BM25 beat MiniLM vectors** on this corpus (vectors were −0.18 MRR [−0.33, −0.03]). The questions are about exact terms and numbers
  ("N = 6", "MRR@10", "340M"), and those suit lexical matching. **Plain RRF hybrid was worse than BM25 alone** here, because the weaker vector list drags it down.
  Hybrid + rerank vs BM25 is not significant.
- **Answer accuracy differences are not significant** between any pair (n = 40). Where the gold passage lands in the top 4, the 1.5B model usually answers correctly either way.
- **Citations:** a first prompt that asked for inline `[n]` citations (results in `results/runs_v1_zeroshot_prompt.jsonl`) got a citation on only
  14/160 answers. Constraining the output to a JSON schema with a `citations` field got one on every answer. Even so, many citations point at a
  neighbouring non-gold chunk, so citation precision stays moderate.
- **OCR:** mean WER vs. pdftotext is about 0.15, and it is worst on pages full of equations, tables or figures. Part of that "error" is reading-order
  differences in two-column layouts, not misread characters.

## Real vs. stubbed
| real | stubbed / simplified |
|---|---|
| public PDFs, checksum-pinned; Tesseract OCR; pdftotext | the "scans" are **simulated** degradations of born-digital pages, not paper scans |
| BM25, MiniLM embeddings, RRF, MiniLM cross-encoder | no vector DB (in-memory numpy); no ANN index |
| Qwen2.5-1.5B answers via llama.cpp, recorded verbatim | one model, one run, temperature 0 |
| NLI faithfulness judge (DeBERTa-v3-base) | NLI judge not calibrated against human labels |
| hand-written 40-question set with verbatim evidence | written by the author of the pipeline (selection bias); no human answer grading |

## Limits
- n = 40 questions, so the CIs are wide. Only the re-rank lift and the BM25 > vector gap are significant.
- Latency was measured on a **shared 8-vCPU box with heavy contention** (load averages 10–16 from other jobs). Treat it as relative only. Re-rank p50 is about 0.7 s
  for 30 pairs on CPU, and LLM generation is 3–4 s p50, p95 5.6 s, plus one 175 s outlier when the box was saturated.
- The faithfulness judge under-credits very short answers ("2048"). They are expanded to `The answer to the question "…" is 2048.`,
  which small NLI models often score as neutral. It also misses paraphrases (e.g. "heads" vs "parallel attention layers").
- Answer accuracy is substring matching, which can over-credit an answer that lists many options.
- OCR WER uses pdftotext output as the reference, which is itself imperfect.
- Corpus text, PDFs, scans and model weights are **not** committed. `results/runs.jsonl` keeps chunk ids, answers and judge scores only.

## Run it
```bash
pip install -e ".[dev,ocr]" && pytest -q   # unit tests, a real Tesseract OCR test, offline replay of the recorded runs
LLAMA_SERVER=.../llama-server GGUF=.../qwen2.5-1.5b-instruct-q4_k_m.gguf scripts/run_all.sh   # full real run
```
MIT licensed. Paper PDFs belong to their authors and are downloaded from arXiv/NIST at run time.
