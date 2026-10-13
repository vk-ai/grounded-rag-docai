#!/usr/bin/env bash
# Real end-to-end run. Needs: tesseract, poppler-utils, llama.cpp llama-server, a Qwen2.5-1.5B-Instruct GGUF.
set -euo pipefail
: "${LLAMA_SERVER:?path to llama-server}" "${GGUF:?path to qwen2.5-1.5b-instruct-q4_k_m.gguf}"
pip install -e ".[dev,models]"
python scripts/build_corpus.py
"$LLAMA_SERVER" -m "$GGUF" --port 8092 -c 8192 --jinja --no-webui > llama.log 2>&1 &
trap 'kill $!' EXIT
until curl -sf localhost:8092/health >/dev/null; do sleep 1; done
python scripts/run_eval.py
python -m grag.report --write
