"""Cited answer generation against llama.cpp's OpenAI-compatible llama-server."""
import json, re, time, urllib.request

SYS_V1 = ("You answer questions using ONLY the numbered passages. Cite every claim with the passage "
          "number in square brackets, e.g. [2]. Answer in 1-2 short sentences. If the passages do not "
          "contain the answer, reply exactly: I don't know.")
# v2: v1 got citations on only 14/160 answers, so v2 adds a worked example and a hard format rule.
SYS = ("You answer questions using ONLY the numbered passages.\n"
       "Rules: answer in one or two full sentences; END EVERY SENTENCE WITH ITS SOURCE PASSAGE NUMBER "
       "in square brackets, like [2]; never answer without a citation. If the passages do not contain the "
       "answer, reply exactly: I don't know.\n"
       "Example. Passages: [1] The cat sat on the mat. [2] Paris is the capital of France.\n"
       "Question: What is the capital of France?\nAnswer: The capital of France is Paris [2].")

def build_prompt(question, passages):
    ctx = "\n\n".join(f"[{i}] ({p['doc']} p.{p['page']}) {p['text']}" for i, p in enumerate(passages, 1))
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {question}"}]

def parse_citations(answer, n):
    """Citation numbers in [i] / [i, j] form that point at a real passage (1..n)."""
    out = []
    for grp in re.findall(r"\[([\d,\s]+)\]", answer):
        for x in re.findall(r"\d+", grp):
            i = int(x)
            if 1 <= i <= n and i not in out: out.append(i)
    return out

def hypotheses(question, answer):
    """NLI hypotheses for an answer. Very short answers ("2048") are not checkable sentences on their
    own, so they are expanded to 'The answer to the question "<q>" is <a>.'"""
    cl = claims(answer)
    if not cl and re.sub(r"\[[\d,\s]+\]", "", answer).strip():
        cl = [re.sub(r"\[[\d,\s]+\]", "", answer).strip()]
    return [c if len(c.split()) >= 5 else f'The answer to the question "{question}" is {c.rstrip(".")}.' for c in cl]

def claims(answer):
    """Split an answer into sentences with citation markers stripped."""
    body = re.sub(r"\[[\d,\s]+\]", "", answer)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 1]

# v3 (used for the reported results): grammar-constrained JSON so a citation list is always present.
SCHEMA = {"type": "object", "properties": {
    "answer": {"type": "string"},
    "citations": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 4}, "maxItems": 4}},
    "required": ["answer", "citations"]}
SYS_JSON = ("You answer questions using ONLY the numbered passages. Reply as JSON: "
            '{"answer": "<one complete sentence that restates the question>", "citations": [<numbers of the passages that support the answer>]}. '
            "If the passages do not contain the answer, use answer \"I don't know.\" and citations [].")

def build_prompt_json(question, passages):
    m = build_prompt(question, passages); m[0]["content"] = SYS_JSON; return m

def parse_json_answer(content, n):
    """-> (answer text, citation numbers within 1..n). Falls back to inline [i] parsing."""
    try:
        d = json.loads(content); ans = str(d.get("answer", "")).strip()
        cites = [int(c) for c in d.get("citations", []) if isinstance(c, (int, float)) and 1 <= int(c) <= n]
        return ans, list(dict.fromkeys(cites))
    except (ValueError, TypeError, AttributeError):
        return content, parse_citations(content, n)

def chat(base_url, messages, max_tokens=160, timeout=600, schema=None):
    body = {"messages": messages, "temperature": 0, "seed": 0, "max_tokens": max_tokens, "cache_prompt": True}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "cited_answer", "schema": schema}}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r["choices"][0]["message"]["content"].strip(), time.time() - t, r.get("usage", {})
