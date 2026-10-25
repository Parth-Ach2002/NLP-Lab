"""Precision@5: TF-IDF vs semantic embeddings.

Setup: pool of papers from several arXiv categories. Each query is the TITLE of a
held-out paper; a retrieved paper counts as relevant if it has the same category.
(Category is a proxy for relevance - state this limitation in your report.)

Run:  python evaluate.py            # fetches data from arXiv (cached in eval_pool.json)
"""
import json
import os
import random

import numpy as np

import nlp

CATS = ["cs.CL", "cs.CV", "cs.IR", "cs.CR", "q-bio.NC", "stat.ML"]
PER_CAT = 40
N_QUERIES_PER_CAT = 10
CACHE = "eval_pool.json"


def load_pool():
    if os.path.exists(CACHE):
        print("Using cached pool:", CACHE, flush=True)
        return json.load(open(CACHE))
    pool = []
    for c in CATS:
        print(f"Fetching {PER_CAT} papers from arXiv for {c} ...", flush=True)
        for attempt in range(3):
            try:
                pool += nlp.fetch_arxiv(search_query=f"cat:{c}", max_results=PER_CAT)
                break
            except Exception as e:
                print(f"  attempt {attempt + 1} failed: {e}", flush=True)
                nlp.polite_sleep(5)
        nlp.polite_sleep(3)
    json.dump(pool, open(CACHE, "w"))
    return pool


def evaluate(pool, embedder=None, k=5, seed=0):
    rng = random.Random(seed)
    by_cat = {}
    for i, p in enumerate(pool):
        by_cat.setdefault(p["category"], []).append(i)

    # pick the queries first, then embed everything ONCE (the slow step)
    picks = [(cat, qi) for cat, idxs in by_cat.items()
             for qi in rng.sample(idxs, min(N_QUERIES_PER_CAT, len(idxs)))]
    print(f"Embedding {len(pool)} papers and {len(picks)} queries (one time) ...", flush=True)
    P = nlp.embed([nlp.paper_text(p) for p in pool], embedder)
    Q = nlp.embed([pool[qi]["title"] for _, qi in picks], embedder)

    res = {"tfidf": [], "semantic": []}
    for n, ((cat, qi), q) in enumerate(zip(picks, Q), 1):
        keep = np.array([j for j in range(len(pool)) if j != qi])
        is_rel = lambda j: pool[j]["category"] == cat
        sem = P[keep] @ q
        tf = nlp.tfidf_scores(pool[qi]["title"], [pool[j] for j in keep])
        for m, s in (("semantic", sem), ("tfidf", tf)):
            top = keep[np.argsort(-s)[:k]]
            res[m].append(sum(is_rel(j) for j in top) / k)
        if n % 10 == 0:
            print(f"  {n}/{len(picks)} queries done", flush=True)
    return {m: float(np.mean(v)) for m, v in res.items()}, len(picks)


if __name__ == "__main__":
    pool = load_pool()
    print(f"{len(pool)} papers loaded. Loading model (first run downloads ~90 MB) ...", flush=True)
    scores, n = evaluate(pool)
    print(f"\n{len(pool)} papers, {n} queries")
    for m, s in scores.items():
        print(f"Precision@5 {m:9s}: {s:.3f}")
