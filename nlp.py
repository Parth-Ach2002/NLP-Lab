"""Core NLP logic: fetch papers, rank them (TF-IDF vs embeddings), summarize."""
import os
import re
import time
import xml.etree.ElementTree as ET

import numpy as np
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

ARXIV_URL = "https://export.arxiv.org/api/query"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = os.environ.get("MODEL_CACHE", os.path.join(os.path.dirname(__file__), "model_cache"))
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}

_embedder = None


# ---------- 1. Fetching papers from arXiv ----------
def query_to_keywords(query, max_words=8):
    """'How can language models detect sarcasm?' -> ['language','models','detect','sarcasm']"""
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", query.lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return list(dict.fromkeys(words))[:max_words]  # unique, keep order


def parse_arxiv(xml_text):
    root = ET.fromstring(xml_text)
    papers = []
    for e in root.findall("a:entry", NS):
        clean = lambda s: re.sub(r"\s+", " ", s or "").strip()
        cat = e.find("x:primary_category", NS)
        papers.append({
            "title": clean(e.findtext("a:title", "", NS)),
            "abstract": clean(e.findtext("a:summary", "", NS)),
            "url": e.findtext("a:id", "", NS),
            "published": (e.findtext("a:published", "", NS) or "")[:10],
            "authors": [clean(a.findtext("a:name", "", NS)) for a in e.findall("a:author", NS)],
            "category": cat.get("term") if cat is not None else "",
        })
    return papers


def fetch_arxiv(query=None, max_results=60, search_query=None):
    """Fetch a candidate pool. Pass `query` (natural language) or a raw arXiv `search_query`."""
    if search_query is None:
        kws = query_to_keywords(query)
        if not kws:
            return []
        search_query = "all:(" + " OR ".join(kws) + ")"
    params = {"search_query": search_query, "start": 0, "max_results": max_results,
              "sortBy": "relevance"}
    r = requests.get(ARXIV_URL, params=params, timeout=30,
                     headers={"User-Agent": "nlp-paper-recommender/1.0 (student project)"})
    r.raise_for_status()
    return parse_arxiv(r.text)


# ---------- 2. Embeddings ----------
def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(MODEL_NAME, cache_dir=MODEL_CACHE)
    return _embedder


def embed(texts, embedder=None):
    """Return L2-normalised vectors, so dot product == cosine similarity."""
    model = embedder or get_embedder()
    vecs = np.array(list(model.embed(list(texts))), dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)


# ---------- 3. Ranking ----------
def paper_text(p):
    return p["title"] + ". " + p["abstract"]


def tfidf_scores(query, papers):
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X = vec.fit_transform([paper_text(p) for p in papers] + [query])
    return (X[:-1] @ X[-1].T).toarray().ravel()  # cosine (rows are L2-normalised)


def semantic_scores(query, papers, embedder=None):
    P = embed([paper_text(p) for p in papers], embedder)
    q = embed([query], embedder)[0]
    return P @ q


def rank(query, papers, method="semantic", embedder=None, top_k=10):
    """Return top_k papers (copies) with a 'score' field, best first."""
    if not papers:
        return []
    s = tfidf_scores(query, papers) if method == "tfidf" else semantic_scores(query, papers, embedder)
    order = np.argsort(-s)[:top_k]
    return [dict(papers[i], score=float(s[i])) for i in order]


def diversify(query, papers, embedder=None, top_k=5, lam=0.7):
    """Optional MMR: balance relevance to the query with novelty vs. already-picked papers."""
    P = embed([paper_text(p) for p in papers], embedder)
    q = embed([query], embedder)[0]
    rel = P @ q
    chosen, rest = [], list(range(len(papers)))
    while rest and len(chosen) < top_k:
        def mmr(i):
            red = max((P[i] @ P[j] for j in chosen), default=0.0)
            return lam * rel[i] - (1 - lam) * red
        best = max(rest, key=mmr)
        chosen.append(best)
        rest.remove(best)
    return [dict(papers[i], score=float(rel[i])) for i in chosen]


# ---------- 4. Extractive summarizer ----------
def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 20]


def summarize(abstract, query=None, n=3, embedder=None, query_weight=0.4):
    """Pick the n most 'central' (and query-relevant) sentences, shown in original order."""
    sents = split_sentences(abstract)
    if len(sents) <= n:
        return " ".join(sents) if sents else abstract
    S = embed(sents, embedder)
    centroid = S.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    score = S @ centroid
    if query:
        score = (1 - query_weight) * score + query_weight * (S @ embed([query], embedder)[0])
    keep = sorted(np.argsort(-score)[:n])
    return " ".join(sents[i] for i in keep)


# ---------- 5. Evaluation ----------
def precision_at_k(ranked, relevant_fn, k=5):
    top = ranked[:k]
    return sum(1 for p in top if relevant_fn(p)) / k if top else 0.0


def polite_sleep(seconds=3.0):
    time.sleep(seconds)
