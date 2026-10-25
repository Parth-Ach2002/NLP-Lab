# Research Paper Recommender (with extractive summarizer)

**Research question:** do semantic sentence embeddings retrieve relevant papers better than the TF-IDF baseline from Practical 2? Measured with Precision@5.

## How it works (pipeline)
```
your question -> keywords -> arXiv API (60 candidate papers)
              -> rank by TF-IDF  OR  rank by embeddings -> top 5
              -> extractive summary of each abstract -> Streamlit page
```

## Files
| File | Job |
|---|---|
| `nlp.py` | all NLP logic |
| `app.py` | Streamlit web page |
| `evaluate.py` | Precision@5 experiment, TF-IDF vs embeddings |
| `prefetch.py` | downloads the model at build time |
| `requirements.txt`, `render.yaml` | deployment |

## Code explained, with simple examples

**1. `query_to_keywords`** – arXiv's API is keyword-based, so we strip stopwords.
`"How can language models detect sarcasm?"` -> `['language','models','detect','sarcasm']` -> arXiv query `all:(language OR models OR detect OR sarcasm)`. This gives a *pool* of candidates; the NLP then re-ranks them. (Same preprocessing idea as Lab 1.)

**2. `fetch_arxiv` / `parse_arxiv`** – arXiv replies in XML (Atom). We pull out title, abstract, link, date, authors, category.

**3. `tfidf_scores` (baseline, Lab 2)** – each paper and the query become TF-IDF vectors; score = cosine similarity. Weakness: it only matches *shared words*. "Irony detection" scores ~0 against "sarcasm".

**4. `embed` + `semantic_scores` (the new idea)** – a pretrained model (`all-MiniLM-L6-v2`, via `fastembed`) maps a text to a 384-number vector where similar *meanings* are close. Like Word2Vec from Lab 3, but for whole sentences. Vectors are normalised, so dot product = cosine similarity.
Example: "app crashes on launch" and "program closes right after opening" share no words, but get a high score.

**5. `rank`** – sorts papers by score, returns the top k.

**6. `diversify` (optional, MMR)** – picks papers one by one, scoring `0.7*relevance - 0.3*similarity to papers already picked`, so you don't get 5 near-duplicates.

**7. `summarize` (extractive)** – splits the abstract into sentences, embeds each, then scores each sentence:
`score = 0.6*(similarity to the abstract's average vector) + 0.4*(similarity to your question)`.
It keeps the top 3 sentences in original order. Extractive = it copies real sentences, never invents text, and needs no big model (important for Render's 512 MB free tier).

**8. `evaluate.py`** – pool = 40 papers from each of 6 arXiv categories. Query = the title of a paper; a result is "relevant" if it has the same category. Precision@5 = (relevant in top 5)/5, averaged over 60 queries. Example: 4 of top 5 are same-category -> 0.8.
*Limitation to state in your report:* category is only a proxy for relevance; it's not human judgement. Title-as-query is an easier task than natural questions, so also show a few manual examples in the app's side-by-side mode.

## Run locally
```
pip install -r requirements.txt
python prefetch.py            # one-time model download (~90 MB)
python evaluate.py            # prints Precision@5 for both methods
streamlit run app.py
```

## Deploy on Render (free tier)
1. Create a GitHub account/repo, upload this whole folder (`app.py`, `nlp.py`, `prefetch.py`, `evaluate.py`, `requirements.txt`, `render.yaml`, `README.md`).
2. Go to render.com -> sign in with GitHub -> **New +** -> **Web Service** -> pick the repo.
3. Settings: Runtime **Python 3**; Build command `pip install -r requirements.txt && python prefetch.py`; Start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`; Instance type **Free**. (Or choose **Blueprint** and Render reads `render.yaml`.)
4. Add environment variable `PYTHON_VERSION` = `3.11.9`.
5. Click **Create Web Service**; the first build takes about 5–10 min. Open the `onrender.com` URL.
6. Free services sleep after ~15 min idle; the first visit afterwards takes ~30–60 s to wake.

## Troubleshooting
- *Out of memory*: use the default model only; keep "Candidate papers" ≤ 60.
- *arXiv error / empty results*: arXiv rate-limits; wait a few seconds and retry. Results are cached for an hour.
- *Slow first search*: the model loads once, then is cached.
