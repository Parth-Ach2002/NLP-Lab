import streamlit as st

import nlp

st.set_page_config(page_title="Research Paper Recommender", page_icon="📄", layout="wide")
st.title("📄 Research Paper Recommender")
st.caption("Ask a research question. Papers are fetched from arXiv, ranked by meaning, and summarized.")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(q, n):
    return nlp.fetch_arxiv(q, max_results=n)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return nlp.get_embedder()


with st.sidebar:
    method = st.radio("Ranking method", ["Semantic (embeddings)", "TF-IDF (baseline)"])
    pool = st.slider("Candidate papers to fetch", 20, 100, 60, 10)
    top_k = st.slider("Results to show", 3, 10, 5)
    n_sent = st.slider("Summary sentences", 1, 5, 3)
    diverse = st.checkbox("Diversify results (MMR)", value=False,
                          help="Avoid near-duplicate papers (semantic mode only)")
    compare = st.checkbox("Show TF-IDF vs Semantic side by side", value=False)

query = st.text_input("Research question",
                     placeholder="How can language models detect sarcasm in social media?")


def show(papers, key):
    for i, p in enumerate(papers, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. [{p['title']}]({p['url']})**  \n"
                        f"`{p['category']}` · {p['published']} · score {p['score']:.3f}")
            st.caption(", ".join(p["authors"][:4]) + (" et al." if len(p["authors"]) > 4 else ""))
            st.write("**Summary:** " + nlp.summarize(p["abstract"], query, n=n_sent, embedder=load_model()))
            with st.expander("Full abstract"):
                st.write(p["abstract"])


if query:
    try:
        with st.spinner("Searching arXiv..."):
            papers = cached_fetch(query, pool)
    except Exception as e:
        st.error(f"Could not reach arXiv: {e}")
        st.stop()
    if not papers:
        st.warning("No papers found. Try different keywords.")
        st.stop()
    st.write(f"Fetched **{len(papers)}** candidates using keywords: `{nlp.query_to_keywords(query)}`")
    model = load_model()
    if compare:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("TF-IDF")
            show(nlp.rank(query, papers, "tfidf", top_k=top_k), "t")
        with c2:
            st.subheader("Semantic")
            show(nlp.rank(query, papers, "semantic", model, top_k=top_k), "s")
    elif method.startswith("TF-IDF"):
        show(nlp.rank(query, papers, "tfidf", top_k=top_k), "t")
    elif diverse:
        show(nlp.diversify(query, papers, model, top_k=top_k), "d")
    else:
        show(nlp.rank(query, papers, "semantic", model, top_k=top_k), "s")
