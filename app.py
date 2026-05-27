import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

import matplotlib
matplotlib.use("Agg")

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Topic Modeling",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Topic Modeling (LDA vs NMF)")
st.caption("Minimal NLP Dashboard")

# ==============================
# SAFE NLTK LOADER
# ==============================
@st.cache_resource
def load_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

load_nltk()

# ==============================
# DATASET
# ==============================
uploaded = st.file_uploader("Upload CSV (must have 'text' column)", type=["csv"])

SAMPLE = {
    "text": [
        "The economy is growing fast with GDP increase",
        "Stock market reaches new highs",
        "Inflation is rising globally",
        "Unemployment rate is falling",
        "Climate change is worsening",
        "Renewable energy is expanding",
        "Carbon emissions must be reduced",
        "Electric vehicles are rising"
    ]
}

if uploaded:
    df = pd.read_csv(uploaded)

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column")
        st.stop()

    documents = df["text"].dropna().tolist()
    st.success("Uploaded dataset loaded")
else:
    documents = SAMPLE["text"]
    st.info("Using sample dataset")

st.write(f"📄 Total Documents: {len(documents)}")

# ==============================
# PREPROCESSING
# ==============================
stop_words = set(stopwords.words("english"))
processed_docs = []

for doc in documents:
    doc = doc.lower()
    doc = doc.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(doc)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    processed_docs.append(tokens)

# REMOVE EMPTY DOCS (IMPORTANT FIX)
processed_docs = [doc for doc in processed_docs if len(doc) > 0]

if len(processed_docs) < 2:
    st.error("Not enough valid documents after preprocessing.")
    st.stop()

with st.expander("🧹 Preprocessing", expanded=False):
    st.write(f"Stopwords: {len(stop_words)}")
    st.write(processed_docs)

# ==============================
# LDA (CACHED)
# ==============================
@st.cache_resource
def train_lda(processed_docs):
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(text) for text in processed_docs]

    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=2,
        passes=10,
        random_state=42
    )

    coherence = CoherenceModel(
        model=model,
        texts=processed_docs,
        dictionary=dictionary,
        coherence="c_v"
    ).get_coherence()

    perplexity = model.log_perplexity(corpus)

    return model, dictionary, corpus, coherence, perplexity


lda_model, lda_dict, lda_corpus, lda_coherence, lda_perplexity = train_lda(processed_docs)

lda_topics = lda_model.print_topics(num_topics=2, num_words=5)

# ==============================
# NMF (CACHED)
# ==============================
@st.cache_resource
def train_nmf(processed_docs):
    text_joined = [" ".join(doc) for doc in processed_docs]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_joined)

    model = NMF(n_components=2, random_state=42)
    model.fit(X)

    terms = vectorizer.get_feature_names_out()

    topics = []
    for topic in model.components_:
        top_idx = topic.argsort()[:-6:-1]
        topics.append([terms[i] for i in top_idx])

    return model, vectorizer, topics


nmf_model, vectorizer, nmf_topics = train_nmf(processed_docs)

nmf_dictionary = corpora.Dictionary(processed_docs)

nmf_coherence = CoherenceModel(
    topics=nmf_topics,
    texts=processed_docs,
    dictionary=nmf_dictionary,
    coherence="c_v"
).get_coherence()

# ==============================
# RESULTS
# ==============================
st.divider()
st.subheader("📈 Results")

col1, col2, col3 = st.columns(3)

col1.metric("LDA Coherence", f"{lda_coherence:.4f}")
col2.metric("NMF Coherence", f"{nmf_coherence:.4f}")
col3.metric("LDA Perplexity", f"{lda_perplexity:.2f}")

if lda_coherence > nmf_coherence:
    winner = "LDA"
elif nmf_coherence > lda_coherence:
    winner = "NMF"
else:
    winner = "Tie"

st.success(f"🏆 Best Model: {winner}")

# ==============================
# TOPICS
# ==============================
st.divider()

st.subheader("🔥 LDA Topics")
for t in lda_topics:
    st.write(t)

st.subheader("📌 NMF Topics")
for i, t in enumerate(nmf_topics):
    st.write(f"Topic {i+1}: {t}")

# ==============================
# TABLE
# ==============================
st.divider()
st.subheader("📊 Comparison Table")

df_results = pd.DataFrame({
    "Model": ["LDA", "NMF"],
    "Coherence": [lda_coherence, nmf_coherence],
    "Perplexity": [lda_perplexity, np.nan]
})

df_results["Perplexity"] = df_results["Perplexity"].fillna("N/A")

st.dataframe(df_results, use_container_width=True)