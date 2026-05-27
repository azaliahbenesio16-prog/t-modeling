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
# PAGE CONFIG (LIGHT + CLEAN)
# ==============================
st.set_page_config(
    page_title="Topic Modeling",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Topic Modeling (LDA vs NMF)")
st.caption("Minimal NLP Dashboard")

# ==============================
# NLTK SETUP
# ==============================
@st.cache_resource
def load_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

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
    documents = df["text"].dropna().tolist()
    st.success("Uploaded dataset loaded")
else:
    documents = SAMPLE["text"]
    st.info("Using sample dataset")

st.write(f"📄 Total Documents: {len(documents)}")

# ==============================
# PREPROCESSING + STOPWORDS
# ==============================
stop_words = set(stopwords.words("english"))
processed_docs = []

for doc in documents:
    doc = doc.lower()
    doc = doc.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(doc)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    processed_docs.append(tokens)

with st.expander("🧹 Preprocessing & Stopwords", expanded=False):
    st.write(f"Stopwords used: {len(stop_words)} words")
    for i, doc in enumerate(processed_docs):
        st.write(f"Doc {i+1}: {doc}")

# ==============================
# LDA MODEL
# ==============================
dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(text) for text in processed_docs]

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=2,
    passes=10,
    random_state=42
)

lda_topics = lda_model.print_topics(num_topics=2, num_words=5)

lda_coherence = CoherenceModel(
    model=lda_model,
    texts=processed_docs,
    dictionary=dictionary,
    coherence="c_v"
).get_coherence()

lda_perplexity = lda_model.log_perplexity(corpus)

# ==============================
# NMF MODEL
# ==============================
text_joined = [" ".join(doc) for doc in processed_docs]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_joined)

nmf_model = NMF(n_components=2, random_state=42)
nmf_model.fit(X)

terms = vectorizer.get_feature_names_out()

nmf_topics = []

for topic in nmf_model.components_:
    top_idx = topic.argsort()[:-6:-1]
    nmf_topics.append([terms[i] for i in top_idx])

nmf_coherence = CoherenceModel(
    topics=nmf_topics,
    texts=processed_docs,
    dictionary=dictionary,
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

winner = "LDA" if lda_coherence > nmf_coherence else "NMF"
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
# TABLE (FIXED)
# ==============================
st.divider()
st.subheader("📊 Comparison Table")

df_results = pd.DataFrame({
    "Model": ["LDA", "NMF"],
    "Coherence": [lda_coherence, nmf_coherence],
    "Perplexity": [lda_perplexity, np.nan]
})

df_display = df_results.copy()
df_display["Perplexity"] = df_display["Perplexity"].fillna("N/A")

st.dataframe(df_display, use_container_width=True)