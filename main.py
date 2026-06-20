import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from nltk.corpus import stopwords
import spacy
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import json
from pathlib import Path
import joblib

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Fake News Analysis", layout="wide")
st.title("📰 Fake News Detection & EDA")

# ---------- Sidebar navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA Dashboard", "Model Comparison", "Live Prediction"])

# ---------- Download NLTK stopwords ----------
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# ---------- Cached resources ----------
@st.cache_resource
def load_artifacts():
    """Load models, vectorizers, and evaluation results."""
    base = Path("saved_models")
    artifacts = {}
    # TF-IDF vectorizer
    artifacts['tfidf'] = joblib.load(base / 'tfidf_vectorizer.pkl')
    # Word2Vec model
    artifacts['w2v'] = Word2Vec.load(str(base / 'word2vec_model.model'))
    # Classification models
    model_files = {
        "Logistic Regression (TF-IDF)": "logistic_regression_tfidf.pkl",
        "Random Forest (TF-IDF)": "random_forest_tfidf.pkl",
        "Logistic Regression (Word2Vec)": "logistic_regression_word2vec.pkl",
        "Random Forest (Word2Vec)": "random_forest_word2vec.pkl"
    }
    artifacts['models'] = {}
    for name, fname in model_files.items():
        path = base / fname
        if path.exists():
            artifacts['models'][name] = joblib.load(path)
    # Evaluation results
    eval_path = base / 'evaluation_results.json'
    if eval_path.exists():
        with open(eval_path) as f:
            artifacts['evaluation'] = json.load(f)
    else:
        artifacts['evaluation'] = None
    return artifacts

@st.cache_resource
def load_spacy():
    nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "parser", "ner"])
    return nlp

# ---------- Load data (cached) ----------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('DataSet/WELFake_Dataset.csv')
    except FileNotFoundError:
        st.warning("Dataset not found. You can upload it manually in the sidebar.")
        return None
    # Drop the first column (unnamed)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df.dropna().drop_duplicates()
    # Add text length features
    for col in ['title', 'text']:
        df[f'{col}_sentences'] = df[col].str.split(r'\.+\s*').str.len()
        df[f'{col}_words'] = df[col].str.split().str.len()
        df[f'{col}_characters'] = df[col].str.len()
    return df

# -----------------------------------------------------------------------------
# EDA DASHBOARD
# -----------------------------------------------------------------------------
if page == "EDA Dashboard":
    st.header("Exploratory Data Analysis")

    # Load dataset
    df = load_data()
    if df is None:
        uploaded = st.file_uploader("Upload WELFake_Dataset.csv", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            if 'Unnamed: 0' in df.columns:
                df.drop(columns=['Unnamed: 0'], inplace=True)
            df = df.dropna().drop_duplicates()
            for col in ['title', 'text']:
                df[f'{col}_sentences'] = df[col].str.split(r'\.+\s*').str.len()
                df[f'{col}_words'] = df[col].str.split().str.len()
                df[f'{col}_characters'] = df[col].str.len()
        else:
            st.stop()

    st.write(f"Dataset shape: {df.shape}")

    # ---------- Pie chart of labels ----------
    with st.expander("📊 Label Distribution", expanded=True):
        fig = px.pie(df['label'].value_counts().reset_index(),
                     values='count', names='label',
                     title='Real (0) vs Fake (1)')
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Histograms ----------
    hist_cols = [f'{col}_{metric}' for col in ['title', 'text']
                 for metric in ['sentences', 'words', 'characters']]
    with st.expander("📈 Text Length Distributions", expanded=True):
        metric = st.selectbox("Select feature", hist_cols)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Real', 'Fake'),
                            shared_yaxes=True)
        fig.add_trace(go.Histogram(x=df[df['label']==0][metric],
                                   name='Real', marker_color='blue',
                                   opacity=0.7, nbinsx=50),
                      row=1, col=1)
        fig.add_trace(go.Histogram(x=df[df['label']==1][metric],
                                   name='Fake', marker_color='red',
                                   opacity=0.7, nbinsx=50),
                      row=1, col=2)
        fig.update_layout(title=f'{metric} Distribution', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Word clouds ----------
    with st.expander("☁️ Word Clouds", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Real Word Cloud"):
                text = ' '.join(df[df['label']==0]['text'])
                wc = WordCloud(width=400, height=300, background_color='white',
                               colormap='Blues', stopwords=STOP_WORDS,
                               max_words=100).generate(text)
                fig = px.imshow(wc)
                fig.update_layout(title="Real News")
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if st.button("Generate Fake Word Cloud"):
                text = ' '.join(df[df['label']==1]['text'])
                wc = WordCloud(width=400, height=300, background_color='white',
                               colormap='Reds', stopwords=STOP_WORDS,
                               max_words=100).generate(text)
                fig = px.imshow(wc)
                fig.update_layout(title="Fake News")
                st.plotly_chart(fig, use_container_width=True)

    # ---------- Top n-grams ----------
    with st.expander("🔤 Top N-grams Comparison", expanded=True):
        ngram_type = st.radio("N-gram type", ["Unigrams", "Bigrams"])
        n = st.slider("Number of n-grams to show", 5, 30, 15)
        ngram_range = (1,1) if ngram_type == "Unigrams" else (2,2)

        if st.button("Generate N-grams"):
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=('Real', 'Fake'),
                                shared_yaxes=True)
            colors = ['blue', 'red']
            for idx, label in enumerate([0, 1]):
                texts = df[df['label']==label]['text'].astype(str)
                vec = CountVectorizer(stop_words=list(STOP_WORDS),
                                      max_features=n,
                                      ngram_range=ngram_range)
                X = vec.fit_transform(texts)
                freq = zip(vec.get_feature_names_out(), X.toarray().sum(axis=0))
                freq = sorted(freq, key=lambda x: x[1], reverse=True)
                ngrams = [f[0] for f in freq]
                freqs = [f[1] for f in freq]
                fig.add_trace(go.Bar(x=ngrams, y=freqs,
                                     marker_color=colors[idx],
                                     text=freqs, textposition='auto',
                                     name='Real' if label==0 else 'Fake'),
                              row=1, col=idx+1)
            fig.update_layout(title=f'Top {n} {ngram_type}: Real vs Fake',
                              height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# MODEL COMPARISON
# -----------------------------------------------------------------------------
elif page == "Model Comparison":
    st.header("Model Comparison")
    artifacts = load_artifacts()
    if artifacts['evaluation'] is None:
        st.error("Evaluation results not found. Please run the extended training script to save them.")
        st.stop()

    eval_data = artifacts['evaluation']  # list of dicts

    # ----- Accuracy & AUC Bar Chart -----
    with st.expander("📊 Performance Metrics", expanded=True):
        names = [d['name'] for d in eval_data]
        acc = [d['accuracy'] for d in eval_data]
        auc = [d['auc'] for d in eval_data]
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=names, y=acc, marker_color='#2E86AB',
                   text=[f"{v:.3f}" for v in acc], textposition='auto'),
            go.Bar(name='AUC', x=names, y=auc, marker_color='#A23B72',
                   text=[f"{v:.3f}" for v in auc], textposition='auto')
        ])
        fig.update_layout(title='Model Performance Comparison',
                          barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ----- ROC Curves -----
    with st.expander("📈 ROC Curves", expanded=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 name='Random', line=dict(dash='dash', color='grey')))
        colors = px.colors.qualitative.Plotly
        for i, d in enumerate(eval_data):
            fig.add_trace(go.Scatter(x=d['fpr'], y=d['tpr'], mode='lines',
                                     name=f"{d['name']} (AUC={d['auc']:.3f})",
                                     line=dict(color=colors[i], width=2)))
        fig.update_layout(title='ROC Curves', xaxis_title='FPR',
                          yaxis_title='TPR', height=500,
                          legend=dict(yanchor="bottom", y=0.01,
                                      xanchor="right", x=0.99))
        st.plotly_chart(fig, use_container_width=True)

    # ----- Confusion Matrices -----
    with st.expander("🔢 Confusion Matrices", expanded=True):
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=[d['name'] for d in eval_data[:4]])
        for i, d in enumerate(eval_data[:4]):
            row = i//2 + 1
            col = i%2 + 1
            cm = np.array(d['confusion_matrix'])
            heatmap = go.Heatmap(z=cm,
                                 x=['Pred Fake','Pred Real'],
                                 y=['Actual Fake','Actual Real'],
                                 text=cm, texttemplate="%{text}",
                                 colorscale='Viridis', showscale=False)
            fig.add_trace(heatmap, row=row, col=col)
        fig.update_layout(height=600, title='Confusion Matrices')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# LIVE PREDICTION
# -----------------------------------------------------------------------------
elif page == "Live Prediction":
    st.header("Live Classification")
    artifacts = load_artifacts()
    nlp = load_spacy()

    # Text preprocessing functions
    def clean_text(text, lemmatize=True):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if lemmatize:
            doc = nlp(text)
            tokens = [tok.lemma_ for tok in doc
                      if tok.lemma_ not in STOP_WORDS
                      and len(tok.lemma_) > 2
                      and not tok.is_punct
                      and not tok.is_space]
            return ' '.join(tokens)
        else:
            tokens = [w for w in text.split()
                      if w not in STOP_WORDS and len(w) > 2]
            return ' '.join(tokens)

    # Model selection
    avail_models = list(artifacts['models'].keys())
    if not avail_models:
        st.error("No models found in saved_models folder.")
        st.stop()
    model_choice = st.sidebar.selectbox("Choose Model", avail_models)
    is_tfidf = "tf-idf" in model_choice.lower()

    # Input fields
    title = st.text_input("Article Title (optional)")
    text = st.text_area("Article Text", height=200)

    if st.button("Check Authenticity"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            full = (title + " " + text).strip()
            processed = clean_text(full, lemmatize=is_tfidf)

            if is_tfidf:
                features = artifacts['tfidf'].transform([processed])
            else:
                # Word2Vec
                words = processed.split()
                vecs = [artifacts['w2v'].wv[w] for w in words if w in artifacts['w2v'].wv]
                features = np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((1, 300))

            model = artifacts['models'][model_choice]
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None

            st.markdown("---")
            if pred == 1:
                st.success("✅ This article is likely **REAL**")
                conf = proba[1]*100 if proba is not None else None
            else:
                st.error("🚨 This article is likely **FAKE**")
                conf = proba[0]*100 if proba is not None else None

            if conf:
                st.metric("Confidence", f"{conf:.2f}%")
                st.progress(conf/100)
            with st.expander("Preprocessed Text"):
                st.code(processed)
