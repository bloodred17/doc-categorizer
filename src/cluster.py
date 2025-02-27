import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_tfid_kmeans(docs, num_clusters=5):
    vectorizer = TfidfVectorizer(max_features=2000)  # 2000
    tfidf_matrix = vectorizer.fit_transform(docs)

    model = KMeans(n_clusters=num_clusters, random_state=42)
    return model.fit_predict(tfidf_matrix)


def cluster_bert_kmeans(docs, num_clusters=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # or another model of your choice
    embeddings = model.encode(docs)
    model = KMeans(n_clusters=num_clusters, random_state=42)
    return model.fit_predict(embeddings)
