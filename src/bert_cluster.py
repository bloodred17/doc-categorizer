import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN


def cluster_bert_kmeans(df, num_clusters=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight & fast BERT model

    # Convert text into dense embeddings
    embeddings = model.encode(df["blog_text"].tolist(), convert_to_tensor=True)

    # Apply clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings.cpu().numpy())

    return df, kmeans, embeddings


def cluster_bert_dbscan(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(df["blog_text"].tolist(), convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()

    # Use cosine distance instead of Euclidean
    dbscan = DBSCAN(metric="cosine", eps=0.6, min_samples=2)
    labels = dbscan.fit_predict(embeddings_np)

    df["cluster"] = labels
    return df, dbscan, embeddings
