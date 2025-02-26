import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.pre_process import clean_text


def cluster_blogs(df, num_clusters=5):
    df["cleaned_text"] = df["blog_text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df["cleaned_text"])

    model = KMeans(n_clusters=num_clusters, random_state=42)
    df["cluster"] = model.fit_predict(X)

    return df, model, vectorizer