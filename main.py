from src.tfidf_cluster import cluster_blogs

if __name__ == "__main__":
    file_path = "data/sample_blogs.csv"  # Replace with actual dataset path
    clustered_df, model, vectorizer = cluster_blogs(file_path)

    print(clustered_df[["blog_text", "cluster"]])
