from src.preprocessor import preprocess_text_alphanumeric
from src.text_extractors import extract_text_from_pdf
from src.cluster import cluster_tfid_kmeans, cluster_bert_kmeans

if __name__ == "__main__":
    # file_path = "data/sample_blogs.csv"  # Replace with actual dataset path
    # df = pd.read_csv(file_path)

    pdfs = [
        "data/CapitalCall.pdf",
        "data/Cap Call3.pdf",
        "data/capitalcall1.pdf",
        "data/CapitalCall2.pdf",
        "data/LPA.pdf",
        "data/LPA 1.pdf",
        "data/Rent Roll Sample Asset DEF.pdf",
    ]

    documents = []
    for pdf_path in pdfs:
        raw_text = extract_text_from_pdf(pdf_path)
        processed_text = preprocess_text_alphanumeric(raw_text)
        documents.append(processed_text)

    tfidf_kmeans_labels = cluster_tfid_kmeans(documents, 5)
    print("===== TF-IDF + KMeans Clusters =====")
    for i, pdf_path in enumerate(pdfs):
        print(f"Document: {pdf_path} -> Cluster {tfidf_kmeans_labels[i]}")
    print()

    bert_kmeans_labels = cluster_bert_kmeans(documents, 5)
    print("===== BERT Embeddings + KMeans Clusters =====")
    for i, pdf_path in enumerate(pdfs):
        print(f"Document: {pdf_path} -> Cluster {bert_kmeans_labels[i]}")
    print()


    # # clustered_df, model, vectorizer = get_bert_cluster(df, num_clusters=len(data["id"]))
    # clustered_df_1, _, _ = cluster_bert_dbscan(df)
    # clustered_df_2, _, _ = cluster_tfid_kmeans(df)
