import io

import pandas as pd
import pdfplumber

from src.tfidf_cluster import cluster_blogs
from src.utils import extract_text_from_pdf


if __name__ == "__main__":
    # file_path = "data/sample_blogs.csv"  # Replace with actual dataset path
    # df = pd.read_csv(file_path)

    pdfs = [
        'data/CapitalCall.pdf',
        'data/Cap Call3.pdf',
        'data/capitalcall1.pdf',
        'data/CapitalCall2.pdf',
        'data/LPA.pdf',
        'data/LPA 1.pdf',
        'data/Rent Roll Sample Asset DEF.pdf'
    ]

    data = {
        "id": [],
        "blog_text": []
    }
    for index, path in enumerate(pdfs):
        pages = []
        try:
            with pdfplumber.open(path) as pdf:
                for i in range(min(5, len(pdf.pages))):
                    text = pdf.pages[i].extract_text()
                    if text:
                        pages.append(text)
                if not pages:
                    raise Exception("No text extracted from PDF")
        except Exception as e:
            print(e)

        all_text = ' '.join(pages)
        data["blog_text"].append(all_text)
        data["id"].append(index + 1)
    print(data)
    df = pd.DataFrame.from_dict(data)
    clustered_df, model, vectorizer = cluster_blogs(df)

    print(clustered_df[["blog_text", "cluster"]])
