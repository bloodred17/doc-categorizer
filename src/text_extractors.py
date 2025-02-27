import PyPDF2


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file using PyPDF2.
    Returns a single string containing all the text in the PDF.
    """
    text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)
