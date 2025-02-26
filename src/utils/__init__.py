import io
from typing import Optional

from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_file: io.BytesIO) -> Optional[str]:
    # Create PDF reader object
    pdf_reader = PdfReader(pdf_file)

    # Get number of pages and limit to first 15
    max_pages = min(15, len(pdf_reader.pages))

    # Extract text from each page
    text = ""
    for page_num in range(max_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text() + "\n"

    return text.strip()