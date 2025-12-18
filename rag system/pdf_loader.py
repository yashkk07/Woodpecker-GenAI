import re
from pypdf import PdfReader

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return clean_text(text)

# Helper function to clean extracted text
def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"(\w+)\s*\n\s*(\w+)", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()
