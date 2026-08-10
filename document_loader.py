import os
import tempfile

from langchain_community.document_loaders import (TextLoader, PyPDFLoader)
from dotenv import load_dotenv

load_dotenv()


def load_text_file():
    # Create a temporary file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello this is a sample text file.\n"
            b"This file is used for learning purposes only."
        )
        temp_file_path = temp_file.name

    try:
        # Load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        # Print the loaded documents
        for doc in documents:
            print(doc.page_content)

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")

    for i, doc in enumerate(documents):
        print(f"\nDocument {i + 1}")
        print(f"Content Preview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")

    return documents

if __name__ == "__main__":
    pdf_loader("docs/sample_rag_document.pdf")