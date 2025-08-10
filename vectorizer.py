import os
import fitz
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "shopflo.pdf"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
EMBEDDINGS_MODEL = "intfloat/multilingual-e5-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def extract_text_from_pdf(pdf_path: str) -> str:
    print("Extracting text from PDF...")
    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"PDF Extraction failed: {e}")
        return ""


def split_text_into_documents(text: str) -> List[Document]:
    print("Splitting text into chunks...")
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        documents = [Document(page_content=text)]
        return splitter.split_documents(documents)
    except Exception as e:
        print(f"Error during splitting: {e}")
        return []


def create_vector_store(documents: list):
    print("Creating vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        Chroma.from_documents(documents, embeddings, persist_directory=DB_PATH)
        print("Vector store saved.")
    except Exception as e:
        print(f"Vector store creation failed: {e}")


def main():
    text = extract_text_from_pdf(PDF_PATH)
    if not text:
        return

    docs = split_text_into_documents(text)
    if docs:
        create_vector_store(docs)


if __name__ == "__main__":
    main()
