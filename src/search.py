import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def search_documents(prompt: str, k: int = 10) -> List[Tuple[Document, float]]:
    for env_var in ("DB_CONNECTION_STRING", "PGVECTOR_COLLECTION_NAME", "GOOGLE_API_KEY"):
        if env_var not in os.environ:
            raise ValueError(f"Environment variable {env_var} is not set.")

    embeddings = GoogleGenerativeAIEmbeddings(model=os.environ["GOOGLE_EMBEDDING_MODEL"])

    vector_store = PGVector(
        embeddings,
        collection_name=os.environ["PGVECTOR_COLLECTION_NAME"],
        connection=os.environ["DB_CONNECTION_STRING"],
    )

    results = vector_store.similarity_search_with_score(
        query=prompt,
        k=k
    )

    return results
