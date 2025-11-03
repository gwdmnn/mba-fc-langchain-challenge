import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

for k in ("DB_CONNECTION_STRING", "PGVECTOR_COLLECTION_NAME", "GOOGLE_API_KEY"):
    if k not in os.environ:
        raise ValueError(f"Environment variable {k} is not set.")
    
path = Path(__file__).parent.parent
document = PyPDFLoader(path/"document.pdf").load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(document)

enriched_documents = [
    Document(
        page_content=chunk.page_content,
        metadata={
            k: v for k, v in chunk.metadata.items() if v not in (None, "")
        },
    )
    for chunk in chunks
]

document_ids = [f"doc-{i}" for i in range(len(enriched_documents))]

embeddings = GoogleGenerativeAIEmbeddings(model=os.environ["GOOGLE_EMBEDDING_MODEL"])

vector_store = PGVector(
    embeddings,
    collection_name=os.environ["PGVECTOR_COLLECTION_NAME"],
    connection=os.environ["DB_CONNECTION_STRING"],
    use_jsonb=True
)

vector_store.add_documents(
    enriched_documents,
    ids=document_ids
)



