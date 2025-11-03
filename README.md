# Semantic Search RAG System

A semantic search and question-answering system using RAG (Retrieval-Augmented Generation) with PostgreSQL/pgvector and Google Gemini.

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Google AI API key

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**

   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```

   Required variables:
   - `GOOGLE_API_KEY` - Your Google AI API key
   - `GOOGLE_EMBEDDING_MODEL` - e.g., "models/embedding-001"
   - `CHAT_MODEL` - e.g., "gemini-2.5-flash-lite"
   - `DB_CONNECTION_STRING` - PostgreSQL connection string
   - `PGVECTOR_COLLECTION_NAME` - Collection name for vector storage

## Running the Application

### Step 1: Start the Database

```bash
docker compose up -d
```

### Step 2: Ingest the PDF

```bash
python src/ingest.py
```

This loads `document.pdf`, chunks it, generates embeddings, and stores them in the vector database.

### Step 3: Run the Chat

```bash
python src/chat.py
```

Start asking questions based on the ingested document content.

## Additional Tools

**Run semantic search:**
```bash
python src/search.py
```

**Run tests:**
```bash
pytest
pytest --cov  # with coverage
```

## Architecture

- **src/ingest.py** - PDF ingestion and vector storage
- **src/search.py** - Similarity search queries
- **src/chat.py** - Interactive RAG chat interface
- **PostgreSQL + pgvector** - Vector database
- **LangChain + Google Gemini** - Embeddings and chat models