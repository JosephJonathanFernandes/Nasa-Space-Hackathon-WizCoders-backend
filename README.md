# RAG Backend (Prototype)

This folder contains a small prototype Retrieval-Augmented Generation (RAG) backend using FastAPI, SentenceTransformers for embeddings, and Chroma as the vector store.

Folders / files
- `app.py` - FastAPI server exposing `/ingest` and `/query` endpoints
- `ingest.py` - simple ingestion utility to load text files from a directory, chunk them, embed and store into Chroma
- `retriever.py` - helper functions for retrieving top-k passages
- `query_example.py` - small client that calls `/query`
- `requirements.txt` - python deps

Notes
- This is a prototype. For production use Pinecone/Weaviate/managed DB and secure your API keys.

