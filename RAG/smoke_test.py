import sys
import os
sys.path.append(os.path.dirname(__file__))
from retriever import Retriever

r = Retriever(persist_directory='./chroma_db_test')
try:
    r.add_documents_from_string('This is a test document. It should produce embeddings.')
    print('add_documents_from_string completed')
except Exception as e:
    print('Raised:', type(e).__name__, str(e))
