from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import time
import logging
import traceback
import sys
import pathlib
import io
import matplotlib.pyplot as plt
from lightkurve import read

# Prefer relative import when the package is imported as `webapp.backend`.
# This allows `uvicorn webapp.backend.main:app` to find `model_utils` in the same package.
try:
    # When running as a package (recommended), use a relative import
    from .model_utils import load_bundle, predict_single  # type: ignore
except Exception:
    # Fall back to previous heuristic: attempt top-level module import by adding
    # the repo RAG folder to sys.path if necessary (keeps backwards compatibility
    # with running the module from different working directories).
    repo_file = pathlib.Path(__file__).resolve()
    try:
        rag_dir = repo_file.parents[3] / 'RAG'
        sys.path.insert(0, str(rag_dir))
        from model_utils import load_bundle, predict_single
    except Exception:
        # final fallback: try importing a sibling `model_utils` at package level
        try:
            from model_utils import load_bundle, predict_single
        except Exception:
            # re-raise the original ImportError to preserve traceback
            raise

# Load model bundle lazily on first request and cache
_MODEL_BUNDLE = None


def _get_model_bundle():
    """Locate and load the model bundle from the RAG `models/` directory.

    This tries to find `RAG/models` relative to this file; if not found it falls back
    to a `models` dir next to this file.
    """
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        # preferred models directory: backend/RAG/models (use same repo layout)
        this_file = pathlib.Path(__file__).resolve()
        try:
            rag_dir = this_file.parents[3] / 'RAG'
            models_dir = str(rag_dir / 'models')
            if not os.path.exists(models_dir):
                # fallback to local models dir next to this file
                models_dir = os.path.join(os.path.dirname(__file__), "models")
        except Exception:
            models_dir = os.path.join(os.path.dirname(__file__), "models")

        _MODEL_BUNDLE = load_bundle(models_dir=models_dir)
    return _MODEL_BUNDLE

load_dotenv()

app = FastAPI(title="RAG Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    answer: str
    retrieved: List[str]


# Stateless backend: no in-memory session store or history trimming

# Lazy load heavy resources
_llm = None
_retriever = None


def get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    try:
        # Local embeddings and Chroma retriever (uses your existing chroma_db)
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # instantiate Chroma against the same persist_directory and collection_name used in your notebook
        # Different versions of the langchain-chroma wrapper expect different kwarg names for the
        # embedding object (embedding_function or embedding). Try the common variants and fall back
        # to constructing in the way supported by the installed package.
        try:
            vectordb = Chroma(persist_directory="chroma_db", collection_name="testing", embedding_function=embeddings)
        except TypeError:
            try:
                vectordb = Chroma(persist_directory="chroma_db", collection_name="testing", embedding=embeddings)
            except TypeError:
                # Last-resort: try constructing without passing embedding and hope the collection already
                # contains embeddings (read-only). If that fails, propagate a helpful error.
                try:
                    vectordb = Chroma(persist_directory="chroma_db", collection_name="testing")
                except Exception as e:
                    raise RuntimeError(
                        "Unable to instantiate Chroma with the installed langchain-chroma package. "
                        "Tried embedding_function, embedding, and no-embedding constructors. "
                        f"Underlying error: {e}"
                    )

        _retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        return _retriever
    except Exception as e:
        raise RuntimeError(f"Failed to create retriever: {e}")
    
class PredictRequest(BaseModel):
    # either features for a single sample, or omitted when uploading CSV
    features: Optional[List[float]] = None


@app.post("/predict")
async def predict(files: Optional[List[UploadFile]] = File(None), payload: Optional[PredictRequest] = None):
    """Predict endpoint: accepts either an uploaded CSV file (first row header optional) or a JSON body with `features`.

    CSV format: numeric columns only (or columns will be converted to floats). Returns per-row predictions.
    JSON format: {"features": [f1, f2, ...]} returns a single prediction.
    """
    try:
        bundle = _get_model_bundle()

        # If JSON payload provided and has features, run single prediction
        if payload is not None and payload.features is not None:
            res = predict_single(bundle, payload.features)
            return {"predictions": [res]}

        # If files provided, expect CSV in the first file
        if files and len(files) > 0:
            f = files[0]
            raw = await f.read()
            text = None
            try:
                text = raw.decode("utf-8")
            except Exception:
                text = raw.decode("latin-1")

            # parse CSV into rows of floats
            import csv
            from io import StringIO

            reader = csv.reader(StringIO(text))
            rows = list(reader)
            # If first row looks like header (non-numeric), skip it
            def _is_numeric_row(r):
                try:
                    [float(x) for x in r]
                    return True
                except Exception:
                    return False

            if len(rows) == 0:
                raise HTTPException(status_code=400, detail="Empty CSV file")

            if not _is_numeric_row(rows[0]) and len(rows) > 1 and _is_numeric_row(rows[1]):
                rows = rows[1:]

            predictions = []
            for r in rows:
                try:
                    feats = [float(x) for x in r]
                except Exception:
                    raise HTTPException(status_code=400, detail=f"Non-numeric value in row: {r}")
                pred = predict_single(bundle, feats)
                predictions.append(pred)

            return {"predictions": predictions}

        # nothing to do
        raise HTTPException(status_code=400, detail="Provide features JSON or upload a CSV file")
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Error in /predict: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def get_llm():
    global _llm
    if _llm is not None:
        return _llm

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            _llm = ChatGroq(model=os.getenv("MODEL_NAME", "openai/gpt-oss-20b"), api_key=groq_key)
            return _llm
        except Exception as e:
            # If model lib is missing, fall through and return None
            print("Failed to load ChatGroq:", e)
    # No LLM available — return None and we'll fallback in route
    return None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    retriever = get_retriever()
    llm = get_llm()
    # Get relevant documents
    try:
        docs = retriever.get_relevant_documents(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    retrieved_texts = [d.page_content if hasattr(d, 'page_content') else str(d) for d in docs]

    # If no LLM configured, return the retrieved passages as a fallback answer
    if llm is None:
        answer = "\n\n".join(retrieved_texts)
        answer = f"(No LLM configured) Retrieved passages:\n\n{answer}"
        return ChatResponse(answer=answer, retrieved=retrieved_texts)

    # Build prompt body combining history and retrieved context
    try:
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an assistant. Use the retrieved context to answer the question.

Context:
{context}

Question:
{question}
""",
        )

        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
        result = chain.run(req.question)
        return ChatResponse(answer=result, retrieved=retrieved_texts)
    except Exception as e:
        # If chain fails, still return retrieved passages with error message
        fallback = f"(Error using LLM: {e})\n\n" + "\n\n".join(retrieved_texts)
        return ChatResponse(answer=fallback, retrieved=retrieved_texts)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload_fits/")
async def upload_fits_file(file: UploadFile = File(...)):
    """Upload a FITS file, read a light curve using lightkurve, and return a PNG plot.

    Expects a single file upload (FITS). Returns image/png.
    """
    try:
        contents = await file.read()
        lc_file = io.BytesIO(contents)

        # Use lightkurve to read the file. This returns a LightCurve or LightCurveCollection.
        lc = read(lc_file)

        # For LightCurveCollection, pick the first LightCurve
        try:
            time_arr = lc.time.value
            flux_arr = lc.flux.value
        except Exception:
            # Try indexing if it's a collection
            if hasattr(lc, '__len__') and len(lc) > 0:
                single = lc[0]
                time_arr = single.time.value
                flux_arr = single.flux.value
            else:
                raise

        # Plot
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 7))
        plt.plot(time_arr, flux_arr, color='teal', linewidth=0.8)
        plt.xlabel("Time (days)", fontsize=12)
        plt.ylabel("Normalized Flux", fontsize=12)
        plt.title(f"Light Curve for {file.filename}", fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logging.error("Error in /upload_fits/: %s", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing FITS file: {e}")


@app.get("/stream_chat")
async def stream_chat(question: str):
    """Stream retrieved context and the final answer using Server-Sent Events (SSE).

    Returns events of type 'context' for the retrieved passages, 'chunk' for the final answer (may be a single chunk),
    and 'done' when complete.
    """
    # Stateless streaming: no session handling here
    retriever = get_retriever()
    llm = get_llm()

    try:
        docs = retriever.get_relevant_documents(question)
    except Exception as e:
        # yield an error event and finish
        def error_gen():
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            yield "event: done\ndata: \n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    retrieved_texts = [d.page_content if hasattr(d, 'page_content') else str(d) for d in docs]

    async def event_generator():
        # First send retrieved context as one or more context events
        for txt in retrieved_texts:
            payload = json.dumps({"text": txt})
            yield f"event: context\ndata: {payload}\n\n"

        # If no LLM is configured, stream the retrieved passages as the answer fallback
        if llm is None:
            answer = "\n\n".join(retrieved_texts)
            yield f"event: chunk\ndata: {json.dumps({'text': answer})}\n\n"
            yield "event: done\ndata: \n\n"
            return

        # Run the retrieval + generation chain with streaming support if available
        try:
            from langchain.prompts import PromptTemplate
            from langchain.chains import RetrievalQA

            # Construct prompt with context only
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template="""
You are an assistant. Use the retrieved context to answer the question.

Context:
{context}

Question:
{question}
""",
            )

            # If the LLM supports streaming, try to hook into its streaming API
            if hasattr(llm, 'stream') or hasattr(llm, 'generate_stream'):
                # Many LLM wrappers expose some streaming generator. We'll try common names.
                stream_fn = getattr(llm, 'stream', None) or getattr(llm, 'generate_stream', None)
                if stream_fn is not None:
                    # Try to call stream function with question/context; the exact signature varies by LLM wrapper
                    try:
                        # Attempt a best-effort call signature
                        for token in stream_fn(question):
                            # token may be string or token dict
                            text = token if isinstance(token, str) else str(token)
                            yield f"event: chunk\ndata: {json.dumps({'text': text})}\n\n"
                        yield "event: done\ndata: \n\n"
                        return
                    except Exception:
                        # fall through to non-streaming path
                        pass

            # Fallback: run the chain and stream the result progressively by tokenizing locally
            chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
            result = chain.run(question)

            # Simple token-level progressive streaming: split by whitespace tokens
            tokens = result.split()
            buffer = ''
            for i, tk in enumerate(tokens):
                buffer += tk + ' '
                # yield every 8 tokens (tunable) to simulate streaming
                if i % 8 == 0:
                    yield f"event: chunk\ndata: {json.dumps({'text': buffer})}\n\n"
                    buffer = ''
                    # tiny sleep to let client render progressively (non-blocking in async generator)
                    time.sleep(0.01)
            if buffer:
                yield f"event: chunk\ndata: {json.dumps({'text': buffer})}\n\n"
            yield "event: done\ndata: \n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            yield "event: done\ndata: \n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class ReindexRequest(BaseModel):
    pdf_path: str


@app.post('/reindex')
async def reindex(req: ReindexRequest):
    """Reindex a PDF into the local Chroma DB. This will overwrite the `testing` collection's contents.

    Request body: {"pdf_path": "relative/or/absolute/path/to/file.pdf"}
    """
    pdf_path = req.pdf_path
    if not pdf_path:
        raise HTTPException(status_code=400, detail="pdf_path is required")

    try:
        # Load PDF
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings
        from langchain_chroma import Chroma

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Try common constructors for Chroma.from_documents or Chroma(...).from_documents
        try:
            vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="chroma_db", collection_name="testing")
        except TypeError:
            try:
                vectordb = Chroma.from_documents(chunks, embedding_function=embeddings, persist_directory="chroma_db", collection_name="testing")
            except Exception:
                # Fall back to building a new Chroma and adding documents if necessary
                try:
                    vectordb = Chroma(persist_directory="chroma_db", collection_name="testing", embedding_function=embeddings)
                    vectordb.add_documents(chunks)
                except Exception as e:
                    raise RuntimeError(f"Failed to create or populate Chroma: {e}")

        return {"status": "ok", "count": getattr(vectordb, '_collection', {}).count() if hasattr(vectordb, '_collection') else None}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
