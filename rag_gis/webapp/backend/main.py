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
import lightkurve as lk
import tempfile
import numpy as np

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Predict endpoint: accepts an uploaded CSV file (preferred) or a JSON body with `features` as fallback.

    CSV flow (preferred):
      - read the first uploaded file as CSV (header allowed)
      - extract `id` column if present (otherwise use row indices)
      - drop `id` (and `class` if present) and use remaining columns as features
      - run prediction per-row using `predict_single` and return a list of {id, prediction, raw}

    JSON flow (fallback): {"features": [f1, f2, ...]} returns a single prediction.
    """
    try:
        bundle = _get_model_bundle()

        # If files provided, expect CSV in the first file (primary flow)
        if files and len(files) > 0:
            f = files[0]
            raw = await f.read()
            try:
                text = raw.decode("utf-8")
            except Exception:
                text = raw.decode("latin-1")

            # Use pandas for robust CSV parsing and numeric conversion
            from io import StringIO
            import pandas as pd

            try:
                df = pd.read_csv(StringIO(text))
            except Exception:
                raise HTTPException(status_code=400, detail="Failed to parse CSV file")

            if df.shape[0] == 0:
                raise HTTPException(status_code=400, detail="Empty CSV file")

            # Extract ids (if present) otherwise use row indices
            if 'id' in df.columns:
                ids = df['id'].tolist()
            else:
                ids = list(range(len(df)))

            # drop id and class (if present) before prediction
            drop_cols = [c for c in ('id', 'class') if c in df.columns]
            X_df = df.drop(columns=drop_cols)

            # Ensure numeric features
            try:
                X = X_df.astype(float)
            except Exception:
                try:
                    X = X_df.apply(pd.to_numeric, errors='raise')
                except Exception:
                    raise HTTPException(status_code=400, detail="CSV contains non-numeric values in feature columns")

            predictions = []
            for idx, row in X.iterrows():
                feats = row.tolist()
                try:
                    res = predict_single(bundle, feats)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Prediction failed for row {idx}: {e}")

                predictions.append({
                    'id': ids[idx],
                    'prediction': res.get('prediction'),
                    'raw': res.get('raw'),
                    'success': res.get('success', False),
                })

            return {"predictions": predictions}

        # JSON fallback: single-sample features
        if payload is not None and payload.features is not None:
            res = predict_single(bundle, payload.features)
            return {"predictions": [{"id": None, "prediction": res.get('prediction'), "raw": res.get('raw'), "success": res.get('success', False)}]}

        # nothing to do
        raise HTTPException(status_code=400, detail="Upload a CSV file or provide features JSON")
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


@app.get("/lightcurve/{kepler_id}")
async def get_lightcurve(kepler_id: str, mission: str = "Kepler"):
    """
    Generate a light curve plot for the given Kepler ID or TIC ID.
    
    Args:
        kepler_id: The Kepler ID, KIC ID, or TIC ID (e.g., "KIC 11904151", "TIC 259377017")
        mission: The mission to search for ("Kepler", "TESS", or "K2"). Default is "Kepler"
    
    Returns:
        PNG image of the light curve plot
    """
    try:
        logger.info(f"Searching for light curve: {kepler_id}, mission: {mission}")
        
        # Search for light curves
        search_result = lk.search_lightcurve(kepler_id, mission=mission)
        
        if len(search_result) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No light curves found for {kepler_id} in {mission} mission"
            )
        
        logger.info(f"Found {len(search_result)} light curves")
        
        # Download the first available light curve
        lc = search_result.download()
        
        if lc is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not download light curve for {kepler_id}"
            )
        
        logger.info(f"Downloaded light curve with {len(lc)} data points")
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        lc.plot()
        plt.title(f"{mission} Light Curve for {kepler_id}")
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot to BytesIO buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()  # Close the figure to free memory
        
        return StreamingResponse(
            io.BytesIO(img_buffer.read()),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename={kepler_id}_lightcurve.png"}
        )
        
    except Exception as e:
        logger.error(f"Error processing {kepler_id}: {str(e)}")
        
        # Close any open figures to prevent memory leaks
        plt.close('all')
        
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error while processing {kepler_id}: {str(e)}"
            )


@app.post("/upload_fits")
async def upload_fits_file(file: UploadFile = File(...)):
    """Upload a FITS file, read a light curve using lightkurve, and return a PNG plot.

    Expects a single file upload (FITS). Returns image/png.
    """
    try:
        contents = await file.read()
        # Write uploaded bytes to a temporary file. Some FITS readers (astropy/lightkurve)
        # may close or expect a real file-like object; using a temp file avoids errors
        # like "Cannot read from/write to a closed file-like object" when objects
        # lazily access the underlying file. Keep the temp file around until we're
        # done plotting, then remove it. If an error occurs while reading, remove
        # the temp file and re-raise.
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
            tmp_path = tmp.name
            tmp.write(contents)
            tmp.flush()
            tmp.close()

            # Use lightkurve to read the file. This returns a LightCurve or LightCurveCollection.
            try:
                lc = read(tmp_path)
            except Exception as _ll_err:
                # Some versions of lightkurve/astropy may read into an internal
                # BytesIO which can be closed; fall back to using astropy directly
                # to extract TIME/FLUX columns if available.
                try:
                    from astropy.io import fits as _fits

                    hdul = _fits.open(tmp_path, memmap=False)
                    time_arr = None
                    flux_arr = None
                    for hdu in hdul:
                        data = getattr(hdu, 'data', None)
                        if data is None:
                            continue
                        # Try table-like columns
                        cols = []
                        try:
                            cols = list(data.columns.names)
                        except Exception:
                            try:
                                cols = list(data.dtype.names or [])
                            except Exception:
                                cols = []

                        keys = [c.upper() for c in cols]
                        if 'TIME' in keys:
                            # choose a likely flux column
                            flux_candidates = ['PDCSAP_FLUX', 'SAP_FLUX', 'FLUX', 'PDCSAP_FLUX']
                            found_flux = None
                            for fc in flux_candidates:
                                if fc in keys:
                                    found_flux = cols[keys.index(fc)]
                                    break
                            if found_flux is not None:
                                time_arr = data[cols[keys.index('TIME')]]
                                flux_arr = data[found_flux]
                                break

                    # If we found arrays, build a minimal dummy object to match expected API
                    if time_arr is not None and flux_arr is not None:
                        class _DummyLC:
                            def __init__(self, t, f):
                                self.time = type('T', (), {'value': t})
                                self.flux = type('F', (), {'value': f})

                        lc = _DummyLC(time_arr, flux_arr)
                    else:
                        # re-raise original lightkurve error if fallback couldn't parse
                        raise _ll_err
                except Exception:
                    # If astropy isn't available or parsing fails, propagate the
                    # original error to be handled by the outer except.
                    raise _ll_err
        except Exception:
            # Clean up on error and propagate
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise

        # For LightCurveCollection, pick the first LightCurve or use the
        # values provided by the dummy object created above. Normalize flux to 1D
        def _flux_to_1d(time_arr, flux_arr):
            t = np.asarray(time_arr)
            f = np.asarray(flux_arr)

            # If already 1D and length matches time, return
            if f.ndim == 1 and f.shape[0] == t.shape[0]:
                return f

            # If first axis matches time length, average over remaining axes
            if f.shape[0] == t.shape[0]:
                return np.nanmean(f.reshape(f.shape[0], -1), axis=1)

            # Search for an axis that matches time length and move it to axis 0
            for axis in range(f.ndim):
                if f.shape[axis] == t.shape[0]:
                    f_moved = np.moveaxis(f, axis, 0)
                    return np.nanmean(f_moved.reshape(f_moved.shape[0], -1), axis=1)

            # Last resort: try to flatten to (time_len, -1)
            try:
                f_flat = f.reshape(t.shape[0], -1)
                return np.nanmean(f_flat, axis=1)
            except Exception:
                raise ValueError(f"Could not convert flux array of shape {f.shape} to 1D matching time length {t.shape[0]}")

        try:
            time_arr = lc.time.value
            flux_arr = lc.flux.value
            flux_arr = _flux_to_1d(time_arr, flux_arr)
        except Exception:
            # Try indexing if it's a collection
            if hasattr(lc, '__len__') and len(lc) > 0:
                single = lc[0]
                time_arr = single.time.value
                flux_arr = single.flux.value
                flux_arr = _flux_to_1d(time_arr, flux_arr)
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

        # Clean up the temporary FITS file now that we're done with lightkurve
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

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
