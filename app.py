from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
# Try to import a Gemma SDK placeholder if available, otherwise None
try:
    import gemma as _gemma_placeholder
except Exception:
    _gemma_placeholder = None
from retriever import Retriever
from config import GEMMA_API_KEY, client

if not GEMMA_API_KEY:
    print("Warning: GEMMA_API_KEY not set. /chat will fail until it's provided in the environment.")

app = FastAPI(title="RAG Prototype")
retriever = Retriever()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    stream: Optional[bool] = False


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """Ingest uploaded text/plain files into the vector DB."""
    saved = []
    for f in files:
        raw = await f.read()
        # try utf-8, then latin-1, then replace malformed bytes to avoid UnicodeDecodeError
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw.decode("latin-1")
            except Exception:
                content = raw.decode("utf-8", errors="replace")
        retriever.add_documents_from_string(content, source=f.filename)
        saved.append(f.filename)
    return {"ingested_files": saved}


def _build_prompt(question: str, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Create a chat-style conversation with system instruction and user message including retrieved context."""
    system = (
        "You are an expert research assistant helping new researchers learn how to search for exoplanets. "
        "Answer concisely, cite the source filenames for any factual claims using square-bracketed tags (e.g. [paper.txt]). "
        "If the answer requires steps, provide an ordered list."
    )
    # Attach retrieved passages as numbered context
    context_parts = []
    for i, d in enumerate(docs, start=1):
        src = d.get("metadata", {}).get("source", "unknown")
        context_parts.append(f"[{i}] source={src} text={d.get('document','')}")

    user_msg = (
        f"Question: {question}\n\nRetrieved passages:\n" + "\n\n".join(context_parts) + "\n\nRespond using these passages where relevant and include citations."
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]


@app.post("/chat")
async def chat(req: QueryRequest, request: Request):
    if not GEMMA_API_KEY:
        raise HTTPException(status_code=500, detail="GEMMA_API_KEY not configured on the server")

    question = req.question
    top_k = req.top_k or 5
    docs = retriever.retrieve(question, top_k=top_k)

    messages = _build_prompt(question, docs)

    model = os.environ.get("GEMMA_MODEL", "gpt-3.5-turbo")

    if not req.stream:
        # Non-streaming call
        if client is not None:
            resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2)
            # support attribute or dict-style access
            try:
                answer = resp.choices[0].message.content
            except Exception:
                answer = resp["choices"][0]["message"]["content"]
        else:
            # legacy/placeholder path: ensure placeholder SDK is present
            if _gemma_placeholder is None:
                raise HTTPException(status_code=500, detail=(
                    "No Gemma client available to create chat completions. "
                    "Set GEMMA_API_KEY/GEMMA_BASE_URL or install a compatible 'gemma' SDK."
                ))
            resp = _gemma_placeholder.ChatCompletion.create(model=model, messages=messages, temperature=0.2)
            answer = resp["choices"][0]["message"]["content"]
        # collect unique sources
        sources = []
        for d in docs:
            s = d.get("metadata", {}).get("source")
            if s and s not in sources:
                sources.append(s)
        return {"answer": answer, "sources": sources, "docs": docs}

    # Streaming path: return Server-Sent-Events
    def event_stream():
        try:
            if client is not None:
                stream_resp = client.chat.completions.create(model=model, messages=messages, temperature=0.2, stream=True)
            else:
                if _gemma_placeholder is None:
                    raise HTTPException(status_code=500, detail=(
                        "No Gemma client available to create chat completions (streaming). "
                        "Set GEMMA_API_KEY/GEMMA_BASE_URL or install a compatible 'gemma' SDK."
                    ))
                stream_resp = _gemma_placeholder.ChatCompletion.create(model=model, messages=messages, temperature=0.2, stream=True)

            for chunk in stream_resp:
                # chunk may be an object with attribute access or a dict
                try:
                    # try modern SDK attributes
                    delta = None
                    if hasattr(chunk, "choices"):
                        c = chunk.choices[0]
                        # modern streaming may use delta or message
                        delta = getattr(c, "delta", None) or getattr(c, "message", None)
                    else:
                        delta = chunk.get("choices", [])[0].get("delta", {})

                    # extract content from delta if present
                    content = None
                    if isinstance(delta, dict):
                        content = delta.get("content")
                    else:
                        content = getattr(delta, "content", None)

                    if content:
                        yield "data: " + json.dumps({"delta": content}) + "\n\n"
                except Exception:
                    continue
        except Exception as e:
            yield "data: " + json.dumps({"error": str(e)}) + "\n\n"
        # at the end send final event with sources
        final_sources = list({d.get("metadata", {}).get("source") for d in docs if d.get("metadata", {}).get("source")})
        yield "data: " + json.dumps({"done": True, "sources": final_sources}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
