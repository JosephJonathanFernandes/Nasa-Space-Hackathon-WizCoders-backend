import asyncio
from types import SimpleNamespace
import json

import main


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_health():
    res = run(main.health())
    assert res == {"status": "ok"}


def test_predict_json_single(monkeypatch):
    # Prevent model bundle loading and mock prediction
    monkeypatch.setattr(main, "_get_model_bundle", lambda: "dummy_bundle")
    monkeypatch.setattr(main, "predict_single", lambda bundle, feats: sum(feats))

    payload = main.PredictRequest(features=[1.5, 2.5])
    res = run(main.predict(files=None, payload=payload))
    assert res == {"predictions": [4.0]}


def test_predict_csv_upload(monkeypatch):
    monkeypatch.setattr(main, "_get_model_bundle", lambda: "dummy_bundle")
    monkeypatch.setattr(main, "predict_single", lambda bundle, feats: round(sum(feats), 2))

    csv_content = "a,b\n1.0,2.0\n3.5,4.5\n".encode("utf-8")

    class DummyUpload:
        def __init__(self, data: bytes):
            self._data = data

        async def read(self):
            return self._data

    dummy = DummyUpload(csv_content)
    res = run(main.predict(files=[dummy], payload=None))
    assert res == {"predictions": [3.0, 8.0]}


def test_chat_fallback_retriever(monkeypatch):
    # Mock retriever to return 2 simple documents
    docs = [SimpleNamespace(page_content="Doc one"), SimpleNamespace(page_content="Doc two")]

    class Retriever:
        def get_relevant_documents(self, q):
            return docs

    monkeypatch.setattr(main, "get_retriever", lambda: Retriever())
    # Ensure no LLM is configured so endpoint returns retrieved passages
    monkeypatch.setattr(main, "get_llm", lambda: None)

    req = main.ChatRequest(question="What is this?")
    res = run(main.chat(req))
    assert isinstance(res, main.ChatResponse)
    assert "(No LLM configured) Retrieved passages:" in res.answer
    assert res.retrieved == ["Doc one", "Doc two"]


def test_stream_chat_returns_sse(monkeypatch):
    docs = [SimpleNamespace(page_content="Alpha"), SimpleNamespace(page_content="Beta")]

    class Retriever:
        def get_relevant_documents(self, q):
            return docs

    monkeypatch.setattr(main, "get_retriever", lambda: Retriever())
    monkeypatch.setattr(main, "get_llm", lambda: None)

    res = run(main.stream_chat("stream?"))
    # Should return a StreamingResponse with SSE media type
    from starlette.responses import StreamingResponse

    assert isinstance(res, StreamingResponse)
    assert res.media_type == "text/event-stream"
