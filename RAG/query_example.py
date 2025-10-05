import requests
import sseclient
import time
import json

BASE = "http://127.0.0.1:8000"


def chat(question: str, top_k: int = 5, stream: bool = False):
    payload = {"question": question, "top_k": top_k, "stream": stream}
    if not stream:
        r = requests.post(f"{BASE}/chat", json=payload)
        print(r.status_code)
        print(r.json())
        return

    # streaming path - use sseclient to consume server-sent events
    with requests.post(f"{BASE}/chat", json=payload, stream=True) as r:
        client = sseclient.SSEClient(r)
        for event in client.events():
            try:
                data = event.data
                if not data:
                    continue
                obj = None
                try:
                    obj = json.loads(data)
                except Exception:
                    print(data)
                    continue
                if obj.get("delta"):
                    print(obj.get("delta"), end="", flush=True)
                if obj.get("done"):
                    print("\n---\nSources:", obj.get("sources"))
                    break
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    q = "How should a new researcher start searching for exoplanets using transit photometry?"
    chat(q, top_k=3, stream=False)
