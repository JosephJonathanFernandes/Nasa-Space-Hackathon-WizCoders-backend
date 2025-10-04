from dotenv import load_dotenv
import os
# Load .env into environment as early as possible
load_dotenv()

# Expose the key for other modules
# Prefer an explicit GEMMA_API_KEY, but allow OPENROUTER_API_KEY to be used
GEMMA_API_KEY = os.environ.get("GEMMA_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

# Create and expose a modern Gemma client instance. We attempt to import a
# Create and expose a client. We support three modes:
# 1) If a real `gemma` SDK is installed and GEMMA_API_KEY is a token, try to use it.
# 2) If GEMMA_API_KEY (or GEMMA_BASE_URL) points to an HTTP OpenRouter/Gemma URL,
#    create a small HTTP adapter that exposes the minimal `client.chat.completions.create`
#    and `client.embeddings.create` used by the rest of the app.
# 3) Otherwise leave `client` as None and code will fall back to local/placeholder paths.
client = None

def _is_url(val: str) -> bool:
    return isinstance(val, str) and val.startswith("http")

# prefer explicit base URL env var if present
GEMMA_BASE_URL = os.environ.get("GEMMA_BASE_URL") or (GEMMA_API_KEY if _is_url(GEMMA_API_KEY) else None)
# token for OpenRouter/Gemma if provided (may be called OPENROUTER_API_KEY or GEMMA_API_KEY)
GEMMA_TOKEN = os.environ.get("GEMMA_API_KEY_TOKEN") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMMA_API_KEY")

try:
    # If a local `gemma`/SDK is installed, attempt to use it when GEMMA_API_KEY looks like a token
    if GEMMA_API_KEY and not _is_url(GEMMA_API_KEY):
        import gemma as _gemma_sdk
        client = _gemma_sdk.Gemma(api_key=GEMMA_API_KEY)
    elif GEMMA_BASE_URL:
        # create a minimal HTTP adapter for OpenRouter/Gemma-compatible endpoints
        import requests

        class _SimpleHTTPClient:
            def __init__(self, base_url: str, token: str | None = None):
                self.base_url = base_url.rstrip('/')
                self.token = token
                # attach namespaces
                self.chat = self._Chat(self)
                self.embeddings = self._Embeddings(self)

            class _Chat:
                def __init__(self, outer):
                    self.outer = outer
                    # provide .completions.create API
                    self.completions = self

                def create(self, model=None, messages=None, temperature=0.0, stream=False):
                    url = f"{self.outer.base_url}/chat/completions"
                    headers = {"Content-Type": "application/json"}
                    if self.outer.token:
                        headers["Authorization"] = f"Bearer {self.outer.token}"
                    payload = {"model": model, "messages": messages, "temperature": temperature}
                    resp = requests.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    if stream:
                        # simple streaming shim: yield a single full response
                        yield data
                    else:
                        return data

            class _Embeddings:
                def __init__(self, outer):
                    self.outer = outer

                def create(self, model=None, input=None):
                    url = f"{self.outer.base_url}/embeddings"
                    headers = {"Content-Type": "application/json"}
                    if self.outer.token:
                        headers["Authorization"] = f"Bearer {self.outer.token}"
                    payload = {"model": model, "input": input}
                    resp = requests.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return resp.json()

        # instantiate
        client = _SimpleHTTPClient(GEMMA_BASE_URL, GEMMA_TOKEN)
except Exception:
    # if any errors creating a client occur, leave client as None and let the rest of the app fall back
    client = None
