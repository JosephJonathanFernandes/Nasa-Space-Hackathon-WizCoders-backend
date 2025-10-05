import json
import requests

# Adjust host/port if using different uvicorn command
URL = "http://127.0.0.1:8000/predict"

sample = {
    "features": [9.488036, 2.9575, 615.8, 2.26, 793.0, 93.59, 5455.0, 4.467, 0.927, 291.93423, 48.141651]
}

resp = requests.post(URL, json=sample, timeout=10)
print("status:", resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text)
