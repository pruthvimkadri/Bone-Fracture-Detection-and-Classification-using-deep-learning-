import os, requests

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "static", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_URLS = {
    "MODEL_URL_BINARY": "binary_classifier.pt",
    "MODEL_URL_MULTI": "multi_classifier.pth",
    "MODEL_URL_MOBILENET": "mobilenetv2_xray_classifier.h5",
}

for env_var, filename in MODEL_URLS.items():
    url = os.getenv(env_var)
    if not url:
        raise RuntimeError(f"{env_var} is not set in environment variables on Render!")

    print(f"[DOWNLOAD] Fetching {filename} from {url}...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    print(f"✅ Saved {filename}")
