import requests
import os
from pathlib import Path

# Create a directory to store the models if it doesn't exist
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ==============================================================================
# IMPORTANT: INSTRUCTIONS FOR THE USER
# ==============================================================================
# 1. Follow the guide in our conversation to upload your three .joblib model
#    files to a cloud storage service like Cloudflare R2.
# 2. Get the public URL for each of the three model files.
# 3. Replace the placeholder strings below with your actual public URLs.
#
# EXAMPLE:
# "heart_disease": "https://pub-xxxxxxxx.r2.dev/heart_disease_model.joblib",
# ==============================================================================
MODEL_URLS = {
    "heart_disease": "PASTE_YOUR_HEART_DISEASE_MODEL_URL_HERE",
    "heart_attack": "PASTE_YOUR_HEART_ATTACK_MODEL_URL_HERE",
    "heart_failure": "PASTE_YOUR_HEART_FAILURE_MODEL_URL_HERE",
}

def download_model(model_name: str) -> Path:
    """
    Downloads a model file from a URL if it doesn't already exist locally.
    """
    url = MODEL_URLS.get(model_name)
    if not url or "PASTE_YOUR_" in url:
        print(f"FATAL: URL for model '{model_name}' is not set. Please update the MODEL_URLS dictionary in model_loader.py with your public model URL.")
        return None

    file_name = f"{model_name}_model.joblib"
    local_path = MODEL_DIR / file_name

    if local_path.exists():
        print(f"Model '{file_name}' already exists locally. Skipping download.")
        return local_path

    print(f"Downloading model '{file_name}' from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded and saved model to {local_path}")
        return local_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading model '{model_name}': {e}")
        if local_path.exists():
            os.remove(local_path)
        return None

def load_all_models():
    """Downloads all models defined in MODEL_URLS."""
    print("Loading all models...")
    for model_name in MODEL_URLS.keys():
        download_model(model_name)
    print("Model loading process complete.")

if __name__ == '__main__':
    load_all_models()
