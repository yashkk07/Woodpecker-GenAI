import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:latest"  # change if needed

def generate_text(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)

    if response.status_code != 200:
        raise RuntimeError(response.text)

    return response.json()["response"].strip()