# import os
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv(dotenv_path="../.env")

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# def generate_answer(prompt):
#     response = client.chat.completions.create(
#         model="openai/gpt-oss-120b",
#         messages=[
#             {"role": "system", "content": "You answer strictly from provided context."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.1
#     )
#     return response.choices[0].message.content



import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:latest"

def generate_answer(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error: {response.text}")

    result = response.json()
    return result["response"].strip()
