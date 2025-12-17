import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_code(prompt: str) -> str:
    response = client.chat.completions.create(
        model="groq/compound-mini",
        messages=[
            {"role": "system", "content": "You generate safe Python data analysis code."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content
