import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")
print("Loaded GROQ_API_KEY:", api_key)

# Create Groq client
client = Groq(api_key=api_key)

prompt = """
Generate clean Python code for:
print the Fibonacci sequence up to n terms, where n is 10.
"""

response = client.chat.completions.create(
    model=os.getenv('model', 'groq/compound-mini'),
    messages=[
        {"role": "system", "content": "You are a senior software engineer."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)

print(response.choices[0].message.content)
