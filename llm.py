
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_answer(question, context):
    prompt = f"""
You are a helpful assistant for a document question-answering system.

Rules:
1. Answer using only the provided document context.
2. Do not make up information.
3. If the answer is not present in the context, say:
   "I could not find the answer in the uploaded documents."
4. Answer only the user's question.
5. Give a clear and simple answer.

Document context:
{context}

User question:
{question}

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=300
        )

        response.raise_for_status()
        result = response.json()

        return result.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama took too long to respond. Check whether the model can run on your computer."
        )

    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Ollama API error: {error}")
