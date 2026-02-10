import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (parent of app/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_openai_api_key() -> str:
    key = OPENAI_API_KEY
    if not key or not key.strip():
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project root."
        )
    return key.strip()
