import os
from dotenv import load_dotenv

load_dotenv()

LLM7_API_KEY = os.getenv("LLM7_API_KEY")

if not LLM7_API_KEY:
    raise RuntimeError("LLM7_API_KEY is missing from .env")
