import os
from dotenv import load_dotenv
load_dotenv()

SKIP_GIT_CHECKPOINT = os.getenv("SKIP_GIT_CHECKPOINT", "0").lower() in ("1", "true", "yes")
load_dotenv()

LLM7_API_KEY = os.getenv("LLM7_API_KEY")
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")

if not LLM7_API_KEY:
    raise RuntimeError("LLM7_API_KEY is missing from .env")
LLM7_MODEL = "gpt-5.4-mini"
MISTRAL_MODEL = "mistral-small-latest"
OLLAMA_CLOUD_MODEL = "gpt-oss:20b"
OLLAMA_LOCAL_KEY = "ollama"
OLLAMA_LOCAL_MODEL="qwen3-coder:30b"
OLLAMA_CLOUD_URL = "https://api.ollama.com/v1"
OLLAMA_LOCAL_URL = "http://localhost:11434/v1"
OLLAMA_CLOUD_KEY="202e38be937b4d5aa20aa065c7b73d12.H2g_eOXtPld02kmuCfDlEPzt"
