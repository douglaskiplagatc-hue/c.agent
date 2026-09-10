import os
from dotenv import load_dotenv

load_dotenv()


def _env(name, default=""):
    raw = os.getenv(name, default)
    if raw is None:
        raw = default
    return str(raw).split("#", 1)[0].strip()


def _model_env(name, default=""):
    value = _env(name, default)
    prefix = f"{name}="
    if value.upper().startswith(prefix.upper()):
        value = value[len(prefix) :].strip()
    return value.strip("\"'")


def _model_list_env(name, default=""):
    value = _model_env(name, default)
    return tuple(
        model.strip() for model in value.replace(",", "|").split("|") if model.strip()
    )


PROJECT_WORKSPACE = os.path.abspath(_env("PROJECT_WORKSPACE", "workspace"))


SKIP_GIT_CHECKPOINT = _env("SKIP_GIT_CHECKPOINT", "0").lower() in (
    "1",
    "true",
    "yes",
)

LLM7_API_KEY = _env("LLM7_API_KEY")
MISTRAL_API_KEY = _env("MISTRAL_API_KEY")
GOOGLE_API_KEY = _env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")
OPENROUTER_API_KEY = _env("OPENROUTER_API_KEY")

LLM7_BASE_URL = _env("LLM7_BASE_URL", "https://api.llm7.io/v1")
LLM7_MODEL = _env("LLM7_MODEL", "deepseek-r1")
MISTRAL_MODEL = _env("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_BASE_URL = _env("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
GOOGLE_MODEL = _env("GOOGLE_MODEL", _env("GEMINI_MODEL", "models/gemini-3.6-flash"))
OPENROUTER_MODELS = _model_list_env(
    "OPENROUTER_MODEL",
    "openrouter/free|nvidia/nemotron-3-ultra-550b-a55b:free|nvidia/nemotron-3.5-lightning:free|nvidia/nemotron-3.5-ultra:free|nvidia/nemotron-3.5-lightning-a100:free|nvidia/nemotron-3.5-ultra-a100:free|nvidia/nemotron-3.5-lightning-a10:free|nvidia/nemotron-3.5-ultra-a10:free|nvidia/nemotron-3.5-lightning-h100:free|nvidia/nemotron-3.5-ultra-h100:free|nvidia/nemotron-3.5-lightning-h800:free|nvidia/nemotron-3.5-ultra-h800:free|nvidia/nemotron-3.5-lightning-h200:free|nvidia/nemotron-3.5-ultra-h200:free|nvidia/nemotron-3.5-lightning-h400:free|nvidia/nemotron-3.5-ultra-h400:free",
)
OPENROUTER_MODEL = OPENROUTER_MODELS[0]
OPENROUTER_BASE_URL = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

MAX_MODEL_REQUESTS_PER_TASK = int(_env("MAX_MODEL_REQUESTS_PER_TASK", "8"))
MAX_MODEL_REQUESTS_PER_MINUTE = int(_env("MAX_MODEL_REQUESTS_PER_MINUTE", "12"))
MAX_PROVIDER_RETRIES = int(_env("MAX_PROVIDER_RETRIES", "0"))
MAX_PROVIDER_ATTEMPTS_PER_TASK = int(
    _env("MAX_PROVIDER_ATTEMPTS_PER_TASK", "24")
)
ENABLE_LLM_PLANNING = _env("ENABLE_LLM_PLANNING", "0").lower() in (
    "1",
    "true",
    "yes",
)
MAX_AGENT_TURNS = int(_env("MAX_AGENT_TURNS", "20"))
MODEL_REQUEST_TIMEOUT = float(_env("MODEL_REQUEST_TIMEOUT", "30"))
GROQ_MODEL = _env("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_API_KEY = _env("GROQ_API_KEY")
