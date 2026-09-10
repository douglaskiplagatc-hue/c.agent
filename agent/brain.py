import json
import time
import traceback
from collections import deque
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from openai import OpenAI

from agent import tools as agent_tools
from agent.controller import AgentController, TaskPhase
from agent.prompts import PLANNING_PROMPT, SYSTEM_PROMPT
from agent.reporting import Reporter
from agent.state import TaskState
from agent.tools import (
    analyze_python_file,
    create_file,
    edit_file,
    git_checkpoint,
    git_rollback,
    git_status,
    inspect_project,
    list_files,
    patch_file,
    read_file,
    read_project_memory,
    run_command,
    scan_files,
    search_code,
    search_project,
    update_project_memory,
    verify_file_scan,
    verify_project_inventory,
)
from config import (
    ENABLE_LLM_PLANNING,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    LLM7_API_KEY,
    LLM7_BASE_URL,
    LLM7_MODEL,
    MAX_AGENT_TURNS,
    MAX_MODEL_REQUESTS_PER_MINUTE,
    MAX_MODEL_REQUESTS_PER_TASK,
    MAX_PROVIDER_ATTEMPTS_PER_TASK,
    MAX_PROVIDER_RETRIES,
    MISTRAL_API_KEY,
    MISTRAL_BASE_URL,
    MISTRAL_MODEL,
    MODEL_REQUEST_TIMEOUT,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS,
    PROJECT_WORKSPACE,
)

# Module-level reporter – will be replaced in run_agent()
reporter = Reporter(verbose=False)

_model_request_count = 0
_model_request_times = deque()
_provider_attempt_count = 0
_preferred_provider = None
_failed_providers = set()


class ModelRequestLimitExceeded(RuntimeError):
    """Raised before a provider call would exceed the configured request budget."""


def reset_model_request_budget():
    global _model_request_count, _provider_attempt_count
    _model_request_count = 0
    _provider_attempt_count = 0
    _model_request_times.clear()
    _failed_providers.clear()


def consume_model_request(provider_name):
    """Reserve one successful-response slot and count one provider attempt."""
    global _model_request_count, _provider_attempt_count

    if _provider_attempt_count >= MAX_PROVIDER_ATTEMPTS_PER_TASK:
        raise ModelRequestLimitExceeded(
            "Provider attempt limit reached for this task "
            f"({MAX_PROVIDER_ATTEMPTS_PER_TASK})."
        )

    now = time.monotonic()
    while _model_request_times and now - _model_request_times[0] >= 60:
        _model_request_times.popleft()

    if _model_request_count >= MAX_MODEL_REQUESTS_PER_TASK:
        raise ModelRequestLimitExceeded(
            "Model request limit reached for this task "
            f"({MAX_MODEL_REQUESTS_PER_TASK})."
        )

    if len(_model_request_times) >= MAX_MODEL_REQUESTS_PER_MINUTE:
        raise ModelRequestLimitExceeded(
            "Model request rate limit reached "
            f"({MAX_MODEL_REQUESTS_PER_MINUTE} requests per minute)."
        )

    _provider_attempt_count += 1
    _model_request_times.append(now)
    reporter.debug(
        f"[LLM] Attempt {_provider_attempt_count}/"
        f"{MAX_PROVIDER_ATTEMPTS_PER_TASK}; successful budget: "
        f"{_model_request_count}/{MAX_MODEL_REQUESTS_PER_TASK}; "
        f"{MAX_MODEL_REQUESTS_PER_TASK} for task; provider={provider_name}"
    )


def commit_model_request():
    """Count a provider response against the task and minute budgets."""
    global _model_request_count
    _model_request_count += 1


def release_model_request():
    """Release a failed attempt so failures do not consume usage budget."""
    if _model_request_times:
        _model_request_times.pop()


# ============================================================
# MODEL CONFIGURATION
# ============================================================


# ============================================================
# TOOL DEFINITIONS (unchanged)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scan_files",
            "description": (
                "Deterministically scan the configured target project recursively. "
                "Filter by extensions and excluded directory names. Never replace "
                "the configured target root with a workspace alias."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "recursive": {"type": "boolean"},
                    "extensions": {"type": "array", "items": {"type": "string"}},
                    "exclude_directories": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "directory",
                    "recursive",
                    "extensions",
                    "exclude_directories",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_project",
            "description": (
                "Inspect the configured project by reading supported source files "
                "and returning a concrete file, line, class, function, and import inventory."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files recursively inside a project directory. "
                "Use this when you need to understand project structure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Directory relative to the selected target project. "
                            "Use '.' for the selected project root; do not invent '/workspace'."
                        ),
                    },
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the complete contents of a project file. "
                "Always inspect a file before editing it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": ("Search project files for text or code patterns."),
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                    },
                    "query": {
                        "type": "string",
                    },
                },
                "required": [
                    "directory",
                    "query",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_project",
            "description": (
                "Search the project index for relevant files, "
                "Python functions, and classes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": ("Create a new file inside the workspace."),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                    "content": {
                        "type": "string",
                    },
                },
                "required": [
                    "file_path",
                    "content",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace one exact section of an existing file. "
                "The old_text must match the actual current contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                    "old_text": {
                        "type": "string",
                    },
                    "new_text": {
                        "type": "string",
                    },
                },
                "required": [
                    "file_path",
                    "old_text",
                    "new_text",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Safely replace one exact section of a file. "
                "The operation fails if old_text is missing "
                "or appears more than once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                    "old_text": {
                        "type": "string",
                    },
                    "new_text": {
                        "type": "string",
                    },
                },
                "required": [
                    "file_path",
                    "old_text",
                    "new_text",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a terminal command inside the workspace. "
                "Use this for tests, Python scripts, linting, "
                "builds, verification, and inspecting command output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                    },
                    "timeout": {
                        "type": "integer",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": ("Check the Git status of the workspace."),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_checkpoint",
            "description": ("Create a Git checkpoint of current workspace changes."),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                    },
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_rollback",
            "description": ("Discard uncommitted changes in the workspace."),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_python_file",
            "description": (
                "Analyze a Python file and return its actual "
                "function and class definitions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_project_memory",
            "description": ("Read persistent project memory."),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_project_memory",
            "description": ("Update or create a section in persistent project memory."),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string",
                    },
                    "content": {
                        "type": "string",
                    },
                },
                "required": [
                    "section",
                    "content",
                ],
            },
        },
    },
]


# ============================================================
# TOOL REGISTRY
# ============================================================

AVAILABLE_TOOLS = {
    "scan_files": scan_files,
    "inspect_project": inspect_project,
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "search_project": search_project,
    "create_file": create_file,
    "edit_file": edit_file,
    "patch_file": patch_file,
    "run_command": run_command,
    "git_status": git_status,
    "git_checkpoint": git_checkpoint,
    "git_rollback": git_rollback,
    "analyze_python_file": analyze_python_file,
    "read_project_memory": read_project_memory,
    "update_project_memory": update_project_memory,
}


WRITE_TOOLS = {
    "create_file",
    "edit_file",
    "patch_file",
}

DESTRUCTIVE_TOOLS = {
    "git_rollback",
}

INSPECTION_TOOLS = {
    "scan_files",
    "inspect_project",
    "list_files",
    "read_file",
    "search_code",
    "search_project",
    "analyze_python_file",
    "read_project_memory",
    "git_status",
}


# ============================================================
# VERIFICATION COMMAND INDICATORS
# ============================================================

VERIFICATION_INDICATORS = (
    "pytest",
    "unittest",
    "python ",
    "python3 ",
    "py ",
    "ruff",
    "mypy",
    "flake8",
    "pyright",
    "npm test",
    "npm run test",
    "npm run build",
    "npm run lint",
)


# ============================================================
# PATH VALIDATION (added for safety)
# ============================================================


def _get_workspace_root():
    """Return absolute path to the workspace directory."""
    # The server updates this value when the user selects a project folder.
    return Path(PROJECT_WORKSPACE)


def _validate_workspace_path(file_path):
    """
    Ensure file_path is inside the workspace.
    Raises ValueError if not.
    """
    if not file_path:
        raise ValueError("File path is empty.")
    workspace_root = _get_workspace_root().resolve()
    path = Path(file_path)
    if path.is_absolute():
        target = path.resolve()
    else:
        parts = path.parts
        if parts and parts[0].lower() == "workspace":
            parts = parts[1:]
        target = workspace_root.joinpath(*parts).resolve()
    try:
        target.relative_to(workspace_root)
    except ValueError:
        raise ValueError(f"Path '{file_path}' is outside the workspace.")


# ============================================================
# LLM REQUEST WITH MULTI-PROVIDER FALLBACK
# ============================================================


def llm_request_with_retry(
    messages,
    tools=None,
    max_retries=None,
    max_providers=None,
):
    """
    Send an LLM request using multiple providers.

    Provider order (try first → fallback):
    1. LLM7
    2. Google AI Studio
    3. OpenRouter
    4. Mistral

    Non-transient errors (auth, invalid model, quota) cause an immediate
    switch to the next provider without retrying.
    """

    if max_retries is None:
        max_retries = MAX_PROVIDER_RETRIES

    clients = {}
    providers = []

    # --------------------------------------------------------
    # LLM7
    # --------------------------------------------------------
    if LLM7_API_KEY and LLM7_MODEL:
        try:
            clients["llm7"] = OpenAI(
                api_key=LLM7_API_KEY,
                base_url=LLM7_BASE_URL,
                timeout=MODEL_REQUEST_TIMEOUT,
            )
            providers.append(
                {
                    "client": clients["llm7"],
                    "model": LLM7_MODEL,
                    "name": "LLM7",
                    "key": "llm7",
                }
            )
        except Exception as error:
            reporter.debug(f"[LLM] LLM7 client initialization failed: {error}")

    # --------------------------------------------------------
    # Google AI Studio (OpenAI-compatible Gemini endpoint)
    # --------------------------------------------------------
    if GOOGLE_API_KEY and GOOGLE_MODEL:
        try:
            clients["google"] = OpenAI(
                api_key=GOOGLE_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=MODEL_REQUEST_TIMEOUT,
            )
            providers.append(
                {
                    "client": clients["google"],
                    "model": GOOGLE_MODEL,
                    "name": "Google AI Studio",
                    "key": "google",
                }
            )
        except Exception as error:
            reporter.debug(
                f"[LLM] Google AI Studio client initialization failed: {error}"
            )

    # --------------------------------------------------------
    # OpenRouter
    # --------------------------------------------------------
    if OPENROUTER_API_KEY and OPENROUTER_MODELS:
        try:
            clients["openrouter"] = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                timeout=MODEL_REQUEST_TIMEOUT,
            )
            for model in OPENROUTER_MODELS:
                providers.append(
                    {
                        "client": clients["openrouter"],
                        "model": model,
                        "name": f"OpenRouter ({model})",
                        "key": f"openrouter:{model}",
                    }
                )
        except Exception as error:
            reporter.debug(f"[LLM] OpenRouter client initialization failed: {error}")

    # --------------------------------------------------------
    # Mistral
    # --------------------------------------------------------
    if MISTRAL_API_KEY and MISTRAL_MODEL:
        try:
            clients["mistral"] = OpenAI(
                api_key=MISTRAL_API_KEY,
                base_url=MISTRAL_BASE_URL,
                timeout=MODEL_REQUEST_TIMEOUT,
            )
            providers.append(
                {
                    "client": clients["mistral"],
                    "model": MISTRAL_MODEL,
                    "name": "Mistral",
                    "key": "mistral",
                }
            )
        except Exception as error:
            reporter.debug(f"[LLM] Mistral client initialization failed: {error}")

    if not providers:
        raise RuntimeError("No LLM providers are configured. Check your config.py.")

    global _preferred_provider

    if _preferred_provider:
        providers.sort(key=lambda provider: provider["key"] != _preferred_provider)

    providers = [
        provider for provider in providers if provider["key"] not in _failed_providers
    ] or providers

    if max_providers is not None:
        providers = providers[:max_providers]

    last_exception = None
    provider_errors = []

    for provider in providers:
        for attempt in range(max_retries + 1):
            consume_model_request(provider["name"])
            try:
                kwargs = {
                    "model": provider["model"],
                    "messages": messages,
                    "timeout": MODEL_REQUEST_TIMEOUT,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                start = time.time()
                response = provider["client"].chat.completions.create(**kwargs)
                elapsed = time.time() - start
                commit_model_request()
                reporter.debug(f"[LLM] {provider['name']} responded in {elapsed:.2f}s")
                reporter.debug(
                    f"[LLM] Success: {provider['name']} ({provider['model']})"
                )
                _preferred_provider = provider["key"]
                return response

            except Exception as error:
                release_model_request()
                last_exception = error
                _failed_providers.add(provider["key"])
                error_msg = str(error).lower()
                reporter.debug(
                    f"[LLM] {provider['name']} attempt {attempt + 1} failed: {error}"
                )

                # Non-transient errors – switch provider immediately
                non_transient = any(
                    term in error_msg
                    for term in (
                        "insufficient balance",
                        "insufficient_quota",
                        "invalid model",
                        "invalid api key",
                        "authentication",
                        "unauthorized",
                        "permission",
                        "invalid_request",
                        "unexpected role",
                        "invalid_request_message_order",
                        "method not allowed",
                        "model_unavailable",
                        "retired",
                        "402",
                        "405",
                        "410",
                    )
                )
                if non_transient:
                    provider_errors.append(f"{provider['name']}: {error}")
                    reporter.debug(
                        f"[LLM] {provider['name']} returned a non-transient error. Switching provider."
                    )
                    break

                # Transient errors – retry with backoff
                transient = any(
                    term in error_msg
                    for term in (
                        "timeout",
                        "timed out",
                        "connection",
                        "connection error",
                        "remoteprotocolerror",
                        "429",
                        "500",
                        "502",
                        "503",
                        "504",
                        "temporarily unavailable",
                        "rate limit",
                    )
                )
                if not transient:
                    provider_errors.append(f"{provider['name']}: {error}")
                    reporter.debug(
                        f"[LLM] {provider['name']} returned an unrecognized error. Switching provider."
                    )
                    break

                if attempt < max_retries:
                    delay = 2**attempt
                    reporter.debug(f"[LLM] Retrying {provider['name']} in {delay}s...")
                    time.sleep(delay)
                else:
                    provider_errors.append(f"{provider['name']}: {error}")
                    reporter.debug(
                        f"[LLM] {provider['name']} retries exhausted. Switching provider."
                    )

    # All providers failed
    details = "\n".join(provider_errors) if provider_errors else str(last_exception)
    raise RuntimeError("All LLM providers failed.\n" + details) from last_exception


# ============================================================
# TOOL RESULT HANDLING
# ============================================================


def append_tool_result(
    messages,
    tool_call,
    result_text,
):
    """
    Append a tool result.

    IMPORTANT:

    This function must only be called after the assistant
    message containing the corresponding tool call has been
    appended.

    Tool results must remain together before any new user
    message is appended.
    """

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result_text),
        }
    )


# ============================================================
# REPAIR INSTRUCTION
# ============================================================


def append_repair_instruction(
    messages,
    instruction,
):
    """
    Add a repair instruction AFTER ALL tool results from
    the current assistant turn have been appended.

    This prevents:

        assistant
        tool
        user
        tool

    which is invalid.

    Instead we produce:

        assistant
        tool
        tool
        tool
        user
    """

    if not instruction:
        return

    messages.append(
        {
            "role": "user",
            "content": str(instruction),
        }
    )


# ============================================================
# PROJECT MEMORY
# ============================================================


def load_project_memory():

    memory_file = Path("memory/project.md")

    if not memory_file.exists():
        return ""

    try:
        return memory_file.read_text(encoding="utf-8")

    except Exception as error:
        reporter.error(f"[Memory] Failed to read memory: {error}")

        return ""


def extract_memory_section(memory_content, section):
    heading = f"## {section}"
    lines = (memory_content or "").splitlines()
    start = None

    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i
            break

    if start is None:
        return ""

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break

    return "\n".join(lines[start + 1 : end]).strip()


def parse_forbidden_files(
    memory_content,
):

    patterns = []

    in_section = False

    for line in memory_content.splitlines():
        stripped = line.strip()

        if stripped.lower() == "## forbidden files":
            in_section = True

            continue

        if in_section and stripped.startswith("##"):
            break

        if in_section and stripped.startswith("- "):
            pattern = stripped[2:].strip()

            if pattern:
                patterns.append(pattern)

    return patterns


def is_forbidden_file(
    file_path,
    forbidden_patterns,
):

    if not file_path:
        return False

    normalized = str(file_path).replace(
        "\\",
        "/",
    )

    for pattern in forbidden_patterns:
        normalized_pattern = str(pattern).replace(
            "\\",
            "/",
        )

        if fnmatch(
            normalized,
            normalized_pattern,
        ):
            return True

    return False


# ============================================================
# OBJECTIVE CLASSIFICATION
# ============================================================

MODIFICATION_PHRASES = (
    "fix ",
    "modify ",
    "edit ",
    "change ",
    "update ",
    "repair ",
    "implement ",
    "create ",
    "refactor ",
    "remove ",
    "add ",
    "write ",
    "delete ",
    "should contain",
    "must have",
    "needs to",
    "requires ",
    "should include",
    "implement ... function",
    "add ... function",
)


EXPLICIT_READ_ONLY_PHRASES = (
    "do not modify",
    "don't modify",
    "do not change",
    "don't change",
    "read-only",
    "read only",
    "without modifying",
    "without changing",
    "without editing",
    "only inspect",
    "just inspect",
    "analyze only",
)


CONVERSATIONAL_REQUESTS = {
    "hello",
    "hello there",
    "hi",
    "hi there",
    "hey",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "who are you",
    "what can you do",
    "thanks",
    "thank you",
}


def is_conversational_request(objective):
    """Return True for standalone conversation that needs no project context."""
    text = re.sub(r"\s+", " ", (objective or "").lower().strip())
    normalized = text.rstrip(".!?,")

    if normalized in CONVERSATIONAL_REQUESTS:
        return True

    greeting_prefixes = ("hello ", "hi ", "hey ")
    return (
        len(normalized) <= 60
        and normalized.startswith(greeting_prefixes)
        and not objective_requires_modification(normalized)
        and not any(
            marker in normalized
            for marker in (
                "file",
                "code",
                "project",
                "workspace",
                "inspect",
                "check",
                "test",
                "run ",
                "read ",
                "search ",
            )
        )
    )


def objective_requires_modification(
    objective,
):
    """
    Determine whether the given objective requires file modification.
    This is the single, canonical definition (the more comprehensive one).
    """

    text = (objective or "").lower().strip()

    if any(phrase in text for phrase in EXPLICIT_READ_ONLY_PHRASES):
        return False

    if "should contain" in text or "must have" in text:
        return True

    return any(phrase in text for phrase in MODIFICATION_PHRASES)


def is_read_only_task(
    objective,
):

    return not objective_requires_modification(objective)


def is_project_inspection_request(objective):
    """Identify requests that require a complete project inventory first."""

    text = re.sub(r"\s+", " ", (objective or "").lower()).strip()
    return not objective_requires_modification(text) and (
        ("inspect" in text and "project" in text)
        or "project inventory" in text
        or "project structure" in text
        or ("analyze" in text and "project" in text)
    )


def is_file_inventory_request(objective):
    """Identify requests that can be answered entirely by a deterministic scan."""

    text = re.sub(r"\s+", " ", (objective or "").lower()).strip()
    return is_project_inspection_request(text) and (
        "python source" in text
        or "python file" in text
        or "list every python" in text
        or "find every python" in text
    )


def objective_requires_verification(objective):
    """Identify requests whose completion requires recorded verification."""

    text = re.sub(r"\s+", " ", (objective or "").lower()).strip()
    return any(
        marker in text
        for marker in (
            "verify",
            "verification",
            "validated",
            "validate",
            "confirmed by the filesystem",
        )
    )


# ============================================================
# TOOL EXECUTION
# ============================================================


def execute_tool(
    name,
    arguments,
):

    if name not in AVAILABLE_TOOLS:
        raise ValueError(f"Unknown tool: {name}")

    return AVAILABLE_TOOLS[name](**arguments)


# ============================================================
# STATE TRACKING (UPDATED: only record on success)
# ============================================================


def update_state(
    state,
    tool_name,
    arguments,
    result,
):
    """
    Update the state based on tool execution.
    For write tools, we record only if the result indicates success.
    """
    if tool_name == "scan_files":
        if isinstance(result, dict):
            for file_path in result.get("matched_files", []):
                state.record_inspected_file(file_path)

    elif tool_name == "read_file":
        state.record_inspected_file(arguments["file_path"])

    elif tool_name == "inspect_project":
        if isinstance(result, dict):
            for entry in result.get("files", []):
                if entry.get("file"):
                    state.record_inspected_file(entry["file"])

    elif tool_name in WRITE_TOOLS:
        # Check if result is a dict with success=True, or if it's not a dict assume success?
        # We'll consider success if result is not a dict, or if it is a dict and success is True.
        success = True
        if isinstance(result, dict):
            success = result.get("success", True)
        if success:
            state.record_modified_file(arguments["file_path"])

    elif tool_name == "run_command":
        state.record_command(
            arguments.get(
                "command",
                "",
            )
        )


# ============================================================
# PLANNING
# ============================================================


def create_plan(
    objective,
    project_memory,
):

    requires_modification = objective_requires_modification(objective)

    if not requires_modification:
        planning_instruction = """
This is a READ-ONLY task.

The user wants information, inspection,
analysis, or diagnosis.

Do NOT modify files.

The plan must contain only:

1. Relevant files to inspect.
2. Inspection/search tools to use.
3. Information that must be extracted.
4. How findings will be verified without changing files.
"""

    else:
        planning_instruction = """
This is a MODIFICATION task.

Create a precise implementation plan.

Rules:

1. Identify the relevant files.

2. Inspect actual file contents before editing.

3. Modify only files necessary for the objective.

4. Never invent existing code.

5. Make the smallest appropriate implementation change.

6. If a file already exists, use edit_file or patch_file.
   Do NOT use create_file to overwrite an existing file.

7. Verify the actual requested behavior after modification.

8. Repair failures based on actual error output.

9. Do not declare success merely because an edit succeeded.

10. If verification fails, diagnose the actual failure
    before making another modification.
"""

    response = llm_request_with_retry(
        messages=[
            {
                "role": "system",
                "content": (
                    PLANNING_PROMPT
                    + "\n\n"
                    + planning_instruction
                    + "\n\nPROJECT MEMORY:\n"
                    + project_memory
                ),
            },
            {
                "role": "user",
                "content": objective,
            },
        ],
        tools=None,
    )

    return response.choices[0].message.content or ""


# ============================================================
# PYTHON SYNTAX VERIFICATION
# ============================================================


def verify_python_file(
    file_path,
):

    try:
        content = read_file(file_path)

        if not isinstance(
            content,
            str,
        ):
            return (
                False,
                "Unable to read file after modification.",
            )

        compile(
            content,
            file_path,
            "exec",
        )

        return (
            True,
            f"Python syntax verification passed for {file_path}.",
        )

    except SyntaxError as error:
        return (
            False,
            f"Syntax error in {file_path}: {error}",
        )

    except Exception as error:
        return (
            False,
            f"Python verification failed: {error}",
        )


# ============================================================
# TOOL CALL IDENTIFICATION
# ============================================================


def make_tool_call_key(
    name,
    arguments,
):

    try:
        encoded = json.dumps(
            arguments,
            sort_keys=True,
            default=str,
        )

    except Exception:
        encoded = str(arguments)

    return (
        name,
        encoded,
    )


# ============================================================
# VERIFICATION HELPERS
# ============================================================


def command_is_verification(
    command,
):

    if not command:
        return False

    command_lower = command.lower().strip()

    return any(indicator in command_lower for indicator in VERIFICATION_INDICATORS)


def command_succeeded(
    result,
):

    if not isinstance(
        result,
        dict,
    ):
        return True

    return (
        result.get(
            "success",
            True,
        )
        is not False
    )


def command_output(
    result,
):

    if isinstance(
        result,
        dict,
    ):
        stdout = result.get(
            "stdout",
            "",
        )

        stderr = result.get(
            "stderr",
            "",
        )

        return (str(stdout) + "\n" + str(stderr)).strip()

    return str(result)


def verification_output_is_bad(
    output,
):

    if not output:
        return False

    lower = output.lower()

    failure_patterns = (
        "traceback (most recent call last)",
        "syntaxerror:",
        "indentationerror:",
        "exception:",
        "assertionerror:",
        "modulenotfounderror:",
        "importerror:",
        "nameerror:",
        "typeerror:",
        "attributeerror:",
        "filenotfounderror:",
        "permissionerror:",
        "file not found",
        "no such file or directory",
        "failed",
        "error:",
        "errors=",
        "failures=",
    )

    return any(pattern in lower for pattern in failure_patterns)


def objective_output_check(
    objective,
    output,
):

    objective_lower = (objective or "").lower()

    output_lower = (output or "").lower()

    if 'print("hello")' in objective_lower:
        return "hello" in output_lower

    if "output should contain" in objective_lower:
        return None

    if verification_output_is_bad(output):
        return False

    return None


# ============================================================
# EFFICIENT VERIFICATION
# ============================================================


def efficient_verify(
    objective,
    modified_files,
    command_output_text,
    state,
):

    if verification_output_is_bad(command_output_text):
        return False

    explicit_result = objective_output_check(
        objective,
        command_output_text,
    )

    if explicit_result is not None:
        return explicit_result

    if state.llm_verification_count >= 2:
        return False

    state.llm_verification_count += 1

    reporter.debug("\n[Verification] Using LLM verifier...")

    try:
        response = llm_request_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict software verification judge. "
                        "Determine whether the command output provides "
                        "reasonable evidence that the user's objective "
                        "was satisfied. "
                        "Do not assume success merely because the command "
                        "ran. "
                        "Syntax validity alone is not functional evidence. "
                        "Answer TRUE or FALSE only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"OBJECTIVE:\n{objective}\n\n"
                        f"MODIFIED FILES:\n{modified_files}\n\n"
                        f"COMMAND OUTPUT:\n{command_output_text}"
                    ),
                },
            ],
            tools=None,
        )

        answer = (response.choices[0].message.content or "").strip().upper()

        return answer == "TRUE"

    except Exception as error:
        reporter.error(f"[Verification] LLM verifier failed: {error}")

        return False


# ============================================================
# FAILURE RECORDING
# ============================================================


def record_failure(
    state,
    controller,
    name,
    arguments,
    error,
):

    error_text = str(error)

    state.record_error(error_text)

    state.record_failed_action(
        name,
        arguments,
        error_text,
    )

    controller.begin_diagnosis()

    if controller.can_repair():
        controller.begin_repair()

        return True

    controller.rollback()

    controller.fail()

    return False


# ============================================================
# RESULT FORMATTER
# ============================================================


def format_task_result(
    content,
    state,
    status,
):

    return (
        (content or "").strip()
        + "\n\nTask state:"
        + f"\nStatus: {status}"
        + f"\nInspected files: "
        f"{len(state.inspected_files)}" + f"\nModified files: "
        f"{len(state.modified_files)}" + f"\nCommands: "
        f"{len(state.commands)}" + f"\nVerifications: "
        f"{len(state.verifications)}" + f"\nErrors: "
        f"{len(state.errors)}"
    )


# ============================================================
# ROLLBACK SNAPSHOT (UPDATED: safe handling)
# ============================================================


def snapshot_file(
    file_path,
    protected_snapshots,
):
    """
    Snapshot the current content of a file.
    Stores three possible states:
      - 'exists': content is the file content (string)
      - 'missing': file did not exist
      - 'error': read failed (content is None)
    """
    if file_path in protected_snapshots:
        return

    path = Path(file_path)

    if not path.exists():
        protected_snapshots[file_path] = ("missing", None)
        return

    try:
        protected_snapshots[file_path] = ("exists", read_file(file_path))
    except Exception:
        protected_snapshots[file_path] = ("error", None)


def restore_snapshots(
    protected_snapshots,
):
    """
    Restore files from snapshots.
    Only delete if the snapshot state was 'missing'.
    For 'error' we do nothing.
    """
    for file_path, snapshot in protected_snapshots.items():
        try:
            state, content = snapshot
            if state == "missing":
                path = Path(file_path)
                if path.exists():
                    path.unlink()
            elif state == "exists":
                path = Path(file_path)
                path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                path.write_text(
                    str(content),
                    encoding="utf-8",
                )
            # else: 'error' – do nothing
        except Exception as error:
            reporter.error(f"[Rollback] Failed restoring {file_path}: {error}")


# ============================================================
# TASK COMPLETENESS
# ============================================================


def determine_task_completeness(
    objective,
    state,
    last_output="",
):

    requires_modification = getattr(
        state,
        "requires_modification",
        objective_requires_modification(objective),
    )

    functional_verification_passed = getattr(
        state,
        "functional_verification_passed",
        False,
    )

    if not requires_modification:
        return (True, "The read-only objective was inspected and analyzed.")

    if not state.modified_files:
        if functional_verification_passed:
            return (
                True,
                (
                    "The existing implementation was inspected and "
                    "successfully verified as satisfying the objective."
                ),
            )

        return (
            False,
            (
                "The task requires modification, but no required file "
                "has been modified and functional verification has not "
                "established that the existing implementation satisfies "
                "the objective."
            ),
        )

    if not functional_verification_passed:
        return (
            False,
            (
                "Files were modified, but successful functional "
                "verification has not been established."
            ),
        )

    if state.contradictions:
        return (
            False,
            "Contradictions were detected between the objective and observed behavior.",
        )

    return (
        True,
        (
            "The required implementation was modified and functional "
            "verification succeeded."
        ),
    )


# ============================================================
# AGENT
# ============================================================


def run_agent(objective, verbose=False):
    global reporter
    reporter = Reporter(verbose=verbose)

    objective = (objective or "").strip()

    if not objective:
        return "No objective was provided."

    reset_model_request_budget()

    if is_conversational_request(objective):
        response = llm_request_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly coding assistant. "
                        "Respond naturally to the user's conversational message. "
                        "This is not a project task, so do not inspect files, "
                        "use tools, create a plan, or modify the workspace."
                    ),
                },
                {"role": "user", "content": objective},
            ],
            tools=None,
        )
        return response.choices[0].message.content or ""

    requires_modification = objective_requires_modification(objective)
    read_only = not requires_modification

    # ------------------------------------------------------------
    # MEMORY
    # ------------------------------------------------------------

    project_memory = load_project_memory()

    forbidden_patterns = parse_forbidden_files(project_memory)

    # ------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------

    state = TaskState(objective=objective)

    state.requires_modification = requires_modification
    state.llm_verification_count = 0
    state.functional_verification_passed = False
    state.contradictions = []

    controller = AgentController(state, reporter)

    if is_project_inspection_request(objective):
        controller.begin_inspection()
        target_root = agent_tools.WORKSPACE.resolve()
        reporter.debug(f"[Inspection] Target root: {target_root}")

        if is_file_inventory_request(objective):
            scan_arguments = {
                "directory": ".",
                "recursive": True,
                "extensions": [".py"],
                "exclude_directories": [
                    ".git",
                    "venv",
                    ".venv",
                    "__pycache__",
                    "node_modules",
                    "instance",
                    "site-packages",
                    ".tox",
                    "dist",
                    "build",
                ],
            }
            scan = scan_files(**scan_arguments)
            update_state(state, "scan_files", scan_arguments, scan)

            controller.begin_verification()
            verification = verify_file_scan(scan)
            verification_ok = verification.get("success", False)
            state.record_verification("Python file scan verification", verification_ok)

            if not verification_ok:
                state.record_error(
                    verification.get("error", "File scan verification failed.")
                )
                controller.fail()
                return format_task_result(
                    "Deterministic file scan failed verification.",
                    state,
                    "FAILED",
                )

            grouped_lines = []
            for directory, files in scan["grouped_files"].items():
                grouped_lines.append(f"\n{directory}/")
                grouped_lines.extend(f"  - {Path(file).name}" for file in files)
            content = (
                "PROJECT FILE INVENTORY\n\n"
                f"Target root: {scan['target_root']}\n"
                f"Recursive: {scan['recursive']}\n"
                f"Python files: {scan['total_count']}\n"
                "Excluded directories: "
                + ", ".join(scan["excluded_directories"])
                + "\n"
                + "".join(grouped_lines)
            )
            controller.complete()
            return format_task_result(content, state, "COMPLETED")

        inventory = inspect_project()
        for entry in inventory.get("files", []):
            if entry.get("file"):
                state.record_inspected_file(entry["file"])

        controller.begin_verification()
        verification = verify_project_inventory(inventory)
        verification_ok = verification.get("success", False)
        state.record_verification("Project inventory verification", verification_ok)

        if not verification_ok:
            state.record_error(
                verification.get("error", "Inventory verification failed.")
            )
            controller.fail()
            return format_task_result(
                "Project inspection failed verification.\n\n"
                + verification.get("error", "Unknown verification error."),
                state,
                "FAILED",
            )

        try:
            formatted = llm_request_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Format the verified project inventory for the user. "
                            "Summarize the architecture, entry points, source files, "
                            "Python classes/functions, and notable syntax errors. "
                            "Use only the supplied structured data. Do not claim "
                            "commands, edits, or files that are not present."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "target_root": str(target_root),
                                "inventory": inventory,
                                "verification": verification,
                            },
                            indent=2,
                            default=str,
                        ),
                    },
                ],
                tools=None,
                max_retries=0,
                max_providers=1,
            )
        except Exception as error:
            state.record_error(str(error))
            controller.fail()
            return format_task_result(
                "Verified project inventory could not be formatted.\n\n" + str(error),
                state,
                "FAILED",
            )
        content = (
            formatted.choices[0].message.content
            or "Verified project inventory received."
        )
        controller.complete()
        return format_task_result(content, state, "COMPLETED")

    # ------------------------------------------------------------
    # INITIAL PHASE
    # ------------------------------------------------------------

    controller.set_phase(TaskPhase.PLANNING)

    # ------------------------------------------------------------
    # GIT CHECKPOINT
    # ------------------------------------------------------------

    if requires_modification:
        controller.set_phase(TaskPhase.CHECKPOINT)

        try:
            if not controller.prepare_workspace():
                reporter.debug("[Agent] Git checkpoint unavailable.")

        except Exception as error:
            reporter.debug(f"[Agent] Checkpoint preparation failed: {error}")

    # ------------------------------------------------------------
    # PLAN
    # ------------------------------------------------------------

    if ENABLE_LLM_PLANNING:
        try:
            state.plan = create_plan(objective, project_memory)

        except Exception as error:
            state.record_error(str(error))
            state.plan = (
                "Planning service was unavailable. Continue with evidence-driven execution:\n"
                "1. Inspect the workspace and relevant files.\n"
                "2. Make only the requested change, if needed.\n"
                "3. Run focused verification and report the actual result."
            )
            reporter.debug(
                f"[Agent] Planning failed; continuing with fallback plan: {error}"
            )
    else:
        state.plan = (
            "Direct execution mode: inspect only the files relevant to the objective, "
            "make the smallest required change, and run focused verification."
        )

    reporter.debug("\n[Agent Plan]")

    # ------------------------------------------------------------
    # INSPECTION
    # ------------------------------------------------------------

    controller.begin_inspection()

    # ------------------------------------------------------------
    # MODEL CONTEXT
    # ------------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(project_memory=project_memory),
        },
        {
            "role": "user",
            "content": f"""
USER OBJECTIVE:

{objective}

IMPLEMENTATION PLAN:

{state.plan}

TASK TYPE:

{"MODIFICATION" if requires_modification else "READ-ONLY"}

MANDATORY AGENT RULES:

1. Inspect relevant files before modifying them.
2. Never invent file contents.
3. For edit_file or patch_file, use exact text obtained from the actual current file.
4. Never repeat an identical failed edit.
5. Never repeat an identical tool call.
6. Inspection is not completion.
7. Searching is not completion.
8. Reading a file is not completion.
9. A successful modification is not automatically completion.
10. For modification tasks, the required file must actually be modified unless the existing implementation already satisfies the objective and is successfully verified.
11. For modification tasks, the final implementation must be functionally verified.
12. Python syntax verification alone does not prove that the requested behavior works.
13. When the objective requires running a Python file, actually run it.
14. When a verification command fails, inspect the actual error before attempting repair.
15. Make only the changes necessary for the objective.
16. Never modify forbidden files.
17. Read-only tasks must never modify files.
18. If verification fails, diagnose the actual problem, make a targeted repair, and verify again.
19. Do not claim completion merely because a tool call returned successfully.
20. Once the objective has genuinely been verified, stop immediately.
21. For a project-wide inspection request, your first tool call MUST be inspect_project.
    Do not treat list_files alone as project inspection. It is only a path listing.
22. BEFORE reading or editing any file, you MUST call list_files on the workspace directory.
23. NEVER guess a file path without confirming it exists.
24. When the assistant returns multiple tool calls, ALL tool results must be returned to the model BEFORE adding any new user/repair message.
25. Never place a user message between two tool results belonging to the same assistant turn.
26. Syntax verification is a safety check, not proof of functional correctness.
27. A Python command with no output is not automatically proof of functional success.
28. Determine task completeness from actual evidence, not from the number of tools executed.
29. Model requests are limited. Batch independent inspection calls in one response when possible.
30. Prioritize completing the requested change and focused verification over optional exploration.

BEGIN BY INSPECTING THE RELEVANT FILE(S).
""",
        },
    ]

    # ------------------------------------------------------------
    # HISTORY / PROTECTION
    # ------------------------------------------------------------

    recent_tool_call_keys = []
    protected_snapshots = {}
    failed_actions = set()
    pending_repair_instructions = []
    turn_count = 0
    max_turns = MAX_AGENT_TURNS
    inspection_performed = False

    # ============================================================
    # MAIN LOOP
    # ============================================================

    while not state.completed:
        turn_count += 1

        if turn_count > max_turns:
            reporter.debug(f"\n[Agent] Maximum of {max_turns} turns reached.")

            restore_snapshots(protected_snapshots)

            try:
                controller.rollback()
            except Exception:
                pass

            try:
                controller.fail()
            except Exception:
                pass

            state.completed = True

            return format_task_result(
                (f"The agent stopped after reaching the maximum of {max_turns} turns."),
                state,
                "FAILED",
            )

        if state.attempts >= state.max_attempts:
            reporter.debug("\n[Agent] Maximum repair attempts reached.")

            restore_snapshots(protected_snapshots)

            try:
                controller.rollback()

            except Exception:
                pass

            try:
                controller.fail()

            except Exception:
                pass

            state.completed = True

            return format_task_result(
                (
                    "The agent stopped after reaching "
                    f"the maximum of {state.max_attempts} "
                    "repair attempts."
                ),
                state,
                "FAILED",
            )

        try:
            response = llm_request_with_retry(messages=messages, tools=TOOLS)

        except Exception as error:
            state.record_error(str(error))

            try:
                controller.fail()

            except Exception:
                pass

            return format_task_result(
                (f"The model request failed.\n\nError: {error}"), state, "FAILED"
            )

        assistant_message = response.choices[0].message

        tool_calls = assistant_message.tool_calls or []

        assistant_payload = {
            "role": "assistant",
            "content": assistant_message.content or "",
        }

        if tool_calls:
            assistant_payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]

        messages.append(assistant_payload)

        if not tool_calls:
            content = assistant_message.content or ""

            if read_only:
                if not inspection_performed:
                    reporter.debug(
                        "\n[Agent] Read-only task with no inspection; blocking completion."
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have not performed any inspection. "
                                "To complete this read-only task, you MUST call at least one inspection tool "
                                "(e.g., list_files, read_file, search_code, analyze_python_file, etc.) "
                                "and examine the relevant files before reporting your findings."
                            ),
                        }
                    )
                    continue

                if objective_requires_verification(objective) and not any(
                    item.get("success") for item in state.verifications
                ):
                    reporter.debug(
                        "\n[Agent] Verification required but no successful verification was recorded."
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Verification is required for this task. "
                                "Do not report completion until a deterministic verification "
                                "operation succeeds and is recorded in task state."
                            ),
                        }
                    )
                    continue

                try:
                    controller.complete()

                except Exception:
                    pass

                state.completed = True

                result_text = format_task_result(content, state, "COMPLETED")

                reporter.user(result_text)

                return result_text

            complete, reason = determine_task_completeness(
                objective,
                state,
                content,
            )

            if complete:
                try:
                    controller.complete()

                except Exception:
                    pass

                state.completed = True

                result_text = format_task_result(content or reason, state, "COMPLETED")

                reporter.user(result_text)

                return result_text

            reporter.debug("\n[Agent] Completion blocked.")

            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The task is NOT proven complete.\n\n"
                        f"COMPLETENESS ANALYSIS:\n{reason}\n\n"
                        "Continue working.\n\n"
                        "If the existing implementation already "
                        "satisfies the objective, inspect and "
                        "functionally verify it.\n\n"
                        "If it does not satisfy the objective, "
                        "make the necessary targeted modification "
                        "and verify the actual requested behavior.\n\n"
                        "Do NOT claim completion without evidence."
                    ),
                }
            )

            continue

        turn_results = []

        for tool_call in tool_calls:
            error_text = None
            output = None

            name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments)

            except Exception as error:
                error_text = "Invalid tool arguments: " + str(error)

                reporter.debug(f"\n[Agent] {error_text}")

                state.record_error(error_text)

                result_text = json.dumps(
                    {
                        "success": False,
                        "error": error_text,
                    }
                )

                append_tool_result(
                    messages,
                    tool_call,
                    result_text,
                )

                pending_repair_instructions.append(
                    "TOOL ARGUMENT ERROR.\n"
                    f"Tool: {name}\n"
                    f"Actual error: {error_text}\n"
                    "Do not repeat the malformed call."
                )

                turn_results.append(
                    {
                        "tool": name,
                        "result": {
                            "success": False,
                            "error": error_text,
                        },
                        "error": error_text,
                    }
                )

                continue

            reporter.debug(f"\n[Agent → {name}]")

            reporter.debug(f"Arguments: {arguments}")

            # ----------------------------------------------------
            # Read-only enforcement: block write/destructive tools
            # ----------------------------------------------------
            if read_only and (
                name in WRITE_TOOLS
                or name in DESTRUCTIVE_TOOLS
                or name == "run_command"
            ):
                error = (
                    f"Tool '{name}' is not allowed in read-only mode. "
                    "Use inspection tools only; commands are blocked because "
                    "shell commands cannot be proven non-mutating."
                )
                reporter.debug(f"\n[Agent] Read-only block: {error}")

                result_text = json.dumps(
                    {
                        "success": False,
                        "error": error,
                    }
                )
                append_tool_result(messages, tool_call, result_text)
                pending_repair_instructions.append(
                    f"READ-ONLY VIOLATION: {error}\nStick to inspection tools only."
                )
                state.record_failed_action(name, arguments, error)
                continue

            # ----------------------------------------------------
            # Path validation for tools with file_path
            # ----------------------------------------------------
            if "file_path" in arguments:
                try:
                    _validate_workspace_path(arguments["file_path"])
                except ValueError as e:
                    error = str(e)
                    reporter.debug(f"\n[Agent] Path validation failed: {error}")
                    result_text = json.dumps(
                        {
                            "success": False,
                            "error": error,
                        }
                    )
                    append_tool_result(messages, tool_call, result_text)
                    pending_repair_instructions.append(
                        f"INVALID PATH: {error}\n"
                        "You must provide a path that is inside the workspace."
                    )
                    state.record_failed_action(name, arguments, error)
                    continue

            call_key = make_tool_call_key(name, arguments)

            if call_key in recent_tool_call_keys:
                error = "This exact tool call was already executed. Do not repeat it."

                reporter.debug("\n[Agent] Duplicate tool call blocked.")

                state.record_failed_action(
                    name,
                    arguments,
                    error,
                )

                result_text = json.dumps(
                    {
                        "success": False,
                        "error": error,
                    }
                )

                append_tool_result(
                    messages,
                    tool_call,
                    result_text,
                )

                pending_repair_instructions.append(
                    "DUPLICATE TOOL CALL.\n"
                    f"Tool: {name}\n"
                    f"Arguments: {arguments}\n"
                    "Do not repeat it."
                )

                turn_results.append(
                    {
                        "tool": name,
                        "result": {
                            "success": False,
                            "error": error,
                        },
                        "error": error,
                    }
                )

                continue

            recent_tool_call_keys.append(call_key)

            if len(recent_tool_call_keys) > 30:
                recent_tool_call_keys.pop(0)

            if name in WRITE_TOOLS:
                file_path = arguments.get("file_path")

                if is_forbidden_file(
                    file_path,
                    forbidden_patterns,
                ):
                    error = (
                        f"Modification of '{file_path}' "
                        "is prohibited by project memory."
                    )

                    reporter.debug("\n[Agent] Forbidden modification blocked.")

                    state.record_error(error)

                    state.record_failed_action(
                        name,
                        arguments,
                        error,
                    )

                    result_text = json.dumps(
                        {
                            "success": False,
                            "error": error,
                        }
                    )

                    append_tool_result(
                        messages,
                        tool_call,
                        result_text,
                    )

                    pending_repair_instructions.append(
                        f"FORBIDDEN FILE.\nFile: {file_path}\nDo not modify it."
                    )

                    turn_results.append(
                        {
                            "tool": name,
                            "result": {
                                "success": False,
                                "error": error,
                            },
                            "error": error,
                        }
                    )

                    continue

            if name in {
                "edit_file",
                "patch_file",
            }:
                patch_key = (
                    arguments.get("file_path"),
                    arguments.get("old_text"),
                )

                if patch_key in failed_actions:
                    error = "This exact edit already failed. Do not repeat it."

                    reporter.debug("\n[Agent] Repeated failed edit blocked.")

                    state.record_failed_action(
                        name,
                        arguments,
                        error,
                    )

                    result_text = json.dumps(
                        {
                            "success": False,
                            "error": error,
                        }
                    )

                    append_tool_result(
                        messages,
                        tool_call,
                        result_text,
                    )

                    pending_repair_instructions.append(
                        "REPEATED FAILED EDIT.\n"
                        f"Tool: {name}\n"
                        f"File: {arguments.get('file_path')}\n"
                        "Read the current file and determine "
                        "the correct replacement."
                    )

                    turn_results.append(
                        {
                            "tool": name,
                            "result": {
                                "success": False,
                                "error": error,
                            },
                            "error": error,
                        }
                    )

                    continue

            if name in WRITE_TOOLS:
                file_path = arguments.get("file_path")

                snapshot_file(file_path, protected_snapshots)

            result = None
            tool_failed = False

            try:
                result = execute_tool(name, arguments)

                result_success = not isinstance(result, dict) or result.get(
                    "success", True
                )
                if name in INSPECTION_TOOLS and result_success:
                    inspection_performed = True

                update_state(
                    state,
                    name,
                    arguments,
                    result,
                )

                result_text = json.dumps(
                    result,
                    indent=2,
                    default=str,
                )

                reporter.debug("\n[Tool successful]")

            except FileExistsError as error:
                tool_failed = True

                error_text = str(error)

                reporter.debug("\n[Tool failure]")

                reporter.error(error_text)

                traceback.print_exc()

                state.record_error(error_text)

                state.record_failed_action(
                    name,
                    arguments,
                    error_text,
                )

                result = {
                    "success": False,
                    "error": error_text,
                    "hint": (
                        "The file already exists. "
                        "Use read_file to inspect it, "
                        "then use edit_file or patch_file."
                    ),
                }

                result_text = json.dumps(
                    result,
                    indent=2,
                )

                pending_repair_instructions.append(
                    "TOOL FAILURE.\n"
                    f"Tool: {name}\n"
                    f"Arguments: {arguments}\n"
                    f"Actual error: {error_text}\n"
                    "Do not repeat the failed call. "
                    "Inspect and make one targeted correction."
                )

            except Exception as error:
                tool_failed = True

                error_text = str(error)

                reporter.debug("\n[Tool failure]")

                reporter.error(error_text)

                traceback.print_exc()

                state.record_error(error_text)

                state.record_failed_action(
                    name,
                    arguments,
                    error_text,
                )

                result = {
                    "success": False,
                    "error": error_text,
                    "instruction": ("The tool failed. Do not blindly repeat it."),
                }

                result_text = json.dumps(
                    result,
                    indent=2,
                )

                if name in {
                    "edit_file",
                    "patch_file",
                }:
                    failed_actions.add(
                        (
                            arguments.get("file_path"),
                            arguments.get("old_text"),
                        )
                    )

                pending_repair_instructions.append(
                    "TOOL FAILURE.\n"
                    f"Tool: {name}\n"
                    f"Arguments: {arguments}\n"
                    f"Actual error: {error_text}\n"
                    "Diagnose, inspect, make ONE targeted "
                    "correction, and continue."
                )

            append_tool_result(
                messages,
                tool_call,
                result_text,
            )

            if (
                name in WRITE_TOOLS
                and isinstance(
                    result,
                    dict,
                )
                and result.get("success") is False
            ):
                file_path = arguments.get("file_path")

                error = (
                    result.get("error")
                    or result.get("stderr")
                    or "File modification failed."
                )

                reporter.debug("\n[Agent] File modification failed.")

                reporter.error(f"Error: {error}")

                state.record_error(str(error))

                state.record_failed_action(
                    name,
                    arguments,
                    str(error),
                )

                if name in {
                    "edit_file",
                    "patch_file",
                }:
                    failed_actions.add(
                        (
                            file_path,
                            arguments.get("old_text"),
                        )
                    )

                pending_repair_instructions.append(
                    "FILE MODIFICATION FAILED.\n"
                    f"File: {file_path}\n"
                    f"Error: {error}\n"
                    "Read the current file and make one "
                    "targeted correction."
                )

            elif name in WRITE_TOOLS and not tool_failed:
                file_path = arguments.get("file_path")

                reporter.debug("\n[File modification successful]")

                if file_path and file_path.endswith(".py"):
                    ok, message = verify_python_file(file_path)

                    if ok:
                        reporter.debug(f"\n[Syntax verification] {message}")

                    else:
                        reporter.error(f"\n[Syntax verification failed] {message}")

                        state.record_error(message)

                        state.record_failed_action(
                            name,
                            arguments,
                            message,
                        )

                        failed_actions.add(
                            (
                                file_path,
                                arguments.get("old_text"),
                            )
                        )

                        pending_repair_instructions.append(
                            "PYTHON SYNTAX VERIFICATION FAILED.\n"
                            f"File: {file_path}\n"
                            f"Error: {message}\n"
                            "Do NOT declare completion. "
                            "Fix syntax and verify again."
                        )

            if name == "run_command":
                command = arguments.get(
                    "command",
                    "",
                )

                output = command_output(result)

                succeeded = command_succeeded(result)

                is_verification = command_is_verification(command)

                reporter.debug(f"\n[Command] {command}")

                reporter.debug(f"[Command success] {succeeded}")

                if output:
                    reporter.debug("\n[Command output]")

                    reporter.debug(output)

                if not succeeded:
                    error = (
                        result.get("stderr")
                        or result.get("error")
                        or output
                        or "Command failed."
                    )

                    reporter.error("\n[Agent] Verification/command failed.")

                    state.record_error(str(error))

                    state.record_failed_action(
                        name,
                        arguments,
                        str(error),
                    )

                    pending_repair_instructions.append(
                        "COMMAND FAILED.\n"
                        f"Command: {command}\n"
                        f"Output: {output}\n"
                        "Diagnose the failure and make one "
                        "targeted repair."
                    )

                    state.attempts += 1

                elif is_verification:
                    reporter.debug("\n[Verification command detected]")

                    output_lower = output.lower()

                    verified = False

                    if (
                        "all tests passed" in output_lower
                        or "all tests pass" in output_lower
                    ):
                        verified = True

                        reporter.debug(
                            "[Verification] Deterministic success: All tests passed."
                        )

                    elif (
                        ("passed" in output_lower or "pass" in output_lower)
                        and "failed" not in output_lower
                        and "error" not in output_lower
                        and "errors=" not in output_lower
                        and "failures=" not in output_lower
                    ):
                        if (
                            "test" in output_lower
                            or "pytest" in output_lower
                            or "unittest" in output_lower
                        ):
                            verified = True

                            reporter.debug(
                                "[Verification] Output indicates successful tests."
                            )

                    if not verified:
                        if succeeded and not verification_output_is_bad(output):
                            explicit_result = objective_output_check(
                                objective,
                                output,
                            )

                            if explicit_result is True:
                                verified = True

                            elif (
                                command.strip().startswith(
                                    (
                                        "python ",
                                        "python3 ",
                                        "py ",
                                    )
                                )
                                and output.strip()
                            ):
                                verified = efficient_verify(
                                    objective,
                                    state.modified_files,
                                    output,
                                    state,
                                )

                            else:
                                verified = efficient_verify(
                                    objective,
                                    state.modified_files,
                                    output,
                                    state,
                                )

                    if verified:
                        state.functional_verification_passed = True

                        state.record_verification(command, True)

                        try:
                            controller.begin_verification()

                        except Exception:
                            pass

                        reporter.debug("\n[Verification successful]")

                        try:
                            controller.complete()

                        except Exception:
                            pass

                        state.completed = True

                    else:
                        reporter.error("\n[Verification failed]")

                        error = (
                            "Verification output did not provide "
                            "sufficient evidence that the objective "
                            "was satisfied."
                        )

                        state.record_error(error)

                        state.record_failed_action(
                            name,
                            arguments,
                            error,
                        )

                        pending_repair_instructions.append(
                            "VERIFICATION FAILED.\n"
                            f"Objective: {objective}\n"
                            f"Command: {command}\n"
                            f"Output: {output}\n"
                            "Do NOT claim completion. "
                            "Diagnose and repair."
                        )

                        state.attempts += 1

            turn_results.append(
                {
                    "tool": name,
                    "arguments": arguments,
                    "result": result,
                    "error": error_text,
                    "command_output": (output if name == "run_command" else None),
                }
            )

            process_agent_intelligence(
                state=state,
                controller=controller,
                objective=objective,
                tool_name=name,
                arguments=arguments,
                result=result,
                error=error_text,
                command_output_text=(output if name == "run_command" else None),
            )

            if state.completed:
                break

        if state.completed:
            break

        inject_intelligence_context(
            messages,
            state,
            objective,
        )

        if pending_repair_instructions:
            combined = "\n\n".join(pending_repair_instructions)

            append_repair_instruction(
                messages,
                combined,
            )

            pending_repair_instructions.clear()

        if state.completed:
            break

    complete, reason = determine_task_completeness(
        objective,
        state,
    )

    if requires_modification and not complete:
        result_text = format_task_result(
            (f"Task could not be safely completed.\n\nReason: {reason}"),
            state,
            "FAILED",
        )

        reporter.user(result_text)

        return result_text

    result_text = format_task_result(
        "Task completed.",
        state,
        "COMPLETED",
    )

    reporter.user(result_text)

    return result_text


# ============================================================
# PHASE 2: AGENT INTELLIGENCE (complete, unchanged)
# ============================================================

import re


class ShortTermMemory:
    def __init__(self, max_entries=20):
        self.history = []
        self.context = {
            "phase": None,
            "last_tool": None,
            "last_error": None,
            "last_diagnosis": None,
            "last_repair": None,
            "last_verification": None,
            "current_goal": None,
            "goal_status": "unknown",
        }
        self.max_entries = max_entries

    def add(self, tool_name, arguments, reflection, result=None, error=None):
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "arguments": arguments,
            "reflection": reflection,
            "result": self._safe_result(result),
            "error": str(error) if error else None,
        }
        self.history.append(entry)
        if len(self.history) > self.max_entries:
            self.history.pop(0)
        self.context["last_tool"] = tool_name
        if error:
            self.context["last_error"] = str(error)

    def _safe_result(self, result):
        if result is None:
            return None
        if isinstance(result, str):
            return result[:1000]
        if isinstance(result, dict):
            safe = {}
            for key, value in result.items():
                try:
                    text = str(value)
                    if len(text) > 500:
                        text = text[:500] + "...[truncated]"
                    safe[key] = text
                except Exception:
                    safe[key] = "<unserializable>"
            return safe
        try:
            return str(result)[:1000]
        except Exception:
            return "<unserializable result>"

    def get_recent(self, n=5):
        if not self.history:
            return []
        return self.history[-n:]

    def set_context(self, key, value):
        self.context[key] = value

    def get_context(self, key, default=None):
        return self.context.get(key, default)

    def last_tool(self):
        if not self.history:
            return None
        return self.history[-1]["tool"]

    def last_error(self):
        return self.context.get("last_error")

    def count_tool(self, tool_name):
        return sum(1 for entry in self.history if entry["tool"] == tool_name)

    def repeated_action(self, tool_name, arguments, window=5):
        recent = self.history[-window:]
        for entry in recent:
            if entry["tool"] != tool_name:
                continue
            if entry["arguments"] == arguments:
                return True
        return False

    def format_for_model(self, limit=8):
        recent = self.get_recent(limit)
        if not recent:
            return "No previous actions recorded."
        lines = []
        for index, entry in enumerate(recent, 1):
            tool = entry["tool"]
            reflection = entry["reflection"]
            error = entry.get("error")
            line = f"{index}. TOOL={tool} | REFLECTION={reflection}"
            if error:
                line += f" | ERROR={error}"
            lines.append(line)
        return "\n".join(lines)


def initialize_agent_intelligence(state, objective):
    if not hasattr(state, "short_memory"):
        state.short_memory = ShortTermMemory(max_entries=20)
    if not hasattr(state, "reflections"):
        state.reflections = []
    if not hasattr(state, "diagnoses"):
        state.diagnoses = []
    if not hasattr(state, "repair_strategies"):
        state.repair_strategies = []
    if not hasattr(state, "action_confidence"):
        state.action_confidence = []
    if not hasattr(state, "goal_progress"):
        state.goal_progress = {
            "inspection": False,
            "modification": False,
            "verification": False,
            "repair": False,
        }
    if not hasattr(state, "contradictions"):
        state.contradictions = []
    if not hasattr(state, "functional_verification_passed"):
        state.functional_verification_passed = False
    if not hasattr(state, "llm_verification_count"):
        state.llm_verification_count = 0
    state.short_memory.set_context("current_goal", objective)


def summarize_tool_result(tool_name, result):
    if result is None:
        return "No result returned."
    if isinstance(result, dict):
        success = result.get("success", True)
        summary_parts = [
            f"Tool: {tool_name}",
            f"Success: {success}",
        ]
        if "error" in result:
            summary_parts.append(f"Error: {str(result['error'])[:500]}")
        if "stdout" in result:
            stdout = str(result.get("stdout", ""))
            if len(stdout) > 1000:
                stdout = stdout[:1000] + "\n...[output truncated]"
            if stdout:
                summary_parts.append(f"STDOUT:\n{stdout}")
        if "stderr" in result:
            stderr = str(result.get("stderr", ""))
            if len(stderr) > 1000:
                stderr = stderr[:1000] + "\n...[stderr truncated]"
            if stderr:
                summary_parts.append(f"STDERR:\n{stderr}")
        return "\n".join(summary_parts)
    text = str(result)
    if len(text) > 1500:
        text = text[:1500] + "\n...[result truncated]"
    return f"Tool: {tool_name}\nResult:\n{text}"


def diagnose_error(error_text, command_output=None):
    error_text = str(error_text or "")
    output = str(command_output or "")
    combined = (error_text + "\n" + output).lower()

    if "syntaxerror" in combined or "indentationerror" in combined:
        return {
            "category": "syntax",
            "severity": "high",
            "suggestion": (
                "Inspect the exact file and repair the "
                "reported syntax or indentation problem."
            ),
        }
    if "modulenotfounderror" in combined or "no module named" in combined:
        return {
            "category": "missing_dependency",
            "severity": "high",
            "suggestion": (
                "Determine whether the module is missing, "
                "the import path is wrong, or the environment "
                "is incorrect before changing code."
            ),
        }
    if "importerror" in combined:
        return {
            "category": "import",
            "severity": "high",
            "suggestion": (
                "Inspect the import and verify that the "
                "requested object actually exists."
            ),
        }
    if (
        "assertionerror" in combined
        or "assertion failed" in combined
        or "test failed" in combined
    ):
        return {
            "category": "test_failure",
            "severity": "medium",
            "suggestion": (
                "Compare the expected value with the actual "
                "value and repair the implementation to satisfy "
                "the explicit test requirement."
            ),
        }
    if "typeerror" in combined:
        return {
            "category": "type_error",
            "severity": "high",
            "suggestion": (
                "Inspect the actual values and function "
                "arguments involved in the TypeError."
            ),
        }
    if "nameerror" in combined:
        return {
            "category": "name_error",
            "severity": "high",
            "suggestion": (
                "Check variable and function names and "
                "verify that the referenced symbol exists."
            ),
        }
    if "attributeerror" in combined:
        return {
            "category": "attribute_error",
            "severity": "high",
            "suggestion": (
                "Inspect the actual object type and verify "
                "that the requested attribute or method exists."
            ),
        }
    if (
        "filenotfounderror" in combined
        or "file not found" in combined
        or "no such file or directory" in combined
    ):
        return {
            "category": "missing_file",
            "severity": "medium",
            "suggestion": (
                "Confirm the actual project structure before "
                "creating or modifying a path."
            ),
        }
    if (
        "permissionerror" in combined
        or "permission denied" in combined
        or "access denied" in combined
    ):
        return {
            "category": "permission",
            "severity": "high",
            "suggestion": (
                "Check whether the file or directory is "
                "accessible before attempting another write."
            ),
        }
    if "timeout" in combined or "timed out" in combined:
        return {
            "category": "timeout",
            "severity": "medium",
            "suggestion": (
                "Determine whether the operation is genuinely "
                "slow or whether an external service is unavailable."
            ),
        }
    if "429" in combined or "rate limit" in combined or "too many requests" in combined:
        return {
            "category": "rate_limit",
            "severity": "medium",
            "suggestion": (
                "Do not repeatedly send the same request. "
                "Use retry/backoff or another provider."
            ),
        }
    if (
        "401" in combined
        or "unauthorized" in combined
        or "invalid api key" in combined
        or "authentication" in combined
    ):
        return {
            "category": "authentication",
            "severity": "high",
            "suggestion": (
                "Check provider credentials and configuration "
                "rather than repeatedly retrying."
            ),
        }
    if (
        "402" in combined
        or "insufficient balance" in combined
        or "insufficient quota" in combined
    ):
        return {
            "category": "provider_quota",
            "severity": "high",
            "suggestion": (
                "The provider cannot service the request. "
                "Switch provider instead of retrying."
            ),
        }
    if (
        "unexpected role" in combined
        or "invalid_request_message_order" in combined
        or "tool_call_id" in combined
    ):
        return {
            "category": "tool_protocol",
            "severity": "critical",
            "suggestion": (
                "Repair the conversation message ordering. "
                "Tool results must follow the assistant tool-call "
                "message that requested them."
            ),
        }
    if (
        "connection error" in combined
        or "connection refused" in combined
        or "connection reset" in combined
    ):
        return {
            "category": "connection",
            "severity": "medium",
            "suggestion": (
                "Check whether the external service is running "
                "and reachable before retrying."
            ),
        }
    return {
        "category": "unknown",
        "severity": "unknown",
        "suggestion": (
            "Inspect the complete error and relevant files before deciding on a repair."
        ),
    }


def reflect_on_action(state, controller, tool_name, arguments, result, error=None):
    if not hasattr(state, "short_memory"):
        state.short_memory = ShortTermMemory()

    if error:
        diagnosis = diagnose_error(str(error))
        reflection = (
            f"Action '{tool_name}' failed. "
            f"Category: {diagnosis['category']}. "
            f"Severity: {diagnosis['severity']}. "
            f"Next strategy: {diagnosis['suggestion']}"
        )
        state.short_memory.set_context("last_error", str(error))
        state.short_memory.set_context("last_diagnosis", diagnosis)
        if not hasattr(state, "diagnoses"):
            state.diagnoses = []
        state.diagnoses.append(diagnosis)
    else:
        success = result.get("success", True) if isinstance(result, dict) else True
        if success:
            summary = summarize_tool_result(tool_name, result)
            reflection = f"Action '{tool_name}' succeeded. {summary[:700]}"
        else:
            error_value = (
                result.get("error", "unknown error")
                if isinstance(result, dict)
                else "unknown error"
            )
            diagnosis = diagnose_error(str(error_value))
            reflection = (
                f"Action '{tool_name}' returned a failure. "
                f"Category: {diagnosis['category']}. "
                f"Next strategy: {diagnosis['suggestion']}"
            )
            state.short_memory.set_context("last_error", str(error_value))
            state.short_memory.set_context("last_diagnosis", diagnosis)
            if not hasattr(state, "diagnoses"):
                state.diagnoses = []
            state.diagnoses.append(diagnosis)

    state.short_memory.add(
        tool_name=tool_name,
        arguments=arguments,
        reflection=reflection,
        result=result,
        error=error,
    )
    if not hasattr(state, "reflections"):
        state.reflections = []
    state.reflections.append(reflection)

    if tool_name in INSPECTION_TOOLS:
        state.goal_progress["inspection"] = True
    if tool_name in WRITE_TOOLS:
        state.goal_progress["modification"] = True
    if tool_name == "run_command":
        command = arguments.get("command", "")
        if command_is_verification(command):
            state.goal_progress["verification"] = True

    return reflection


def detect_contradictions(objective, result):
    objective_text = (objective or "").lower()
    output_text = (result or "").lower()
    contradictions = []

    quoted_values = re.findall(r'"([^"]+)"', objective or "")
    for expected in quoted_values:
        expected_lower = expected.lower()
        if (
            expected_lower
            and expected_lower not in output_text
            and (
                "expect" in objective_text
                or "return" in objective_text
                or "output" in objective_text
            )
        ):
            contradictions.append(
                f"Expected explicit value '{expected}' was not found in the observed output."
            )

    if (
        "test passed: false" in output_text
        or "test failed" in output_text
        or "failures=" in output_text
        or "errors=" in output_text
    ):
        contradictions.append("The verification output indicates failure.")

    expected_match = re.search(r"expected\s*:\s*(.+)", output_text, re.IGNORECASE)
    result_match = re.search(r"result\s*:\s*(.+)", output_text, re.IGNORECASE)
    if expected_match and result_match:
        expected_value = expected_match.group(1).strip()
        actual_value = result_match.group(1).strip()
        if expected_value != actual_value:
            contradictions.append("Observed result does not match the expected result.")

    return contradictions


def calculate_goal_progress(state):
    progress = getattr(state, "goal_progress", {})
    score = 0
    if progress.get("inspection"):
        score += 25
    if progress.get("modification"):
        score += 25
    if progress.get("verification"):
        score += 25
    if progress.get("repair"):
        score += 25
    return score


def evaluate_action_confidence(state, tool_name, arguments):
    reasons = []
    confidence = 1.0
    memory = state.short_memory

    if memory.repeated_action(tool_name, arguments):
        confidence -= 0.6
        reasons.append("The same action was recently attempted.")

    if memory.get_context("last_error"):
        confidence -= 0.1
        reasons.append("A previous action failed and requires awareness.")

    if tool_name in WRITE_TOOLS and not state.goal_progress.get("inspection", False):
        confidence -= 0.5
        reasons.append("Modification attempted before inspection.")

    if tool_name == "run_command":
        command = arguments.get("command", "")
        if command_is_verification(command):
            reasons.append("Command appears to be a verification command.")

    confidence = max(0.0, min(1.0, confidence))
    result = {
        "confidence": round(confidence, 2),
        "reasons": reasons,
    }
    if not hasattr(state, "action_confidence"):
        state.action_confidence = []
    state.action_confidence.append(
        {
            "tool": tool_name,
            "confidence": result["confidence"],
            "reasons": reasons,
        }
    )
    return result


def generate_repair_strategy(state, error_text, command_output=None):
    diagnosis = diagnose_error(error_text, command_output)
    category = diagnosis["category"]
    suggestion = diagnosis["suggestion"]

    strategies = {
        "syntax": (
            "Read the affected file, locate the exact syntax "
            "problem, make one minimal correction, then run "
            "syntax verification."
        ),
        "missing_dependency": (
            "Determine whether the dependency is genuinely "
            "missing or whether the import path is wrong. "
            "Do not modify unrelated files."
        ),
        "import": (
            "Inspect the import and the module structure. "
            "Repair the import path or referenced symbol."
        ),
        "test_failure": (
            "Compare the actual result against the expected "
            "result. Modify the implementation to satisfy "
            "the explicit test requirement."
        ),
        "type_error": (
            "Inspect the function arguments and actual runtime "
            "types. Make the smallest correction that resolves "
            "the type mismatch."
        ),
        "name_error": (
            "Inspect the affected file and verify the missing "
            "symbol. Correct the name or definition."
        ),
        "attribute_error": (
            "Inspect the actual object and determine whether "
            "the attribute or method exists."
        ),
        "missing_file": (
            "List the workspace and confirm the actual path "
            "before creating or modifying anything."
        ),
        "permission": (
            "Determine why the file is inaccessible before "
            "attempting another modification."
        ),
        "timeout": (
            "Determine whether the operation is genuinely slow "
            "or whether the external dependency is unavailable."
        ),
        "rate_limit": (
            "Stop repeating the same request. Wait or switch "
            "provider according to the retry policy."
        ),
        "authentication": (
            "Check provider configuration and credentials. "
            "Do not repeatedly retry invalid credentials."
        ),
        "provider_quota": (
            "Stop using the unavailable provider and allow "
            "the fallback provider system to continue."
        ),
        "tool_protocol": (
            "Repair the message ordering. Ensure every assistant "
            "tool call is followed by its corresponding tool "
            "result before adding a new conversational message."
        ),
        "connection": (
            "Check whether the service is available. If not, "
            "use another configured provider or tool."
        ),
    }
    strategy = strategies.get(category, suggestion)
    repair = {
        "category": category,
        "severity": diagnosis["severity"],
        "strategy": strategy,
    }
    if not hasattr(state, "repair_strategies"):
        state.repair_strategies = []
    state.repair_strategies.append(repair)
    state.short_memory.set_context("last_repair", repair)
    return repair


def update_project_memory_from_reflection(state, file_path, tool_name, arguments):
    try:
        memory_result = read_project_memory()
        if not isinstance(memory_result, dict):
            raise TypeError("Project memory reader returned an invalid result.")
        if not memory_result.get("success"):
            raise RuntimeError(
                memory_result.get(
                    "error",
                    "Could not read project memory.",
                )
            )
        existing = memory_result.get("content", "")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        note = f"- {timestamp}: Modified '{file_path}' using {tool_name}."

        if "new_text" in arguments:
            change = str(arguments.get("new_text", "")).strip()
            if change:
                compact = re.sub(r"\s+", " ", change)
                if len(compact) > 160:
                    compact = compact[:160] + "..."
                note += f" Change: {compact}"

        recent = extract_memory_section(existing, "Recent Changes")
        notes = []
        for line in recent.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "# Project Memory" or (
                stripped.startswith("## ") and stripped != "## Recent Changes"
            ):
                break
            if stripped.startswith("- "):
                notes.append(stripped)

        notes.append(note)
        notes = notes[-20:]
        update_project_memory(
            "Recent Changes",
            "\n".join(notes),
        )
    except Exception as error:
        reporter.error(f"[Memory] Could not update project memory: {error}")


def build_intelligence_context(state, objective):
    memory = getattr(state, "short_memory", None)
    if memory is None:
        return "PHASE 2 INTELLIGENCE:\nNo short-term intelligence available."

    progress = calculate_goal_progress(state)
    last_error = memory.get_context("last_error")
    diagnosis = memory.get_context("last_diagnosis")
    repair = memory.get_context("last_repair")
    functional_verification_passed = getattr(
        state,
        "functional_verification_passed",
        False,
    )

    lines = [
        "PHASE 2 AGENT INTELLIGENCE",
        "=" * 40,
        "",
        f"GOAL PROGRESS: {progress}%",
        "",
        "PROGRESS STATE:",
        str(getattr(state, "goal_progress", {})),
        "",
        "FUNCTIONAL VERIFICATION:",
        str(functional_verification_passed),
        "",
        "RECENT ACTIONS:",
        memory.format_for_model(limit=8),
    ]

    if last_error:
        lines.extend(["", "LAST ERROR:", str(last_error)])
    if diagnosis:
        lines.extend(
            [
                "",
                "LAST DIAGNOSIS:",
                (
                    f"Category: {diagnosis.get('category')}\n"
                    f"Severity: {diagnosis.get('severity')}\n"
                    f"Suggestion: {diagnosis.get('suggestion')}"
                ),
            ]
        )
    if repair:
        lines.extend(["", "LAST REPAIR STRATEGY:", str(repair.get("strategy"))])

    lines.extend(
        [
            "",
            "INTELLIGENCE RULES:",
            "1. Do not repeat failed actions blindly.",
            "2. Use actual tool results as evidence.",
            "3. Inspect files before modifying them.",
            "4. Distinguish syntax success from functional success.",
            "5. Treat failed verification as an unresolved task.",
            "6. Compare expected and actual behavior.",
            "7. Make one targeted repair at a time.",
            "8. Verify after every meaningful repair.",
            "9. Do not claim completion without evidence.",
            (
                "10. If the objective and implementation conflict, "
                "the explicit verification requirement wins."
            ),
            "11. A syntax check does not establish functional correctness.",
            "12. A command with no output does not automatically prove success.",
            (
                "13. Determine completion from objective evidence, "
                "not merely from successful tool execution."
            ),
        ]
    )
    return "\n".join(lines)


def process_agent_intelligence(
    state,
    controller,
    objective,
    tool_name,
    arguments,
    result,
    error=None,
    command_output_text=None,
):
    initialize_agent_intelligence(state, objective)

    reflection = reflect_on_action(
        state=state,
        controller=controller,
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        error=error,
    )
    reporter.debug(f"\n[Reflection] {reflection}")

    output = (
        command_output_text
        if command_output_text is not None
        else (
            command_output(result)
            if tool_name == "run_command"
            else summarize_tool_result(tool_name, result)
        )
    )

    repair = None
    if tool_name == "run_command" and not command_succeeded(result):
        error_value = ""
        if isinstance(result, dict):
            error_value = result.get("error") or result.get("stderr") or output
        else:
            error_value = output
        repair = generate_repair_strategy(state, error_value, output)
    elif error:
        repair = generate_repair_strategy(state, error, output)

    contradictions = detect_contradictions(objective, output)
    if contradictions:
        reporter.debug("\n[Intelligence] Contradiction detected:")
        for contradiction in contradictions:
            reporter.debug(f"  - {contradiction}")
        if not hasattr(state, "contradictions"):
            state.contradictions = []
        state.contradictions.extend(contradictions)

    confidence = evaluate_action_confidence(state, tool_name, arguments)

    if repair:
        state.goal_progress["repair"] = True

    if (
        tool_name in WRITE_TOOLS
        and not error
        and isinstance(result, dict)
        and result.get("success", True)
    ):
        file_path = arguments.get("file_path")
        if file_path:
            update_project_memory_from_reflection(
                state,
                file_path,
                tool_name,
                arguments,
            )

    intelligence_context = build_intelligence_context(state, objective)

    return {
        "reflection": reflection,
        "diagnosis": (repair if repair else None),
        "contradictions": contradictions,
        "confidence": confidence,
        "progress": calculate_goal_progress(state),
        "context": intelligence_context,
    }


def inject_intelligence_context(messages, state, objective):
    context = build_intelligence_context(state, objective)
    messages.append(
        {
            "role": "user",
            "content": (
                "UPDATED AGENT INTELLIGENCE:\n\n" + context + "\n\n"
                "Use this information to decide the next "
                "action. Do not blindly repeat a previous "
                "failed action."
            ),
        }
    )


def intelligence_report(state):
    memory = getattr(state, "short_memory", None)
    progress = getattr(state, "goal_progress", {})
    reflections = getattr(state, "reflections", [])
    diagnoses = getattr(state, "diagnoses", [])
    contradictions = getattr(state, "contradictions", [])
    confidence_history = getattr(state, "action_confidence", [])

    report = [
        "",
        "================================",
        "PHASE 2 INTELLIGENCE REPORT",
        "================================",
        "",
        f"Goal progress: {calculate_goal_progress(state)}%",
        "",
        f"Progress state: {progress}",
        "",
        f"Reflections: {len(reflections)}",
        "",
        f"Diagnoses: {len(diagnoses)}",
        "",
        f"Contradictions: {len(contradictions)}",
        "",
        f"Confidence evaluations: {len(confidence_history)}",
        "",
        "Functional verification: "
        + str(
            getattr(
                state,
                "functional_verification_passed",
                False,
            )
        ),
    ]
    if memory:
        report.extend(["", "Recent memory:", memory.format_for_model(limit=5)])
    if diagnoses:
        report.extend(["", "Latest diagnosis:", str(diagnoses[-1])])
    if contradictions:
        report.extend(["", "Latest contradiction:", contradictions[-1]])
    return "\n".join(report)


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================


def ask_model(message, verbose=False):
    return run_agent(message, verbose=verbose)
