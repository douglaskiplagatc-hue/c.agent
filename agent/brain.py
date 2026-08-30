import time
import json
import traceback
from pathlib import Path
from fnmatch import fnmatch

from openai import OpenAI

from config import (
    LLM7_API_KEY,
    LLM7_MODEL,
    OLLAMA_LOCAL_URL,
    OLLAMA_LOCAL_KEY,
    OLLAMA_LOCAL_MODEL,
    OLLAMA_CLOUD_URL,
    OLLAMA_CLOUD_KEY,
    OLLAMA_CLOUD_MODEL,
    MISTRAL_API_KEY,  # optional – if you have it
    MISTRAL_MODEL,  # optional
)

from agent.tools import (
    list_files,
    read_file,
    search_code,
    create_file,
    edit_file,
    patch_file,
    run_command,
    git_status,
    git_checkpoint,
    git_rollback,
    analyze_python_file,
    search_project,
    read_project_memory,
    update_project_memory,
)

from agent.prompts import SYSTEM_PROMPT, PLANNING_PROMPT
from agent.state import TaskState
from agent.controller import AgentController, TaskPhase
from agent.reporting import Reporter

reporter = Reporter(verbose=False)
# ============================================================
# LLM CLIENTS
# ============================================================

llm7_client = OpenAI(
    api_key=LLM7_API_KEY,
    base_url="https://api.llm7.io/v1",
    timeout=60.0,
)

mistral_client = OpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",
    timeout=60.0,
)

ollama_client = OpenAI(
    api_key=OLLAMA_CLOUD_KEY,
    base_url=OLLAMA_CLOUD_URL,
    timeout=60.0,
)

client = mistral_client


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MISTRAL_MODEL = "mistral-small-latest"
LLM7_MODEL = "gpt-5.4-mini"
OLLAMA_MODEL = OLLAMA_CLOUD_MODEL


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [
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


INSPECTION_TOOLS = {
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
# LLM REQUEST WITH MULTI-PROVIDER FALLBACK
# ============================================================


def llm_request_with_retry(messages, tools=None, max_retries=2):
    """
    Send an LLM request using multiple providers.

    Provider order (try first → fallback):
        1. Mistral    (if configured)
        2. LLM7
        3. Ollama Cloud
        4. Ollama Local

    Non‑transient errors (auth, invalid model, quota) cause an immediate
    switch to the next provider without retrying.
    """

    # Create clients inside the function (so they are always defined)
    clients = {}

    try:
        clients["mistral"] = OpenAI(
            api_key=MISTRAL_API_KEY,
            base_url="https://api.mistral.ai/v1",
            timeout=60.0,
        )
        mistral_model = MISTRAL_MODEL
        use_mistral = True
    except (NameError, AttributeError):
        use_mistral = False

    try:
        clients["llm7"] = OpenAI(
            api_key=LLM7_API_KEY,
            base_url="https://api.llm7.io/v1",
            timeout=60.0,
        )
        llm7_model = LLM7_MODEL
        use_llm7 = True
    except (NameError, AttributeError):
        use_llm7 = False

    try:
        clients["ollama_cloud"] = OpenAI(
            api_key=OLLAMA_CLOUD_KEY,
            base_url=OLLAMA_CLOUD_URL,
            timeout=60.0,
        )
        ollama_cloud_model = OLLAMA_CLOUD_MODEL
        use_ollama_cloud = True
    except (NameError, AttributeError):
        use_ollama_cloud = False

    try:
        clients["ollama_local"] = OpenAI(
            api_key=OLLAMA_LOCAL_KEY,
            base_url=OLLAMA_LOCAL_URL,
            timeout=60.0,
        )
        ollama_local_model = OLLAMA_LOCAL_MODEL
        use_ollama_local = True
    except (NameError, AttributeError):
        use_ollama_local = False

    # Build the provider list in order of preference
    providers = []
    if use_mistral:
        providers.append(
            {"client": clients["mistral"], "model": mistral_model, "name": "Mistral"}
        )
    if use_llm7:
        providers.append(
            {"client": clients["llm7"], "model": llm7_model, "name": "LLM7"}
        )
    if use_ollama_cloud:
        providers.append(
            {
                "client": clients["ollama_cloud"],
                "model": ollama_cloud_model,
                "name": "Ollama Cloud",
            }
        )
    if use_ollama_local:
        providers.append(
            {
                "client": clients["ollama_local"],
                "model": ollama_local_model,
                "name": "Ollama Local",
            }
        )

    if not providers:
        raise RuntimeError("No LLM providers are configured. Check your config.py.")

    last_exception = None

    for provider in providers:
        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": provider["model"],
                    "messages": messages,
                    "timeout": 60.0,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = provider["client"].chat.completions.create(**kwargs)
                reporter.debug(
                    f"[LLM] Success: {provider['name']} ({provider['model']})"
                )
                return response

            except Exception as error:
                last_exception = error
                error_msg = str(error).lower()
                reporter.debug(
                    f"[LLM] {provider['name']} attempt {attempt + 1} failed: {error}"
                )

                # Non‑transient errors – switch provider immediately
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
                    )
                )
                if non_transient:
                    reporter.debug(
                        f"[LLM] {provider['name']} returned a non‑transient error. Switching provider."
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
                    reporter.debug(
                        f"[LLM] {provider['name']} returned an unrecognized error. Switching provider."
                    )
                    break

                if attempt < max_retries:
                    delay = 2**attempt
                    reporter.debug(f"[LLM] Retrying {provider['name']} in {delay}s...")
                    time.sleep(delay)
                else:
                    reporter.debug(
                        f"[LLM] {provider['name']} retries exhausted. Switching provider."
                    )

        # If we exit the loop, the provider failed entirely – move on.

    # All providers failed
    raise last_exception or RuntimeError("All LLM providers failed.")


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


def objective_requires_modification(
    objective,
):

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
# STATE TRACKING
# ============================================================


def update_state(
    state,
    tool_name,
    arguments,
    result,
):

    if tool_name == "read_file":
        state.record_inspected_file(arguments["file_path"])

    elif tool_name in WRITE_TOOLS:
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
        "exception:",
        "assertionerror:",
        "modulenotfounderror:",
        "importerror:",
        "nameerror:",
        "typeerror:",
        "attributeerror:",
        "file not found",
        "no such file or directory",
        "failed",
        "error:",
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
        return True

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
                        "ran. Answer TRUE or FALSE only."
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
# ROLLBACK SNAPSHOT
# ============================================================


def snapshot_file(
    file_path,
    protected_snapshots,
):

    if file_path in protected_snapshots:
        return

    try:
        protected_snapshots[file_path] = read_file(file_path)

    except Exception:
        protected_snapshots[file_path] = None


def restore_snapshots(
    protected_snapshots,
):

    for file_path, content in protected_snapshots.items():
        try:
            if content is None:
                path = Path(file_path)

                if path.exists():
                    path.unlink()

                continue

            create_file(
                file_path=file_path,
                content=content,
            )

        except Exception as error:
            reporter.error(f"[Rollback] Failed restoring {file_path}: {error}")


# ============================================================
# AGENT
# ============================================================


def run_agent(objective,verbose = False):

    reporter = Reporter(verbose=verbose)

    objective = (objective or "").strip()
    if not objective:
        return "No objective was provided."

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
    controller = AgentController(state, reporter)

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
    try:
        state.plan = create_plan(objective, project_memory)
    except Exception as error:
        state.record_error(str(error))
        controller.fail()
        return format_task_result(
            f"Planning failed.\n\nError: {error}", state, "FAILED"
        )

    reporter.debug("\n[Agent Plan]")
    reporter.debug()

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
10. For modification tasks, the required file must actually be modified.
11. For modification tasks, the final implementation must be verified.
12. Python syntax verification alone does not prove that the requested behavior works.
13. When the objective requires running a Python file, actually run it.
14. When a verification command fails, inspect the actual error before attempting repair.
15. Make only the changes necessary for the objective.
16. Never modify forbidden files.
17. Read-only tasks must never modify files.
18. If verification fails, diagnose the actual problem, make a targeted repair, and verify again.
19. Do not claim completion merely because a tool call returned successfully.
20. Once the objective has genuinely been verified, stop immediately.
21. BEFORE reading or editing any file, you MUST call list_files on the workspace directory.
22. NEVER guess a file path without confirming it exists.
23. When the assistant returns multiple tool calls, ALL tool results must be returned to the model BEFORE adding any new user/repair message.
24. Never place a user message between two tool results belonging to the same assistant turn.

BEGIN BY INSPECTING THE RELEVANT FILE(S).
""",
        },
    ]

    # ------------------------------------------------------------
    # HISTORY / PROTECTION
    # ------------------------------------------------------------
    recent_tool_call_keys = []  # ✅ FIXED: now a list
    protected_snapshots = {}
    failed_actions = set()
    pending_repair_instructions = []

    # ============================================================
    # MAIN LOOP
    # ============================================================
    while not state.completed:
        # --------------------------------------------------------
        # ATTEMPT LIMIT
        # --------------------------------------------------------
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
                f"The agent stopped after reaching the maximum of {state.max_attempts} repair attempts.",
                state,
                "FAILED",
            )

        # --------------------------------------------------------
        # MODEL REQUEST
        # --------------------------------------------------------
        try:
            response = llm_request_with_retry(messages=messages, tools=TOOLS)
        except Exception as error:
            state.record_error(str(error))
            try:
                controller.fail()
            except Exception:
                pass
            return format_task_result(
                f"The model request failed.\n\nError: {error}", state, "FAILED"
            )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        # --------------------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # MODEL FINISHED WITHOUT TOOL CALL
        # --------------------------------------------------------
        if not tool_calls:
            content = assistant_message.content or ""

            if read_only:
                try:
                    controller.complete()
                except Exception:
                    pass
                state.completed = True
                result_text = format_task_result(content, state, "COMPLETED")
                reporter.user(result_text)
                return result_text

            if not state.modified_files:
                if state.functional_verification_passed:
                    try:
                        controller.complete()
                    except Exception:
                        pass
                    state.completed = True
                    return format_task_result(
                        "Task completed (existing implementation already works).",
                        state,
                        "COMPLETED",
                    )
                reporter.debug(
                    "\n[Agent] Completion blocked: no modification has occurred."
                    "Checking whether the existing implementation already satisfies the objective."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": """
The required file may already exist and may already satisfy the objective.
Do NOT modify the file unnecessarily.
1. Inspect the existing implementation.
2. Determine whether it already satisfies the user's objective.
3. If it already satisfies the objective, verify it by running the appropriate test.
4. If verification succeeds, finish the task.
5. If it does not satisfy the objective, modify it and verify again.
Do not claim success without verification.
""",
                    }
                )
                continue

            successful_verifications = [
                v for v in state.verifications if v.get("success") is True
            ]
            if not successful_verifications:
                reporter.debug(
                    "\n[Agent] Completion blocked: no successful verification exists."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": """
STOP.

The implementation was modified, but there is no successful verification recorded.

You are NOT finished.

Verify the actual requested behavior now.

For a Python task:

1. Run the relevant Python file or test.
2. Inspect the complete output.
3. If it fails, diagnose the actual error.
4. Repair the implementation.
5. Run verification again.
6. Only finish after successful verification.

Do NOT claim completion.
""",
                    }
                )
                continue

            # REAL COMPLETION
            try:
                controller.complete()
            except Exception:
                pass
            state.completed = True
            result_text = format_task_result(content, state, "COMPLETED")
            reporter.user(result_text)
            return result_text

        # ============================================================
        # TOOL CALL PROCESSING
        # ============================================================
        for tool_call in tool_calls:
            error_text = None
            name = tool_call.function.name

            # ----------- Parse arguments -----------
            try:
                arguments = json.loads(tool_call.function.arguments)
            except Exception as error:
                error_text = "Invalid tool arguments: " + str(error)
                reporter.debug(f"\n[Agent] {error_text}")
                state.record_error(error_text)
                result_text = json.dumps({"success": False, "error": error_text})
                append_tool_result(messages, tool_call, result_text)
                pending_repair_instructions.append(
                    f"TOOL ARGUMENT ERROR.\nTool: {name}\nActual error: {error_text}\nDo not repeat the malformed call."
                )
                continue

            reporter.debug(f"\n[Agent → {name}]")
            reporter.debug(f"Arguments: {arguments}")

            # ----------- Duplicate call -----------
            call_key = make_tool_call_key(name, arguments)
            if call_key in recent_tool_call_keys:
                error = "This exact tool call was already executed. Do not repeat it."
                reporter.debug("\n[Agent] Duplicate tool call blocked.")
                state.record_failed_action(name, arguments, error)
                result_text = json.dumps({"success": False, "error": error})
                append_tool_result(messages, tool_call, result_text)
                pending_repair_instructions.append(
                    f"DUPLICATE TOOL CALL.\nTool: {name}\nArguments: {arguments}\nDo not repeat it."
                )
                continue

            recent_tool_call_keys.append(call_key)
            if len(recent_tool_call_keys) > 30:
                recent_tool_call_keys.pop(0)

            # ----------- Forbidden file -----------
            if name in WRITE_TOOLS:
                file_path = arguments.get("file_path")
                if is_forbidden_file(file_path, forbidden_patterns):
                    error = f"Modification of '{file_path}' is prohibited by project memory."
                    reporter.debug("\n[Agent] Forbidden modification blocked.")
                    state.record_error(error)
                    state.record_failed_action(name, arguments, error)
                    result_text = json.dumps({"success": False, "error": error})
                    append_tool_result(messages, tool_call, result_text)
                    pending_repair_instructions.append(
                        f"FORBIDDEN FILE.\nFile: {file_path}\nDo not modify it."
                    )
                    continue

            # ----------- Failed edit protection -----------
            if name in {"edit_file", "patch_file"}:
                patch_key = (arguments.get("file_path"), arguments.get("old_text"))
                if patch_key in failed_actions:
                    error = "This exact edit already failed. Do not repeat it."
                    reporter.debug("\n[Agent] Repeated failed edit blocked.")
                    state.record_failed_action(name, arguments, error)
                    result_text = json.dumps({"success": False, "error": error})
                    append_tool_result(messages, tool_call, result_text)
                    pending_repair_instructions.append(
                        f"REPEATED FAILED EDIT.\nTool: {name}\nFile: {arguments.get('file_path')}\nRead the current file and determine the correct replacement."
                    )
                    continue

            # ----------- Snapshot before write -----------
            if name in WRITE_TOOLS:
                file_path = arguments.get("file_path")
                snapshot_file(file_path, protected_snapshots)

            # ----------- Execute tool -----------
            result = None
            tool_failed = False
            try:
                result = execute_tool(name, arguments)
                update_state(state, name, arguments, result)
                result_text = json.dumps(result, indent=2, default=str)
                reporter.debug("\n[Tool successful]")
            except FileExistsError as error:
                tool_failed = True
                error_text = str(error)
                reporter.debug("\n[Tool failure]")
                reporter.error(error_text)
                traceback.print_exc()
                state.record_error(error_text)
                state.record_failed_action(name, arguments, error_text)
                result = {
                    "success": False,
                    "error": error_text,
                    "hint": "The file already exists. Use read_file to inspect it, then use edit_file or patch_file.",
                }
                result_text = json.dumps(result, indent=2)
                pending_repair_instructions.append(
                    f"TOOL FAILURE.\nTool: {name}\nArguments: {arguments}\nActual error: {error_text}\nDo not repeat the failed call. Inspect and make one targeted correction."
                )
            except Exception as error:
                tool_failed = True
                error_text = str(error)
                reporter.debug("\n[Tool failure]")
                reporter.error(error_text)
                traceback.print_exc()
                state.record_error(error_text)
                state.record_failed_action(name, arguments, error_text)
                result = {
                    "success": False,
                    "error": error_text,
                    "instruction": "The tool failed. Do not blindly repeat it.",
                }
                result_text = json.dumps(result, indent=2)
                if name in {"edit_file", "patch_file"}:
                    failed_actions.add(
                        (arguments.get("file_path"), arguments.get("old_text"))
                    )
                pending_repair_instructions.append(
                    f"TOOL FAILURE.\nTool: {name}\nArguments: {arguments}\nActual error: {error_text}\nDiagnose, inspect, make ONE targeted correction, and continue."
                )

            # ----------- Append tool result -----------
            append_tool_result(messages, tool_call, result_text)

            # ----------- Write result handling -----------
            if (
                name in WRITE_TOOLS
                and isinstance(result, dict)
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
                state.record_failed_action(name, arguments, str(error))
                if name in {"edit_file", "patch_file"}:
                    failed_actions.add((file_path, arguments.get("old_text")))
                pending_repair_instructions.append(
                    f"FILE MODIFICATION FAILED.\nFile: {file_path}\nError: {error}\nRead the current file and make one targeted correction."
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
                        state.record_failed_action(name, arguments, message)
                        failed_actions.add((file_path, arguments.get("old_text")))
                        pending_repair_instructions.append(
                            f"PYTHON SYNTAX VERIFICATION FAILED.\nFile: {file_path}\nError: {message}\nDo NOT declare completion. Fix syntax and verify again."
                        )

            # ----------- Command result -----------
            if name == "run_command":
                command = arguments.get("command", "")
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
                    state.record_failed_action(name, arguments, str(error))
                    pending_repair_instructions.append(
                        f"COMMAND FAILED.\nCommand: {command}\nOutput: {output}\nDiagnose the failure and make one targeted repair."
                    )
                    state.attempts += 1  # ✅ increment attempts on command failure

                elif is_verification:
                    reporter.debug("\n[Verification command detected]")
                    output_lower = output.lower()
                    if (
                        "all tests passed" in output_lower
                        or "all tests pass" in output_lower
                    ):
                        verified = True
                        reporter.debug(
                            "[Verification] Deterministic success: All tests passed."
                        )
                    elif all(
                        "pass" in line.lower()
                        for line in output.splitlines()
                        if line.strip()
                    ):
                        verified = True
                        reporter.debug(
                            "[Verification] Deterministic success: All test lines contain PASS."
                        )
                    else:
                        if succeeded and not verification_output_is_bad(output):
                            explicit_result = objective_output_check(objective, output)
                            if explicit_result is True:
                                verified = True
                            elif (
                                command.strip().startswith("python ")
                                and output.strip() == ""
                            ):
                                verified = True
                            else:
                                verified = efficient_verify(
                                    objective, state.modified_files, output, state
                                )
                        else:
                            verified = False

                    if verified:
                        state.functional_verification_passed = True  # <-- add this
                        state.record_verification(command, True)
                        try:
                            controller.begin_verification()
                        except Exception:
                            pass
                        state.record_verification(command, True)
                        reporter.debug("\n[Verification successful]")
                        try:
                            controller.complete()
                        except Exception:
                            pass
                        state.completed = True
                    else:
                        reporter.error("\n[Verification failed]")
                        error = "Verification output did not provide sufficient evidence that the objective was satisfied."
                        state.record_error(error)
                        state.record_failed_action(name, arguments, error)
                        pending_repair_instructions.append(
                            f"VERIFICATION FAILED.\nObjective: {objective}\nCommand: {command}\nOutput: {output}\nDo NOT claim completion. Diagnose and repair."
                        )
                        state.attempts += (
                            1  # ✅ increment attempts on verification failure
                        )

            # ------------------------------------------------------------
            # PHASE 2 INTELLIGENCE – called after EVERY tool execution
            # ------------------------------------------------------------
            process_agent_intelligence(
                state=state,
                controller=controller,
                objective=objective,
                tool_name=name,
                arguments=arguments,
                result=result,
                error=error_text if "error_text" in locals() else None,
                command_output_text=output if name == "run_command" else None,
            )

        # ============================================================
        # AFTER ALL TOOL RESULTS ARE APPENDED – send repair instructions
        # and inject intelligence context
        # ============================================================

        # Inject intelligence context (Phase 2) as a user message
        inject_intelligence_context(messages, state, objective)

        if pending_repair_instructions:
            combined = "\n\n".join(pending_repair_instructions)
            append_repair_instruction(messages, combined)
            pending_repair_instructions.clear()  # ✅ clear after sending

        # ------------------------------------------------------------
        # HARD STOP
        # ------------------------------------------------------------
        if state.completed:
            break

    # ============================================================
    # FINAL RESULT
    # ============================================================
    successful_verifications = [
        v
        for v in state.verifications
        if v.get("success") is True and command_is_verification(v.get("command", ""))
    ]
    if requires_modification and not state.functional_verification_passed:
        result_text = format_task_result(
            "Task could not be safely completed.", state, "FAILED"
        )
        reporter.user(result_text)
        return result_text

    result_text = format_task_result("Task completed.", state, "COMPLETED")
    reporter.user(result_text)
    return result_text


# ============================================================
# PHASE 2: AGENT INTELLIGENCE
# ============================================================
#
# This module gives the coding agent:
#
#   1. Short-term memory
#   2. Action reflection
#   3. Error diagnosis
#   4. Failure pattern detection
#   5. Goal/progress tracking
#   6. Repair strategy generation
#   7. Contradiction detection
#   8. Tool-result summarisation
#   9. Recent-action awareness
#  10. Verification awareness
#  11. Repeated-action detection
#  12. Intelligent context injection
#
# It is intentionally self-contained so that Phase 2 can be
# integrated into the existing run_agent() loop without creating
# many separate files.
# ============================================================

import re
from datetime import datetime


# ============================================================
# SHORT-TERM MEMORY
# ============================================================


class ShortTermMemory:
    """
    Memory for the current task execution.

    Unlike project memory, this memory is temporary and exists
    only while the current agent task is running.

    It remembers:
        - recent tools
        - arguments
        - results
        - reflections
        - errors
        - diagnoses
        - repairs
        - important context
    """

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

    # --------------------------------------------------------
    # Add memory entry
    # --------------------------------------------------------

    def add(
        self,
        tool_name,
        arguments,
        reflection,
        result=None,
        error=None,
    ):

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

    # --------------------------------------------------------
    # Safe result representation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Recent memory
    # --------------------------------------------------------

    def get_recent(self, n=5):

        if not self.history:
            return []

        return self.history[-n:]

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    def set_context(self, key, value):

        self.context[key] = value

    def get_context(self, key, default=None):

        return self.context.get(key, default)

    # --------------------------------------------------------
    # Recent tool detection
    # --------------------------------------------------------

    def last_tool(self):

        if not self.history:
            return None

        return self.history[-1]["tool"]

    # --------------------------------------------------------
    # Last error
    # --------------------------------------------------------

    def last_error(self):

        return self.context.get("last_error")

    # --------------------------------------------------------
    # Count tool usage
    # --------------------------------------------------------

    def count_tool(self, tool_name):

        return sum(1 for entry in self.history if entry["tool"] == tool_name)

    # --------------------------------------------------------
    # Detect repeated action
    # --------------------------------------------------------

    def repeated_action(
        self,
        tool_name,
        arguments,
        window=5,
    ):

        recent = self.history[-window:]

        for entry in recent:
            if entry["tool"] != tool_name:
                continue

            if entry["arguments"] == arguments:
                return True

        return False

    # --------------------------------------------------------
    # Format memory for LLM
    # --------------------------------------------------------

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


# ============================================================
# INITIALISE INTELLIGENCE
# ============================================================


def initialize_agent_intelligence(state, objective):

    if not hasattr(
        state,
        "short_memory",
    ):
        state.short_memory = ShortTermMemory(max_entries=20)

    if not hasattr(
        state,
        "reflections",
    ):
        state.reflections = []

    if not hasattr(
        state,
        "diagnoses",
    ):
        state.diagnoses = []

    if not hasattr(
        state,
        "repair_strategies",
    ):
        state.repair_strategies = []

    if not hasattr(
        state,
        "action_confidence",
    ):
        state.action_confidence = []

    if not hasattr(
        state,
        "goal_progress",
    ):
        state.goal_progress = {
            "inspection": False,
            "modification": False,
            "verification": False,
            "repair": False,
        }

    state.short_memory.set_context(
        "current_goal",
        objective,
    )


# ============================================================
# RESULT SUMMARISATION
# ============================================================


def summarize_tool_result(
    tool_name,
    result,
):
    """
    Convert potentially huge tool output into a compact
    representation suitable for the LLM context.
    """

    if result is None:
        return "No result returned."

    if isinstance(result, dict):
        success = result.get(
            "success",
            True,
        )

        summary_parts = [
            f"Tool: {tool_name}",
            f"Success: {success}",
        ]

        if "error" in result:
            summary_parts.append(f"Error: {str(result['error'])[:500]}")

        if "stdout" in result:
            stdout = str(
                result.get(
                    "stdout",
                    "",
                )
            )

            if len(stdout) > 1000:
                stdout = stdout[:1000] + "\n...[output truncated]"

            if stdout:
                summary_parts.append(f"STDOUT:\n{stdout}")

        if "stderr" in result:
            stderr = str(
                result.get(
                    "stderr",
                    "",
                )
            )

            if len(stderr) > 1000:
                stderr = stderr[:1000] + "\n...[stderr truncated]"

            if stderr:
                summary_parts.append(f"STDERR:\n{stderr}")

        return "\n".join(summary_parts)

    text = str(result)

    if len(text) > 1500:
        text = text[:1500] + "\n...[result truncated]"

    return f"Tool: {tool_name}\nResult:\n{text}"


# ============================================================
# ERROR DIAGNOSIS
# ============================================================


def diagnose_error(
    error_text,
    command_output=None,
):

    error_text = str(error_text or "")

    output = str(command_output or "")

    combined = (error_text + "\n" + output).lower()

    # --------------------------------------------------------
    # Syntax
    # --------------------------------------------------------

    if "syntaxerror" in combined or "indentationerror" in combined:
        return {
            "category": "syntax",
            "severity": "high",
            "suggestion": (
                "Inspect the exact file and repair the "
                "reported syntax or indentation problem."
            ),
        }

    # --------------------------------------------------------
    # Import / dependency
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Assertions / tests
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Type
    # --------------------------------------------------------

    if "typeerror" in combined:
        return {
            "category": "type_error",
            "severity": "high",
            "suggestion": (
                "Inspect the actual values and function "
                "arguments involved in the TypeError."
            ),
        }

    # --------------------------------------------------------
    # Name
    # --------------------------------------------------------

    if "nameerror" in combined:
        return {
            "category": "name_error",
            "severity": "high",
            "suggestion": (
                "Check variable and function names and "
                "verify that the referenced symbol exists."
            ),
        }

    # --------------------------------------------------------
    # Attribute
    # --------------------------------------------------------

    if "attributeerror" in combined:
        return {
            "category": "attribute_error",
            "severity": "high",
            "suggestion": (
                "Inspect the actual object type and verify "
                "that the requested attribute or method exists."
            ),
        }

    # --------------------------------------------------------
    # File
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Permission
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    if "timeout" in combined or "timed out" in combined:
        return {
            "category": "timeout",
            "severity": "medium",
            "suggestion": (
                "Determine whether the operation is genuinely "
                "slow or whether an external service is unavailable."
            ),
        }

    # --------------------------------------------------------
    # API / quota
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Tool protocol
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Connection
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Generic error
    # --------------------------------------------------------

    return {
        "category": "unknown",
        "severity": "unknown",
        "suggestion": (
            "Inspect the complete error and relevant files before deciding on a repair."
        ),
    }


# ============================================================
# REFLECTION ENGINE
# ============================================================


def reflect_on_action(
    state,
    controller,
    tool_name,
    arguments,
    result,
    error=None,
):

    if not hasattr(
        state,
        "short_memory",
    ):
        state.short_memory = ShortTermMemory()

    # --------------------------------------------------------
    # Error reflection
    # --------------------------------------------------------

    if error:
        diagnosis = diagnose_error(
            str(error),
        )

        reflection = (
            f"Action '{tool_name}' failed. "
            f"Category: {diagnosis['category']}. "
            f"Severity: {diagnosis['severity']}. "
            f"Next strategy: {diagnosis['suggestion']}"
        )

        state.short_memory.set_context(
            "last_error",
            str(error),
        )

        state.short_memory.set_context(
            "last_diagnosis",
            diagnosis,
        )

        if not hasattr(
            state,
            "diagnoses",
        ):
            state.diagnoses = []

        state.diagnoses.append(diagnosis)

    else:
        success = (
            result.get(
                "success",
                True,
            )
            if isinstance(
                result,
                dict,
            )
            else True
        )

        if success:
            summary = summarize_tool_result(
                tool_name,
                result,
            )

            reflection = f"Action '{tool_name}' succeeded. {summary[:700]}"

        else:
            error_value = (
                result.get(
                    "error",
                    "unknown error",
                )
                if isinstance(
                    result,
                    dict,
                )
                else "unknown error"
            )

            diagnosis = diagnose_error(str(error_value))

            reflection = (
                f"Action '{tool_name}' returned a failure. "
                f"Category: {diagnosis['category']}. "
                f"Next strategy: {diagnosis['suggestion']}"
            )

            state.short_memory.set_context(
                "last_error",
                str(error_value),
            )

    # --------------------------------------------------------
    # Save reflection
    # --------------------------------------------------------

    state.short_memory.add(
        tool_name=tool_name,
        arguments=arguments,
        reflection=reflection,
        result=result,
        error=error,
    )

    if not hasattr(
        state,
        "reflections",
    ):
        state.reflections = []

    state.reflections.append(reflection)

    # --------------------------------------------------------
    # Progress tracking
    # --------------------------------------------------------

    if tool_name in INSPECTION_TOOLS:
        state.goal_progress["inspection"] = True

    if tool_name in WRITE_TOOLS:
        state.goal_progress["modification"] = True

    if tool_name == "run_command":
        command = arguments.get(
            "command",
            "",
        )

        if command_is_verification(command):
            state.goal_progress["verification"] = True

    return reflection


# ============================================================
# CONTRADICTION DETECTION
# ============================================================


def detect_contradictions(
    objective,
    result,
):
    """
    Detect obvious contradictions between the user's requested
    behavior and observed output.

    This is intentionally conservative.

    It does not attempt to understand every possible objective.
    It catches common situations where the agent implemented one
    thing but verification expects another.
    """

    objective_text = (objective or "").lower()

    output_text = (result or "").lower()

    contradictions = []

    # --------------------------------------------------------
    # Expected output patterns
    # --------------------------------------------------------

    quoted_values = re.findall(
        r'"([^"]+)"',
        objective or "",
    )

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
                (
                    f"Expected explicit value "
                    f"'{expected}' was not found "
                    f"in the observed output."
                )
            )

    # --------------------------------------------------------
    # Test failure
    # --------------------------------------------------------

    if (
        "test passed: false" in output_text
        or "test failed" in output_text
        or "failed" in output_text
    ):
        contradictions.append("The verification output indicates failure.")

    # --------------------------------------------------------
    # Expected vs actual
    # --------------------------------------------------------

    expected_match = re.search(
        r"expected\s*:\s*(.+)",
        output_text,
        re.IGNORECASE,
    )

    result_match = re.search(
        r"result\s*:\s*(.+)",
        output_text,
        re.IGNORECASE,
    )

    if expected_match and result_match:
        expected_value = expected_match.group(1).strip()

        actual_value = result_match.group(1).strip()

        if expected_value != actual_value:
            contradictions.append(
                ("Observed result does not match the expected result.")
            )

    return contradictions


# ============================================================
# GOAL PROGRESS
# ============================================================


def calculate_goal_progress(
    state,
):

    progress = state.goal_progress

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


# ============================================================
# ACTION CONFIDENCE
# ============================================================


def evaluate_action_confidence(
    state,
    tool_name,
    arguments,
):

    reasons = []

    confidence = 1.0

    memory = state.short_memory

    # --------------------------------------------------------
    # Repeated action
    # --------------------------------------------------------

    if memory.repeated_action(
        tool_name,
        arguments,
    ):
        confidence -= 0.6

        reasons.append("The same action was recently attempted.")

    # --------------------------------------------------------
    # Previous error
    # --------------------------------------------------------

    if memory.get_context("last_error"):
        confidence -= 0.1

        reasons.append("A previous action failed and requires awareness.")

    # --------------------------------------------------------
    # Write before inspection
    # --------------------------------------------------------

    if tool_name in WRITE_TOOLS and not state.goal_progress.get(
        "inspection",
        False,
    ):
        confidence -= 0.5

        reasons.append("Modification attempted before inspection.")

    # --------------------------------------------------------
    # Verification awareness
    # --------------------------------------------------------

    if tool_name == "run_command":
        command = arguments.get(
            "command",
            "",
        )

        if command_is_verification(command):
            reasons.append("Command appears to be a verification command.")

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    result = {
        "confidence": round(
            confidence,
            2,
        ),
        "reasons": reasons,
    }

    if not hasattr(
        state,
        "action_confidence",
    ):
        state.action_confidence = []

    state.action_confidence.append(
        {
            "tool": tool_name,
            "confidence": result["confidence"],
            "reasons": reasons,
        }
    )

    return result


# ============================================================
# REPAIR STRATEGY
# ============================================================


def generate_repair_strategy(
    state,
    error_text,
    command_output=None,
):

    diagnosis = diagnose_error(
        error_text,
        command_output,
    )

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

    strategy = strategies.get(
        category,
        suggestion,
    )

    repair = {
        "category": category,
        "severity": diagnosis["severity"],
        "strategy": strategy,
    }

    if not hasattr(
        state,
        "repair_strategies",
    ):
        state.repair_strategies = []

    state.repair_strategies.append(repair)

    state.short_memory.set_context(
        "last_repair",
        repair,
    )

    return repair


# ============================================================
# PROJECT MEMORY UPDATE
# ============================================================


def update_project_memory_from_reflection(
    state,
    file_path,
    tool_name,
    arguments,
):

    try:
        memory_result = read_project_memory()
        if not isinstance(memory_result, dict):
            raise TypeError("Project memory reader returned an invalid result.")
        if not memory_result.get("success"):
            raise RuntimeError(memory_result.get("error", "Could not read project memory.")
    )
        existing = memory_result.get("content", "")


        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        note = f"\n- {timestamp}: Modified '{file_path}' using {tool_name}."

        if "new_text" in arguments:
            change = str(
                arguments.get(
                    "new_text",
                    "",
                )
            ).strip()

            if change:
                # Do not dump an entire file into memory.
                # Store only a compact description.

                compact = re.sub(
                    r"\s+",
                    " ",
                    change,
                )

                if len(compact) > 160:
                    compact = compact[:160] + "..."

                note += f" Change: {compact}"

        update_project_memory(
            "Recent Changes",
            existing + note,
        )

    except Exception as error:
        reporter.error(f"[Memory] Could not update project memory: {error}")


# ============================================================
# INTELLIGENCE CONTEXT FOR LLM
# ============================================================


def build_intelligence_context(
    state,
    objective,
):

    memory = getattr(
        state,
        "short_memory",
        None,
    )

    if memory is None:
        return "PHASE 2 INTELLIGENCE:\nNo short-term intelligence available."

    progress = calculate_goal_progress(state)

    last_error = memory.get_context("last_error")

    diagnosis = memory.get_context("last_diagnosis")

    repair = memory.get_context("last_repair")

    lines = [
        "PHASE 2 AGENT INTELLIGENCE",
        "=" * 40,
        "",
        f"GOAL PROGRESS: {progress}%",
        "",
        "PROGRESS STATE:",
        str(state.goal_progress),
        "",
        "RECENT ACTIONS:",
        memory.format_for_model(limit=8),
    ]

    if last_error:
        lines.extend(
            [
                "",
                "LAST ERROR:",
                str(last_error),
            ]
        )

    if diagnosis:
        lines.extend(
            [
                "",
                "LAST DIAGNOSIS:",
                (
                    f"Category: "
                    f"{diagnosis.get('category')}\n"
                    f"Severity: "
                    f"{diagnosis.get('severity')}\n"
                    f"Suggestion: "
                    f"{diagnosis.get('suggestion')}"
                ),
            ]
        )

    if repair:
        lines.extend(
            [
                "",
                "LAST REPAIR STRATEGY:",
                str(repair.get("strategy")),
            ]
        )

    # --------------------------------------------------------
    # Important agent reminder
    # --------------------------------------------------------

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
            "10. If the objective and implementation conflict, "
            "the explicit verification requirement wins.",
        ]
    )

    return "\n".join(lines)


# ============================================================
# MAIN PHASE 2 PROCESSOR
# ============================================================


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
    """
    Central Phase 2 intelligence processor.

    Call this after every tool execution.

    It performs:

        tool reflection
        error diagnosis
        contradiction detection
        progress tracking
        action confidence
        repair strategy generation
        memory updates
        intelligence context generation

    Returns a dictionary containing all intelligence produced.
    """

    # --------------------------------------------------------
    # Ensure intelligence exists
    # --------------------------------------------------------

    initialize_agent_intelligence(
        state,
        objective,
    )

    # --------------------------------------------------------
    # Reflection
    # --------------------------------------------------------

    reflection = reflect_on_action(
        state=state,
        controller=controller,
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        error=error,
    )

    reporter.debug(f"\n[Reflection] {reflection}")
    # --------------------------------------------------------
    # Command output
    # --------------------------------------------------------

    output = (
        command_output_text
        if command_output_text is not None
        else command_output(result)
        if tool_name == "run_command"
        else summarize_tool_result(
            tool_name,
            result,
        )
    )

    # --------------------------------------------------------
    # Diagnose failed command
    # --------------------------------------------------------

    repair = None

    if tool_name == "run_command" and not command_succeeded(result):
        error_value = ""

        if isinstance(
            result,
            dict,
        ):
            error_value = result.get("error") or result.get("stderr") or output

        else:
            error_value = output

        repair = generate_repair_strategy(
            state,
            error_value,
            output,
        )

    elif error:
        repair = generate_repair_strategy(
            state,
            error,
            output,
        )

    # --------------------------------------------------------
    # Detect contradictions
    # --------------------------------------------------------

    contradictions = detect_contradictions(
        objective,
        output,
    )

    if contradictions:
        reporter.debug("\n[Intelligence] Contradiction detected:")

        for contradiction in contradictions:
            reporter.debug(f"  - {contradiction}")

        if not hasattr(
            state,
            "contradictions",
        ):
            state.contradictions = []

        state.contradictions.extend(contradictions)

    # --------------------------------------------------------
    # Action confidence
    # --------------------------------------------------------

    confidence = evaluate_action_confidence(
        state,
        tool_name,
        arguments,
    )

    # --------------------------------------------------------
    # Repair progress
    # --------------------------------------------------------

    if repair:
        state.goal_progress["repair"] = True

    # --------------------------------------------------------
    # Project memory
    # --------------------------------------------------------

    if (
        tool_name in WRITE_TOOLS
        and not error
        and isinstance(
            result,
            dict,
        )
        and result.get(
            "success",
            True,
        )
    ):
        file_path = arguments.get("file_path")

        if file_path:
            update_project_memory_from_reflection(
                state,
                file_path,
                tool_name,
                arguments,
            )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    intelligence_context = build_intelligence_context(
        state,
        objective,
    )

    # --------------------------------------------------------
    # Return complete intelligence result
    # --------------------------------------------------------

    return {
        "reflection": reflection,
        "diagnosis": (repair if repair else None),
        "contradictions": contradictions,
        "confidence": confidence,
        "progress": calculate_goal_progress(state),
        "context": intelligence_context,
    }


# ============================================================
# PHASE 2 MESSAGE INJECTION
# ============================================================


def inject_intelligence_context(
    messages,
    state,
    objective,
):
    """
    Add the current Phase 2 intelligence to the conversation.

    The context is deliberately added as a USER message because
    it represents fresh information produced after tool execution.

    IMPORTANT:
    This function must be called only AFTER the corresponding
    assistant tool-call message and ALL tool results from that
    assistant turn have been appended.

    This prevents the exact message-ordering problem:

        Unexpected role 'tool' after role 'user'
    """

    context = build_intelligence_context(
        state,
        objective,
    )

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


# ============================================================
# PHASE 2 DEBUG REPORT
# ============================================================


def intelligence_report(state):

    memory = getattr(
        state,
        "short_memory",
        None,
    )

    progress = getattr(
        state,
        "goal_progress",
        {},
    )

    reflections = getattr(
        state,
        "reflections",
        [],
    )

    diagnoses = getattr(
        state,
        "diagnoses",
        [],
    )

    contradictions = getattr(
        state,
        "contradictions",
        [],
    )

    confidence_history = getattr(
        state,
        "action_confidence",
        [],
    )

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
    ]

    if memory:
        report.extend(
            [
                "",
                "Recent memory:",
                memory.format_for_model(limit=5),
            ]
        )

    if diagnoses:
        report.extend(
            [
                "",
                "Latest diagnosis:",
                str(diagnoses[-1]),
            ]
        )

    if contradictions:
        report.extend(
            [
                "",
                "Latest contradiction:",
                contradictions[-1],
            ]
        )

    return "\n".join(report)


# ============================================================
# END PHASE 2
# ============================================================

# ============================================================
# PUBLIC ENTRY POINT
# ============================================================


def ask_model(
    message, verbose=False
):

    return run_agent(message,verbose=verbose)
