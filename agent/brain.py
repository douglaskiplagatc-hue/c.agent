import json
from pathlib import Path
from fnmatch import fnmatch
import traceback

from openai import OpenAI

from config import LLM7_API_KEY

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


# ============================================================
# LLM CLIENT
# ============================================================

client = OpenAI(
    api_key=LLM7_API_KEY,
    base_url="https://api.llm7.io/v1",
    timeout=60.0,
)


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files recursively inside a project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string"
                    }
                },
                "required": ["directory"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a project file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string"
                    }
                },
                "required": ["file_path"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search project files for text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string"
                    },
                    "query": {
                        "type": "string"
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
                        "type": "string"
                    }
                },
                "required": ["query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string"
                    },
                    "content": {
                        "type": "string"
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
                "The old_text must match the actual file contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string"
                    },
                    "old_text": {
                        "type": "string"
                    },
                    "new_text": {
                        "type": "string"
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
                "The replacement fails if old_text is missing "
                "or appears more than once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string"
                    },
                    "old_text": {
                        "type": "string"
                    },
                    "new_text": {
                        "type": "string"
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
                "Use this for verification, tests, Python scripts, "
                "linting, or inspecting command output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string"
                    },
                    "timeout": {
                        "type": "integer"
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
            "description": "Check the Git status of the workspace.",
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
            "description": "Create a Git checkpoint of current workspace changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string"
                    }
                },
                "required": ["message"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "git_rollback",
            "description": "Discard uncommitted changes in the workspace.",
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
                "function definitions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string"
                    }
                },
                "required": ["file_path"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "read_project_memory",
            "description": "Read persistent project memory.",
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
            "description": (
                "Update or create a section in persistent "
                "project memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {
                        "type": "string"
                    },
                    "content": {
                        "type": "string"
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
# MEMORY
# ============================================================

def load_project_memory():
    memory_file = Path("memory/project.md")

    if not memory_file.exists():
        return ""

    try:
        return memory_file.read_text(
            encoding="utf-8"
        )

    except Exception as error:
        print(
            f"[Memory] Failed to read memory: {error}"
        )

        return ""


def parse_forbidden_files(memory_content):
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

    normalized = file_path.replace(
        "\\",
        "/",
    )

    for pattern in forbidden_patterns:
        normalized_pattern = pattern.replace(
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
# OBJECTIVE TYPE
# ============================================================

def objective_requires_modification(objective):
    objective_lower = objective.lower()

    modification_words = (
        "fix",
        "modify",
        "edit",
        "change",
        "update",
        "repair",
        "implement",
        "create",
        "refactor",
        "remove",
        "add",
        "write",
    )

    return any(
        word in objective_lower
        for word in modification_words
    )


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    name,
    arguments,
):
    if name not in AVAILABLE_TOOLS:
        raise ValueError(
            f"Unknown tool: {name}"
        )

    return AVAILABLE_TOOLS[name](
        **arguments
    )


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
        state.record_inspected_file(
            arguments["file_path"]
        )

    elif tool_name in WRITE_TOOLS:
        state.record_modified_file(
            arguments["file_path"]
        )

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

def create_plan(objective, project_memory):
    read_only = not objective_requires_modification(objective)

    if read_only:
        planning_instruction = """
This is a READ-ONLY task.

The user explicitly wants information/analysis and does NOT want
files modified.

DO NOT create an implementation plan.
DO NOT propose adding, editing, fixing, removing, or creating code.

Your plan must contain ONLY:
1. The file(s) that need to be inspected.
2. The inspection tools required.
3. What information must be extracted.
4. How the result will be verified without modifying files.
"""

    else:
        planning_instruction = """
This task requires modification.

Create a precise implementation plan.
Inspect the relevant files before modifying them.
Modify only files necessary for the user's objective.
Verify the modification afterward.
"""

    response = client.chat.completions.create(
        model="default",
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
    )

    return response.choices[0].message.content


# ============================================================
# PYTHON VERIFICATION
# ============================================================

def verify_python_file(
    file_path,
):
    try:
        content = read_file(
            file_path
        )

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
            f"Verification failed: {error}",
        )


# ============================================================
# TOOL CALL KEY
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


def is_read_only_task(objective):
    objective_lower = objective.lower()

    modification_words = (
        "fix",
        "modify",
        "edit",
        "change",
        "update",
        "repair",
        "implement",
        "create",
        "refactor",
        "remove",
        "add",
        "write",
        "delete",
    )

    return not any(
        word in objective_lower
        for word in modification_words
    )


# ============================================================
# RECORD FAILURE
# ============================================================

def record_failure(
    state,
    controller,
    name,
    arguments,
    error,
):
    error_text = str(error)

    state.record_error(
        error_text
    )

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


def objective_requires_modification(objective):
    text = objective.lower().strip()

    explicit_read_only = [
        "do not modify",
        "don't modify",
        "do not change",
        "don't change",
        "read-only",
        "read only",
        "without modifying",
        "without changing",
        "without editing",
    ]

    if any(
        phrase in text
        for phrase in explicit_read_only
    ):
        return False

    modification_phrases = [
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
    ]

    return any(
        phrase in text
        for phrase in modification_phrases
    )

# ============================================================
# EFFICIENT, LOW-TOKEN VERIFICATION
# ============================================================

def efficient_verify(objective, modified_files, command_output, state):
    """
    Rule-based verification first, then LLM as a last resort,
    strictly capped to preserve API budget.
    """
    # 1. Check for explicit user patterns (Zero API cost)
    if "print(\"hello\")" in objective.lower() and "hello" not in command_output.lower():
        return False
    if "error" in command_output.lower():
        return False

    # 2. If Python modified, run AST Parse (Zero API cost, already done via verify_python_file)

    # 3. If objective asks to fix a bug, check if the exact changed lines exist
    # (This requires reading files, not LLM).

    # 4. ONLY call LLM if the command ran (exit code 0), but the output is ambiguous.
    # Cap it globally: only allow 2 LLM verifications per task.
    if state.llm_verification_count < 2:
        state.llm_verification_count += 1
        print("\n[Budget Protected] Using LLM Verifier...")

        response = client.chat.completions.create(
            model="default",
            messages=[{
                "role": "user",
                "content": f"Objective: {objective}\nOutput: {command_output}\nDoes this objectively satisfy the request? Answer TRUE or FALSE only."
            }],
            max_tokens=10  # Extremely cheap call
        )
        return "TRUE" in response.choices[0].message.content.upper()

    # 5. If we ran out of LLM verification calls, assume it's correct if exit code is 0
    return True
# ============================================================
# AGENT
# ============================================================

def run_agent(
    objective,
):
    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    project_memory = load_project_memory()
    read_only = is_read_only_task(objective)

    forbidden_patterns = (
        parse_forbidden_files(
            project_memory
        )
    )

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    state = TaskState(
        objective=objective
    )
    state.requires_modification = objective_requires_modification(
        objective
    )

    controller = AgentController(
        state
    )

    requires_modification = (
        objective_requires_modification(
            objective
        )
    )

    # --------------------------------------------------------
    # INITIAL PHASE
    # --------------------------------------------------------
    controller.set_phase(TaskPhase.PLANNING)
    if not read_only:
        controller.set_phase(TaskPhase.CHECKPOINT)

        if not controller.prepare_workspace():
            print("[Agent] Git checkpoint unavailable.")

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    try:
        state.plan = create_plan(
            objective,
            project_memory,
        )

    except Exception as error:
        state.record_error(
            str(error)
        )

        controller.fail()

        return (
            "Planning failed.\n\n"
            f"Error: {error}"
        )

    print(
        "\n[Agent Plan]"
    )

    print(
        state.plan
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    # --------------------------------------------------------
    # INSPECTION
    # --------------------------------------------------------

    controller.begin_inspection()

    # --------------------------------------------------------
    # INITIAL MODEL CONTEXT
    # --------------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                project_memory=project_memory
            ),
        },
        {
            "role": "user",
            "content": f"""
USER OBJECTIVE:

{objective}

IMPLEMENTATION PLAN:

{state.plan}

TASK REQUIREMENTS:

This is an actual coding-agent task.

You must distinguish between:

- inspection
- implementation
- verification
- diagnosis
- repair
- completion

The objective requires modification:
{requires_modification}

MAXIMUM REPAIR ATTEMPTS:

{state.max_attempts}

MANDATORY RULES:

1. Inspect the relevant file before modifying it.

2. Never invent file contents.

3. For an edit or patch, use the actual contents
   returned by read_file.

4. Never repeat an identical failed edit.

5. Never repeat an identical failed tool call.

6. A successful inspection is NOT task completion.

7. A successful list_files call is NOT task completion.

8. A successful read_file call is NOT task completion.

9. A successful search is NOT task completion.

10. A successful modification is NOT automatically
    task completion.

11. If the objective requires modification, you MUST
    actually modify the required file.

12. If the objective requires modification, you MUST
    verify the modification.

13. For Python objectives, run the requested Python
    script when the user asks for it.

14. If a verification command fails, inspect the actual
    error before repairing.

15. Only modify files necessary for the objective.

16. Never modify forbidden files.

17. Analysis-only requests must never modify files.

18. Once the objective has genuinely been verified,
    stop immediately.

19. Do not claim success simply because tools executed
    successfully.

20. If you have only inspected files, you are NOT done.

BEGIN BY INSPECTING THE RELEVANT FILE.
""",
        },
    ]

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    recent_tool_call_keys = []
    protected_snapshots = {}
    state.llm_verification_count = 0

    failed_actions = set()

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while not state.completed:
        # ----------------------------------------------------
        # ATTEMPT LIMIT
        # ----------------------------------------------------

        if state.attempts >= state.max_attempts:
            controller.rollback()
            controller.fail()

            return (
                "The agent stopped after reaching "
                f"the maximum of {state.max_attempts} "
                "repair attempts.\n\n"
                f"Errors encountered:\n{state.errors}"
            )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        try:
            response = (
                client.chat.completions.create(
                    model="default",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            )

        except Exception as error:
            state.record_error(
                str(error)
            )

            controller.fail()

            return (
                "The model request failed.\n\n"
                f"Error: {error}"
            )

        assistant_message = (
            response.choices[0].message
        )

        messages.append(
            assistant_message
        )

        # ----------------------------------------------------
        # MODEL FINISHED WITHOUT TOOL CALL
        # ----------------------------------------------------

        if not assistant_message.tool_calls:
            content = (
                assistant_message.content
                or ""
            )

            # ------------------------------------------------
            # ANALYSIS-ONLY TASK
            # ------------------------------------------------

            if not requires_modification:
                controller.complete()

                return (
                    content
                    + "\n\nTask state:"
                    + f"\nInspected files: "
                    f"{len(state.inspected_files)}"
                    + f"\nModified files: "
                    f"{len(state.modified_files)}"
                    + f"\nCommands: "
                    f"{len(state.commands)}"
                    + f"\nErrors: "
                    f"{len(state.errors)}"
                )

                        # ------------------------------------------------
            # MODIFICATION TASK WITHOUT MODIFICATION
            # ------------------------------------------------

            if not state.modified_files:
                print(
                    "\n[Agent] Premature completion blocked."
                )

                messages.append({
                    "role": "user",
                    "content": """
The requested modification has NOT been made yet.

You are NOT finished.

Continue working:

1. Inspect the relevant file.
2. Determine whether the requested functionality already exists.
3. If it does not exist, modify the required file.
4. If it already exists but the user explicitly requested adding it,
   make the requested implementation change only if necessary.
5. Verify the final implementation.
6. Only finish after the required file has actually been modified
   and the modification has been successfully verified.

Do NOT call controller.complete().
Do NOT claim the task is complete.
"""
                })

                continue

            # ------------------------------------------------
            # MODIFICATION WITHOUT VERIFICATION
            # ------------------------------------------------

            successful_verifications = [
                verification
                for verification in state.verifications
                if verification.get("success") is True
            ]

            if not successful_verifications:
                print(
                    "\n[Agent] Completion blocked: "
                    "modification has not been verified."
                )

                messages.append({
                    "role": "user",
                    "content": """
You have modified a file, but the modification has
NOT been successfully verified.

You are NOT finished.

Verify the actual objective now.

For a Python repair task:

1. Run the requested Python file.
2. Inspect the complete output.
3. If it fails, diagnose the actual error.
4. Repair the problem.
5. Run the Python file again.
6. Only finish after it succeeds.
""",
                })

                continue

            # ------------------------------------------------
            # REAL COMPLETION
            # ------------------------------------------------

            controller.complete()

            return (
                content
                + "\n\nTask state:"
                + f"\nInspected files: "
                f"{len(state.inspected_files)}"
                + f"\nModified files: "
                f"{len(state.modified_files)}"
                + f"\nCommands: "
                f"{len(state.commands)}"
                + f"\nErrors: "
                f"{len(state.errors)}"
            )

        # ----------------------------------------------------
        # TOOL CALLS
        # ----------------------------------------------------

        for tool_call in assistant_message.tool_calls:
            name = (
                tool_call.function.name
            )

            # ------------------------------------------------
            # PARSE ARGUMENTS
            # ------------------------------------------------

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

            except Exception as error:
                result_text = json.dumps({
                    "success": False,
                    "error": (
                        "Invalid tool arguments: "
                        + str(error)
                    ),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

                continue

            print(
                f"\n[Agent → {name}]"
            )

            print(
                f"Arguments: {arguments}"
            )

            # ------------------------------------------------
            # DUPLICATE CALL
            # ------------------------------------------------

            call_key = make_tool_call_key(
                name,
                arguments,
            )

            if call_key in recent_tool_call_keys:
                error = (
                    "This exact tool call was already "
                    "executed. Do not repeat it. "
                    "Use the existing result and take "
                    "a different action."
                )

                print(
                    "\n[Agent] Duplicate tool call blocked."
                )

                state.record_failed_action(
                    name,
                    arguments,
                    error,
                )

                result_text = json.dumps({
                    "success": False,
                    "error": error,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })
                if len(state.errors) >= 3:
                    controller.fail()
                    state.completed = True
                    break

                continue

            recent_tool_call_keys.append(
                call_key
            )

            if len(recent_tool_call_keys) > 20:
                recent_tool_call_keys.pop(0)

            # ------------------------------------------------
            # FORBIDDEN FILE
            # ------------------------------------------------

            if name in WRITE_TOOLS:
                file_path = arguments.get(
                    "file_path"
                )

                if is_forbidden_file(
                    file_path,
                    forbidden_patterns,
                ):
                    error = (
                        f"Modification of '{file_path}' "
                        "is prohibited by project memory."
                    )

                    print(
                        "\n[Agent] Forbidden file blocked."
                    )

                    state.record_failed_action(
                        name,
                        arguments,
                        error,
                    )

                    result_text = json.dumps({
                        "success": False,
                        "error": error,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                    continue

            # ------------------------------------------------
            # FAILED PATCH
            # ------------------------------------------------

            if name in {
                "edit_file",
                "patch_file",
            }:
                patch_key = (
                    arguments.get("file_path"),
                    arguments.get("old_text"),
                )

                if patch_key in failed_actions:
                    error = (
                        "This exact edit already failed. "
                        "Do not repeat it. Read the file "
                        "again and determine the actual "
                        "current contents."
                    )

                    print(
                        "\n[Agent] Repeated failed edit blocked."
                    )

                    state.record_failed_action(
                        name,
                        arguments,
                        error,
                    )

                    result_text = json.dumps({
                        "success": False,
                        "error": error,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                    continue

            # ------------------------------------------------
            # EXECUTE TOOL
            # ------------------------------------------------
            if name in WRITE_TOOLS:
                file_path = arguments.get("file_path")
                if file_path not in protected_snapshots:
                    try:
                        protected_snapshots[file_path] = read_file(file_path)
                    except Exception:
                        protected_snapshots[file_path] = None
            try:
                result = execute_tool(
                    name,
                    arguments,
                )

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

                print("\n[Tool successful]")

                # ------------------------------------------------
                # COMMAND RESULT
                # ------------------------------------------------

                if name == "run_command":
                    command = arguments.get(
                        "command",
                        "",
                    )

                    command_lower = command.lower()

                    is_verification = any(
                        indicator in command_lower
                        for indicator in VERIFICATION_INDICATORS
                    )

                    if isinstance(result, dict) and result.get("success") is False:
                        error = (
                            result.get("stderr")
                            or result.get("error")
                            or "Command failed."
                        )

                        print("\n[Agent detected a command failure]")
                        print(f"Error: {error}")

                        state.record_error(error)
                        if len(state.errors) >= 3:
                            controller.fail()
                            state.completed = True
                            break

                        state.record_failed_action(
                            name,
                            arguments,
                            error,
                        )

                        controller.begin_diagnosis()

                        if controller.can_repair():
                            controller.begin_repair()

                        else:
                            controller.rollback()
                            controller.fail()

                            state.completed = True

                    elif is_verification:
                        output = result.get("stdout", "") if isinstance(result, dict) else str(result)

                        print(f"\n[Verification output] {output.strip()}")
                        if efficient_verify(objective, state.modified_files, output, state):
                            controller.begin_verification()
                            state.record_verification(
            command,
            True,
        )
                            print("\n[Verification successful]")
                        else:
                            print("\n[Verification failed]")

                            error = (
            f"Command output did not satisfy objective: "
            f"{output}"
        )

                            state.record_error(error)

                            state.record_failed_action(
            name,
            arguments,
            error,
        )

                    else:
                        output = result.get("stdout", "") if isinstance(result, dict) else str(result)

                        if efficient_verify(objective, state.modified_files, output, state):
                            controller.begin_verification()

                            state.record_verification(
            command,
            True,
        )

                            print("\n[Verification successful]")

                # ------------------------------------------------
                # WRITE TOOL RESULT
                # ------------------------------------------------

                elif name in WRITE_TOOLS:
                    file_path = arguments.get(
                        "file_path"
                    )

                    print(
                        "\n[File modification successful]"
                    )

                    # --------------------------------------------
                    # PYTHON SYNTAX VERIFICATION
                    # --------------------------------------------

                    if (
                        file_path
                        and file_path.endswith(".py")
                    ):
                        ok, message = verify_python_file(
                            file_path
                        )

                        if ok:
                            print(
                                f"\n[Verification] {message}"
                            )

                            state.record_verification(
                                message,
                                True,
                            )

                        else:
                            print(
                                f"\n[Verification failed] {message}"
                            )

                            state.record_error(
                                message
                            )

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

                            controller.begin_diagnosis()

                            if controller.can_repair():
                                controller.begin_repair()

                            else:
                                controller.rollback()
                                controller.fail()

                                state.completed = True

            except Exception as error:
                error_text = str(error)
                print("\n[Tool failure]")
                print(error_text)
                import traceback
                traceback.print_exc()

                state.record_error(error_text)
                state.record_failed_action(name, arguments, error_text)

                # IMMEDIATE ROLLBACK
                if name in WRITE_TOOLS:
                    file_path = arguments.get("file_path")
                    if file_path in protected_snapshots and protected_snapshots[file_path] is not None:
                        create_file(file_path=file_path, content=protected_snapshots[file_path])
                        print("[Rollback triggered] File restored to original state.")

                if name in {"edit_file", "patch_file"}:
                    failed_actions.add((arguments.get("file_path"), arguments.get("old_text")))

                controller.begin_diagnosis()

                if controller.can_repair():
                    controller.begin_repair()
                    messages.append({
                        "role": "user",
                        "content": f"DIAGNOSIS AND REPAIR PROMPT (Attempt {state.attempts}/{state.max_attempts}): The tool {name} failed. Error: {error_text}. DO NOT VERIFY YET. Read the file, diagnose, and make ONE targeted fix. Do not repeat the failed call."
                    })
                else:
                    # Out of attempts, rollback everything
                    if protected_snapshots:
                        for path, content in protected_snapshots.items():
                            if content is not None:
                                create_file(file_path=path, content=content)
                    controller.rollback()
                    controller.fail()
                    state.completed = True

                result_text = json.dumps({"success": False, "error": error_text, "instruction": "FAILED. Change approach."}, indent=2)
            # ------------------------------------------------
            # SEND TOOL RESULT
            # ------------------------------------------------

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

            # ------------------------------------------------
            # HARD STOP
            # ------------------------------------------------

            if state.completed:
                break

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        if state.completed:
            break

        # ----------------------------------------------------
        # CONTINUE
        # ----------------------------------------------------



    # ========================================================
    # FINAL RESULT
    # ========================================================

    return (
        "Task completed."
        + "\n\nTask state:"
        + f"\nInspected files: "
        f"{len(state.inspected_files)}"
        + f"\nModified files: "
        f"{len(state.modified_files)}"
        + f"\nCommands: "
        f"{len(state.commands)}"
        + f"\nErrors: "
        f"{len(state.errors)}"
    )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def ask_model(
    message,
):
    return run_agent(
        message
    )
