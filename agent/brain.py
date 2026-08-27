import json
from pathlib import Path
from fnmatch import fnmatch

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
                "Use this for verification, tests, linting, "
                "or inspecting command output."
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
            "description": (
                "Create a Git checkpoint of current workspace changes."
            ),
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
            "description": (
                "Discard uncommitted changes in the workspace."
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
            "description": (
                "Read the persistent project memory."
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

    normalized = file_path.replace("\\", "/")

    for pattern in forbidden_patterns:

        pattern = pattern.replace("\\", "/")

        if fnmatch(normalized, pattern):
            return True

    return False


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

    elif tool_name in {
        "edit_file",
        "patch_file",
        "create_file",
    }:

        state.record_modified_file(
            arguments["file_path"]
        )

    elif tool_name == "run_command":

        state.record_command(
            arguments["command"]
        )

        if isinstance(result, dict):

            if not result.get(
                "success",
                False,
            ):

                error = (
                    result.get("stderr")
                    or result.get("error")
                    or "Command failed."
                )

                state.record_error(error)


# ============================================================
# PLANNING
# ============================================================

def create_plan(
    objective,
    project_memory,
):

    response = client.chat.completions.create(

        model="default",

        messages=[
            {
                "role": "system",
                "content": (
                    PLANNING_PROMPT
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

    return (
        response.choices[0]
        .message.content
    )


# ============================================================
# PYTHON VERIFICATION
# ============================================================

def verify_python_file(
    file_path
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
                "Unable to read file after modification."
            )

        compile(
            content,
            file_path,
            "exec",
        )

        return (
            True,
            f"Python syntax verification passed for {file_path}."
        )

    except SyntaxError as error:

        return (
            False,
            f"Syntax error in {file_path}: {error}"
        )

    except Exception as error:

        return (
            False,
            f"Verification failed: {error}"
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


# ============================================================
# AGENT
# ============================================================

def run_agent(
    objective
):

    # --------------------------------------------------------
    # LOAD MEMORY
    # --------------------------------------------------------

    project_memory = (
        load_project_memory()
    )

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

    controller = AgentController(
        state
    )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    controller.set_phase(
        TaskPhase.CHECKPOINT
    )

    controller.set_phase(
        TaskPhase.PLANNING
    )

    # --------------------------------------------------------
    # CREATE PLAN
    # --------------------------------------------------------

    state.plan = create_plan(
        objective,
        project_memory,
    )

    print(
        "\n[Agent Plan]"
    )

    print(
        state.plan
    )

    # --------------------------------------------------------
    # PREPARE WORKSPACE
    # --------------------------------------------------------

    if not controller.prepare_workspace():

        print(
            "[Agent] Git checkpoint unavailable."
        )

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

TASK STATE:

Maximum repair attempts:
{state.max_attempts}

IMPORTANT OPERATING RULES:

1. Inspect files before modifying them.

2. Never invent file contents.

3. When editing a file, first read the actual current
   contents of that file.

4. Never blindly repeat an edit that failed because
   old_text was not found.

5. A successful tool call does NOT mean the user's
   objective is complete.

6. list_files, search_code, search_project, read_file,
   and similar inspection tools are NOT verification
   of the user's objective.

7. After making a modification, verify the modification.

8. For Python files, compile the file or run the requested
   Python test/script.

9. If a command fails, inspect the actual error before
   attempting a repair.

10. Never repeat the exact same failed tool call.

11. Only modify files necessary for the objective.

12. Do not modify forbidden files from project memory.

13. If the user's objective is analysis-only, do NOT modify
    anything.

14. Once the user's objective has genuinely been verified,
    STOP immediately.

15. Do not continue making unnecessary changes after
    successful verification.

Begin by inspecting the project.
""",
        },
    ]

    # --------------------------------------------------------
    # CALL HISTORY
    # --------------------------------------------------------

    recent_tool_calls = []

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
        # ASK MODEL
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
        # MODEL FINISHED
        # ----------------------------------------------------

        if not assistant_message.tool_calls:

            content = (
                assistant_message.content
                or "Task completed."
            )

            # The model is allowed to finish only when it has
            # decided the objective is complete.
            controller.complete()

            return (
                content
                + "\n\n"
                + "Task state:"
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
        # PROCESS TOOL CALLS
        # ----------------------------------------------------

        for tool_call in assistant_message.tool_calls:

            name = (
                tool_call.function.name
            )

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
            # DUPLICATE TOOL CALL
            # ------------------------------------------------

            call_key = make_tool_call_key(
                name,
                arguments,
            )

            if call_key in recent_tool_calls:

                error = (
                    "This exact tool call was already "
                    "executed recently. Do not repeat it. "
                    "Inspect the result and take a different "
                    "action if necessary."
                )

                print(
                    "\n[Agent] Duplicate tool call blocked."
                )

                result_text = json.dumps({
                    "success": False,
                    "error": error,
                })

                state.record_failed_action(
                    name,
                    arguments,
                    error,
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

                continue

            recent_tool_calls.append(
                call_key
            )

            if len(recent_tool_calls) > 15:

                recent_tool_calls.pop(0)

            # ------------------------------------------------
            # FORBIDDEN FILE PROTECTION
            # ------------------------------------------------

            write_tools = {
                "create_file",
                "edit_file",
                "patch_file",
            }

            if name in write_tools:

                file_path = (
                    arguments.get(
                        "file_path"
                    )
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

                    result_text = json.dumps({
                        "success": False,
                        "error": error,
                    })

                    state.record_failed_action(
                        name,
                        arguments,
                        error,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                    continue

            # ------------------------------------------------
            # FAILED EDIT PROTECTION
            # ------------------------------------------------

            if name in {
                "edit_file",
                "patch_file",
            }:

                file_path = (
                    arguments.get(
                        "file_path"
                    )
                )

                old_text = (
                    arguments.get(
                        "old_text"
                    )
                )

                patch_key = (
                    file_path,
                    old_text,
                )

                if patch_key in failed_actions:

                    error = (
                        "This exact edit already failed. "
                        "Do NOT repeat it. Read the file again "
                        "and use the actual current contents."
                    )

                    print(
                        "\n[Agent] Repeated failed edit blocked."
                    )

                    result_text = json.dumps({
                        "success": False,
                        "error": error,
                    })

                    state.record_failed_action(
                        name,
                        arguments,
                        error,
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                    continue

            # ------------------------------------------------
            # EXECUTE TOOL
            # ------------------------------------------------

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

                # --------------------------------------------
                # TOOL RESULT
                # --------------------------------------------

                result_text = json.dumps(
                    result,
                    indent=2,
                    default=str,
                )

                # --------------------------------------------
                # COMMAND
                # --------------------------------------------

                if name == "run_command":

                    success = (
                        isinstance(
                            result,
                            dict,
                        )
                        and result.get(
                            "success",
                            False,
                        )
                    )

                    if success:

                        print(
                            "\n[Verification successful]"
                        )

                        state.record_verification(
                            arguments.get(
                                "command",
                                "command",
                            ),
                            True,
                        )

                    else:

                        error = (
                            result.get("stderr")
                            or result.get("error")
                            or "Command failed."
                        )

                        print(
                            "\n[Agent detected a command failure]"
                        )

                        print(
                            f"Error: {error}"
                        )

                        state.record_error(
                            error
                        )

                        state.record_failed_action(
                            name,
                            arguments,
                            error,
                        )

                        controller.begin_diagnosis()

                        diagnosis = (
                            result.get("stderr")
                            or result.get("error")
                            or result.get("stdout")
                            or "Unknown failure"
                        )

                        print(
                            "\n[Failure Diagnosis]"
                        )

                        print(
                            diagnosis
                        )

                        if controller.can_repair():

                            controller.begin_repair()

                        else:

                            print(
                                "\n[Agent] Maximum repair "
                                "attempts reached."
                            )

                            controller.rollback()
                            controller.fail()

                            state.completed = True

                            break

                # --------------------------------------------
                # WRITE TOOLS
                # --------------------------------------------

                elif name in write_tools:

                    success = (
                        isinstance(
                            result,
                            dict,
                        )
                        and result.get(
                            "success",
                            False,
                        )
                    )

                    if not success:

                        error = (
                            result.get("error")
                            if isinstance(
                                result,
                                dict,
                            )
                            else "File modification failed."
                        )

                        print(
                            "\n[File modification failed]"
                        )

                        print(
                            f"Error: {error}"
                        )

                        state.record_error(
                            error
                        )

                        state.record_failed_action(
                            name,
                            arguments,
                            error,
                        )

                        # ------------------------------------
                        # Remember failed patch
                        # ------------------------------------

                        if name in {
                            "edit_file",
                            "patch_file",
                        }:

                            failed_actions.add(
                                (
                                    arguments.get(
                                        "file_path"
                                    ),
                                    arguments.get(
                                        "old_text"
                                    ),
                                )
                            )

                        controller.begin_diagnosis()

                        if controller.can_repair():

                            controller.begin_repair()

                        else:

                            controller.rollback()
                            controller.fail()

                            state.completed = True

                            break

                    else:

                        print(
                            "\n[File modification successful]"
                        )

                        # ------------------------------------
                        # Automatic Python syntax verification
                        # ------------------------------------

                        file_path = (
                            arguments.get(
                                "file_path"
                            )
                        )

                        if (
                            file_path
                            and file_path.endswith(".py")
                        ):

                            ok, message = (
                                verify_python_file(
                                    file_path
                                )
                            )

                            if ok:

                                print(
                                    f"\n[Verification] "
                                    f"{message}"
                                )

                                state.record_verification(
                                    message,
                                    True,
                                )

                            else:

                                print(
                                    f"\n[Verification failed] "
                                    f"{message}"
                                )

                                state.record_error(
                                    message
                                )

                                state.record_failed_action(
                                    name,
                                    arguments,
                                    message,
                                )

                                if name in {
                                    "edit_file",
                                    "patch_file",
                                }:

                                    failed_actions.add(
                                        (
                                            file_path,
                                            arguments.get(
                                                "old_text"
                                            ),
                                        )
                                    )

                                controller.begin_diagnosis()

                                if controller.can_repair():

                                    controller.begin_repair()

                                else:

                                    controller.rollback()
                                    controller.fail()

                                    state.completed = True

                                    break

                # --------------------------------------------
                # OTHER TOOLS
                # --------------------------------------------

                else:

                    print(
                        "\n[Tool successful]"
                    )

            # ------------------------------------------------
            # TOOL EXCEPTION
            # ------------------------------------------------

            except Exception as error:

                error_text = str(error)

                print(
                    "\n[Tool failure]"
                )

                print(
                    error_text
                )

                state.record_error(
                    error_text
                )

                state.record_failed_action(
                    name,
                    arguments,
                    error_text,
                )

                # --------------------------------------------
                # Remember failed edit
                # --------------------------------------------

                if name in {
                    "edit_file",
                    "patch_file",
                }:

                    failed_actions.add(
                        (
                            arguments.get(
                                "file_path"
                            ),
                            arguments.get(
                                "old_text"
                            ),
                        )
                    )

                result_text = json.dumps({
                    "success": False,
                    "error": error_text,
                    "instruction": (
                        "THIS ACTION FAILED. "
                        "Do not repeat the exact same action. "
                        "Inspect the actual current state and "
                        "choose a different approach."
                    ),
                })

                controller.begin_diagnosis()

                if controller.can_repair():

                    controller.begin_repair()

                else:

                    controller.rollback()
                    controller.fail()

                    state.completed = True

                    break

            # ------------------------------------------------
            # SEND TOOL RESULT BACK TO MODEL
            # ------------------------------------------------

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

        # ----------------------------------------------------
        # STOP IF COMPLETED
        # ----------------------------------------------------

        if state.completed:
            break

        # ----------------------------------------------------
        # CONTINUE AFTER TOOLS
        # ----------------------------------------------------

        messages.append({
            "role": "user",
            "content": """
Continue working on the objective.

IMPORTANT:

A successful tool call does NOT automatically mean the
objective is complete.

If you are still inspecting:
- inspect the relevant file
- read the actual contents
- analyze the code

If implementation is required:
- make the smallest necessary change

If a modification was made:
- verify it

If verification fails:
- inspect the actual failure
- diagnose the cause
- make one targeted correction
- verify again

NEVER:
- repeat an identical tool call
- repeat an edit whose old_text was not found
- invent file contents
- modify unrelated files

When the user's objective has genuinely been verified,
stop immediately.
""",
        })

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
    message
):

    return run_agent(
        message
    )
