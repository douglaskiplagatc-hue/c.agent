import json
from pathlib import Path

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


client = OpenAI(
    api_key=LLM7_API_KEY,
    base_url="https://api.llm7.io/v1",
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
                    "directory": {"type": "string"}
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
                    "file_path": {"type": "string"}
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
                    "directory": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["directory", "query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "search_project",
            "description": "Search the project index for relevant files, Python functions, and classes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
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
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact section of an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
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
            "description": "Safely replace one exact section of a file. Replacement is refused if the section is missing or duplicated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
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
            "description": "Run a terminal command inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
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
                    "message": {"type": "string"}
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
            "description": "Analyze a Python file and return its actual function definitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"}
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
            "description": "Update a section in persistent project memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["section", "content"],
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

    return memory_file.read_text(
        encoding="utf-8"
    )


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(name, arguments):

    if name not in AVAILABLE_TOOLS:
        raise ValueError(
            f"Unknown tool: {name}"
        )

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

    return response.choices[0].message.content


# ============================================================
# AGENT
# ============================================================

def run_agent(objective):

    project_memory = load_project_memory()

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
    # PLAN
    # --------------------------------------------------------

    state.plan = create_plan(
        objective,
        project_memory,
    )

    print("\n[Agent Plan]")
    print(state.plan)

    # --------------------------------------------------------
    # WORKSPACE CHECKPOINT
    # --------------------------------------------------------

    if not controller.prepare_workspace():

        print(
            "[Agent] Git checkpoint unavailable."
        )

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

You have a maximum of {state.max_attempts} repair attempts.

Rules:

1. Inspect before modifying.
2. Never invent file contents.
3. Never repeat a failed edit blindly.
4. After modifying code, verify it.
5. If verification succeeds, STOP.
6. Do not continue making unnecessary changes after success.
7. If a tool fails, inspect the actual failure before retrying.
8. Only modify files necessary for the objective.
""",
        },
    ]

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while not state.completed:

        if state.attempts >= state.max_attempts:

            return (
                "The agent stopped after reaching the "
                f"maximum of {state.max_attempts} attempts.\n\n"
                f"Errors encountered:\n{state.errors}"
            )

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        messages.append(
            assistant_message
        )

        # ----------------------------------------------------
        # MODEL FINISHED WITHOUT TOOL
        # ----------------------------------------------------

        if not assistant_message.tool_calls:

            state.completed = True
            state.success = True

            content = (
                assistant_message.content
                or "Task completed."
            )

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

        successful_verification = False

        for tool_call in assistant_message.tool_calls:

            name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(
                f"\n[Agent → {name}]"
            )

            print(
                f"Arguments: {arguments}"
            )

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

                # ------------------------------------------------
                # COMMAND RESULT
                # ------------------------------------------------

                if name == "run_command":

                    success = (
                        isinstance(result, dict)
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
                            arguments["command"],
                            True,
                        )

                        successful_verification = True

                    else:

                        error = (
                            result.get("stderr")
                            or result.get("error")
                            or "Command failed."
                        )

                        print(
                            "\n[Agent detected a command failure]"
                        )

                        print(error)

                        state.record_error(
                            error
                        )

                        state.attempts += 1

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

                        print(diagnosis)

                        if controller.can_repair():

                            controller.begin_repair()

                        else:

                            controller.rollback()
                            controller.fail()

                            state.completed = True

                            break

                else:

                    print(
                        "\n[Tool successful]"
                    )

            except Exception as error:

                state.attempts += 1

                error_text = str(error)

                state.record_error(
                    error_text
                )

                state.record_failed_action(
                    name,
                    arguments,
                    error_text,
                )

                result_text = json.dumps({
                    "success": False,
                    "error": error_text,
                })

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            })

            # ----------------------------------------------------
            # HARD STOP AFTER SUCCESSFUL VERIFICATION
            # ----------------------------------------------------

            if successful_verification:

                state.completed = True
                state.success = True

                print(
                    "\n[PHASE → COMPLETE]"
                )

                break

        # ----------------------------------------------------
        # STOP — DO NOT ASK THE MODEL TO CONTINUE
        # ----------------------------------------------------

        if state.completed:

            break

        # ----------------------------------------------------
        # FAILURE / CONTINUE
        # ----------------------------------------------------

        messages.append({
            "role": "user",
            "content": """
Continue the objective.

If a tool failed:

1. Read the exact error.
2. Inspect the relevant file.
3. Determine the actual cause.
4. Make one targeted correction.
5. Verify the correction.

Never repeat an edit whose old_text was not found.

If verification succeeds, stop immediately.
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


def ask_model(message):

    return run_agent(message)
