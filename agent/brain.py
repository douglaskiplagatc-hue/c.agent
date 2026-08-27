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
    analyze_python_file,search_project,read_project_memory,
update_project_memory,
)
from agent.prompts import SYSTEM_PROMPT, PLANNING_PROMPT
from agent.state import TaskState
from agent.controller import (
    AgentController,
    TaskPhase,
)

client = OpenAI(
    api_key=LLM7_API_KEY,
    base_url="https://api.llm7.io/v1",
)


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
                        "type": "string",
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
                        "type": "string",
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
            "Search the project index for relevant "
            "files, Python functions, and classes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Concept, filename, function, "
                        "or class to search for."
                    ),
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
        "name": "patch_file",
        "description": (
            "Safely replace one exact section of a file. "
            "The replacement is refused if the section is "
            "missing or appears more than once."
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
            "name": "edit_file",
            "description": "Replace one exact section of an existing file.",
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
            "description": "Run a terminal command inside the workspace.",
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
{
    "type": "function",
    "function": {
        "name": "analyze_python_file",
        "description": (
            "Analyze a Python file and return "
            "its actual function definitions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                }
            },
            "required": [
                "file_path"
            ],
        },
    },
},
]


AVAILABLE_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "create_file": create_file,
    "edit_file": edit_file,
    "run_command": run_command,
    "git_status": git_status,
    "git_checkpoint": git_checkpoint,
    "git_rollback": git_rollback,
    "analyze_python_file": analyze_python_file,
    "patch_file": patch_file,"search_project": search_project,
}


def load_project_memory() -> str:

    memory_file = Path("memory/project.md")

    if not memory_file.exists():
        return ""

    return memory_file.read_text(
        encoding="utf-8"
    )


def execute_tool(name, arguments):

    if name not in AVAILABLE_TOOLS:
        raise ValueError(
            f"Unknown tool: {name}"
        )

    return AVAILABLE_TOOLS[name](**arguments)


def update_state(
    state: TaskState,
    tool_name: str,
    arguments: dict,
    result,
):

    if tool_name == "read_file":

        state.record_inspected_file(
            arguments["file_path"]
        )

    elif tool_name == "edit_file":

        state.record_modified_file(
            arguments["file_path"]
        )

    elif tool_name == "create_file":

        state.record_modified_file(
            arguments["file_path"]
        )

    elif tool_name == "run_command":

        state.record_command(
            arguments["command"]
        )

        if isinstance(result, dict):

            if not result.get("success", False):

                error = (
                    result.get("stderr")
                    or result.get("error")
                    or "Command failed."
                )

                state.record_error(error)


def create_plan(
    objective: str,
    project_memory: str,
) -> str:

    response = client.chat.completions.create(
        model="default",
        messages=[
            {
                "role": "system",
                "content": PLANNING_PROMPT
                + "\n\nPROJECT MEMORY:\n"
                + project_memory,
            },
            {
                "role": "user",
                "content": objective,
            },
        ],
    )

    return response.choices[0].message.content


def run_agent(objective: str) -> str:

    project_memory = load_project_memory()

    state = TaskState(
        objective=objective
    )
    controller = AgentController(
        state
    )

    controller.set_phase(
    TaskPhase.CHECKPOINT
)
    controller.set_phase(
    TaskPhase.PLANNING
)
    state.plan = create_plan(
        objective,
        project_memory,
    )

    print("\n[Agent Plan]")
    print(state.plan)
    if not controller.prepare_workspace():
        print(
        "[Agent] Git checkpoint unavailable."
    )
        controller.begin_inspection()


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

Begin by inspecting the project.
Then implement the objective.
Verify your work before finishing.
""",
        },
    ]

    while not state.completed:

        if state.attempts >= state.max_attempts:

            return (
                "The agent stopped after reaching the "
                f"maximum of {state.max_attempts} attempts.\n\n"
                f"Errors encountered:\n{state.errors}"
            )

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

        if not assistant_message.tool_calls:

            state.completed = True

            return (
                assistant_message.content
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

                if name == "run_command":

                    if isinstance(result, dict):

                        if not result.get(
                            "success",
                            False,
                        ):
                            print(
                "\n[Agent detected a command failure]"
            )
                            controller.begin_repair()

                        else:

                            print(
                                "\n[Verification "
                                "successful]"
                            )

            except Exception as error:

                state.attempts += 1

                error_text = str(error)

                state.record_error(
                    error_text
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

        messages.append({
            "role": "user",
            "content": """
Continue working on the objective.
Use the available tools directly.
Do not merely describe commands or tool calls.
Actually call the appropriate tool.

If something failed:
1. Examine the actual error.
2. Determine the cause.
3. Inspect relevant files if necessary.
4. Make a targeted correction.
5. Run verification again.

Do not recreate an existing file unless necessary.

Stop when the objective has been successfully verified.
""",
        })


def ask_model(message: str) -> str:
    return run_agent(message)
