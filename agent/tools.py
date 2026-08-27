from pathlib import Path
import subprocess
from agent.indexer import search_index

WORKSPACE = Path("workspace").resolve()


def safe_path(file_path: str) -> Path:
    """Ensure the path stays inside the workspace."""

    path = Path(file_path).resolve()

    try:
        path.relative_to(WORKSPACE)
    except ValueError:
        raise PermissionError(
            f"Access denied: {file_path} is outside the workspace."
        )

    return path


def list_files(directory: str = "workspace") -> list[str]:
    """List files recursively."""

    path = Path(directory).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    return [
        str(file)
        for file in path.rglob("*")
        if file.is_file()
    ]


def read_file(file_path: str) -> str:
    """Read a text file."""

    path = safe_path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Not a file: {file_path}"
        )

    return path.read_text(encoding="utf-8")


def search_code(directory: str, query: str) -> list[dict]:
    """Search text files for a query."""

    path = Path(directory).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    results = []

    for file in path.rglob("*"):

        if not file.is_file():
            continue

        try:
            content = file.read_text(
                encoding="utf-8"
            )
        except (UnicodeDecodeError, PermissionError):
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):

            if query.lower() in line.lower():

                results.append({
                    "file": str(file),
                    "line": line_number,
                    "content": line.strip(),
                })

    return results


def create_file(
    file_path: str,
    content: str,
) -> str:
    """Create a new file."""

    path = safe_path(file_path)

    if path.exists():
        raise FileExistsError(
            f"File already exists: {file_path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    return f"Created file: {file_path}"


def edit_file(
    file_path: str,
    old_text: str,
    new_text: str,
) -> str:
    """Replace an exact section of a file."""

    path = safe_path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    content = path.read_text(
        encoding="utf-8"
    )

    if old_text not in content:
        raise ValueError(
            "The requested old_text was not found "
            f"in {file_path}"
        )

    updated = content.replace(
        old_text,
        new_text,
        1,
    )

    path.write_text(
        updated,
        encoding="utf-8",
    )

    return f"Updated file: {file_path}"


def run_command(
    command: str,
    timeout: int = 30,
) -> dict:
    """Run a terminal command from the project root."""

    project_root = Path.cwd()

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "command": command,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

    except subprocess.TimeoutExpired:

        return {
            "command": command,
            "success": False,
            "error": f"Command timed out after {timeout} seconds.",
        }
def git_status() -> dict:
    """Return the Git status of the workspace."""

    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def git_checkpoint(message: str = "Agent checkpoint") -> dict:
    """Create a Git checkpoint for the current workspace."""

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )

    if status.returncode != 0:
        return {
            "success": False,
            "error": (
                "Workspace is not a Git repository.\n"
                + status.stderr
            ),
        }

    if not status.stdout.strip():
        return {
            "success": True,
            "message": "No changes to checkpoint.",
        }

    add = subprocess.run(
        ["git", "add", "-A"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )

    if add.returncode != 0:
        return {
            "success": False,
            "error": add.stderr,
        }

    commit = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )

    return {
        "success": commit.returncode == 0,
        "stdout": commit.stdout,
        "stderr": commit.stderr,
    }


def git_rollback() -> dict:
    """Discard uncommitted workspace changes."""

    result = subprocess.run(
        ["git", "reset", "--hard", "HEAD"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
    )

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
import ast


def analyze_python_file(file_path: str) -> dict:
    """Analyze the structure of a Python file."""

    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
        }

    try:
        source = path.read_text(
            encoding="utf-8"
        )

        tree = ast.parse(source)

        functions = []

        for node in ast.walk(tree):

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):

                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "async": isinstance(
                        node,
                        ast.AsyncFunctionDef,
                    ),
                })

        return {
            "success": True,
            "file": file_path,
            "functions": functions,
        }

    except SyntaxError as error:

        return {
            "success": False,
            "error": (
                f"Python syntax error: {error}"
            ),
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
        }
def patch_file(
    file_path: str,
    old_text: str,
    new_text: str,
) -> dict:
    """
    Safely replace an exact section of a file.

    The replacement only happens when old_text occurs
    exactly once.
    """

    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
        }

    try:
        content = path.read_text(
            encoding="utf-8"
        )

        occurrences = content.count(old_text)

        if occurrences == 0:
            return {
                "success": False,
                "error": (
                    "The requested section was not found "
                    "in the file."
                ),
            }

        if occurrences > 1:
            return {
                "success": False,
                "error": (
                    "The requested section occurs "
                    f"{occurrences} times. "
                    "Refusing to guess which one to modify."
                ),
            }

        updated = content.replace(
            old_text,
            new_text,
            1,
        )

        path.write_text(
            updated,
            encoding="utf-8",
        )

        return {
            "success": True,
            "file": file_path,
            "message": "Patch applied successfully.",
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
        }
def search_project(query: str) -> dict:
    """Search the indexed project structure."""

    try:

        results = search_index(query)

        return {
            "success": True,
            "query": query,
            "results": results[:20],
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
        }
def read_project_memory() -> dict:
    """Read persistent project memory."""

    path = Path("memory/project.md")

    if not path.exists():
        return {
            "success": True,
            "content": "",
        }

    try:
        return {
            "success": True,
            "content": path.read_text(
                encoding="utf-8"
            ),
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }
def update_project_memory(
    section: str,
    content: str,
) -> dict:
    """Update one section of project memory."""

    path = Path("memory/project.md")

    try:

        if path.exists():
            existing = path.read_text(
                encoding="utf-8"
            )
        else:
            existing = "# Project Memory\n"

        heading = f"## {section}"

        lines = existing.splitlines()

        start = None

        for i, line in enumerate(lines):
            if line.strip() == heading:
                start = i
                break

        if start is not None:

            end = len(lines)

            for i in range(
                start + 1,
                len(lines),
            ):
                if lines[i].startswith("## "):
                    end = i
                    break

            lines[start:end] = [
                heading,
                "",
                content.strip(),
                "",
            ]

            updated = "\n".join(lines)

        else:

            updated = (
                existing.rstrip()
                + "\n\n"
                + heading
                + "\n\n"
                + content.strip()
                + "\n"
            )

        path.write_text(
            updated,
            encoding="utf-8",
        )

        return {
            "success": True,
            "message": (
                f"Memory section '{section}' updated."
            ),
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error),
        }
