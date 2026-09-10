import ast
import os
import subprocess
from pathlib import Path

from config import PROJECT_WORKSPACE
from agent.indexer import search_index

WORKSPACE = Path(PROJECT_WORKSPACE).resolve()


def _workspace_path(file_path: str) -> Path:
    if not file_path:
        file_path = "."
    path = Path(file_path)
    # Older model tool calls used "workspace" for the project root.
    if not path.is_absolute() and path.as_posix().lower() == "workspace":
        path = Path(".")
    if path.is_absolute():
        return path.resolve()
    target = (WORKSPACE / path).resolve()
    try:
        target.relative_to(WORKSPACE)
    except ValueError:
        raise PermissionError(
            f"Access denied: {file_path} is outside the configured project."
        )
    return target


def safe_path(file_path: str) -> Path:
    """Ensure the path stays inside the workspace."""

    path = _workspace_path(file_path)

    try:
        path.relative_to(WORKSPACE)
    except ValueError:
        raise PermissionError(f"Access denied: {file_path} is outside the workspace.")

    return path


def list_files(directory: str = ".") -> list[str]:
    """List files recursively."""

    path = _workspace_path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    return [
        file.relative_to(WORKSPACE).as_posix()
        for file in path.rglob("*")
        if file.is_file()
    ]


def scan_files(
    directory: str = ".",
    recursive: bool = True,
    extensions: list[str] | None = None,
    exclude_directories: list[str] | None = None,
) -> dict:
    """Scan the target project filesystem without using an LLM."""

    root = _workspace_path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    normalized_extensions = None
    if extensions:
        normalized_extensions = {
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in extensions
        }

    default_exclusions = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "instance",
        "site-packages",
        ".tox",
        "dist",
        "build",
    }
    exclusions = default_exclusions | {
        name.lower() for name in (exclude_directories or [])
    }
    matched_files = []
    errors = []

    def include_file(path):
        if normalized_extensions and path.suffix.lower() not in normalized_extensions:
            return False
        return True

    if recursive:
        for current, directories, filenames in os.walk(
            root, onerror=lambda error: errors.append(str(error))
        ):
            directories[:] = [
                name for name in directories if name.lower() not in exclusions
            ]
            for filename in filenames:
                path = Path(current) / filename
                if include_file(path):
                    matched_files.append(path.relative_to(WORKSPACE).as_posix())
    else:
        try:
            for path in root.iterdir():
                if path.is_file() and include_file(path):
                    matched_files.append(path.relative_to(WORKSPACE).as_posix())
        except OSError as error:
            errors.append(str(error))

    matched_files.sort()
    grouped = {}
    for relative in matched_files:
        directory_name = str(Path(relative).parent).replace("\\", "/")
        grouped.setdefault("." if directory_name == "." else directory_name, []).append(
            relative
        )

    return {
        "target_root": str(WORKSPACE),
        "scan_directory": str(root),
        "recursive": recursive,
        "extensions": sorted(normalized_extensions) if normalized_extensions else [],
        "excluded_directories": sorted(exclusions),
        "matched_files": matched_files,
        "grouped_files": grouped,
        "total_count": len(matched_files),
        "errors": errors,
    }


def inspect_project() -> dict:
    """Read supported project files and return a concrete source inventory."""

    supported_extensions = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".html",
        ".css",
        ".sql",
    }
    ignored_directories = {".git", ".venv", "venv", "__pycache__", "node_modules"}
    scan = scan_files(
        directory=".",
        recursive=True,
        extensions=sorted(supported_extensions),
        exclude_directories=sorted(ignored_directories),
    )
    files = []

    for relative in scan["matched_files"]:
        path = _workspace_path(relative)
        entry = {"file": relative, "extension": path.suffix.lower()}

        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            entry["error"] = str(error)
            files.append(entry)
            continue

        entry["lines"] = len(source.splitlines())
        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(source)
                entry["imports"] = [
                    node.names[0].name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import) and node.names
                ]
                entry["classes"] = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ]
                entry["functions"] = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
            except SyntaxError as error:
                entry["syntax_error"] = str(error)

        files.append(entry)

    return {
        "workspace": str(WORKSPACE),
        "files_inspected": len(files),
        "files": files,
        "scan": scan,
    }


def verify_project_inventory(inventory: dict) -> dict:
    """Verify that an inspection result still describes the active workspace."""

    if not isinstance(inventory, dict):
        return {"success": False, "error": "Inspection result is not structured data."}

    expected_root = str(WORKSPACE)
    reported_root = str(inventory.get("workspace", ""))
    entries = inventory.get("files", [])
    missing = []

    if reported_root != expected_root:
        return {
            "success": False,
            "error": f"Inspection root changed: {reported_root} != {expected_root}",
        }

    for entry in entries:
        relative = entry.get("file") if isinstance(entry, dict) else None
        if not relative or not _workspace_path(relative).is_file():
            missing.append(relative or "<missing path>")

    if missing:
        return {
            "success": False,
            "error": f"{len(missing)} inspected files are no longer present.",
            "missing": missing,
        }

    return {
        "success": True,
        "workspace": expected_root,
        "files_verified": len(entries),
    }


def verify_file_scan(scan: dict) -> dict:
    """Verify a deterministic scan against the current target root."""

    if not isinstance(scan, dict):
        return {"success": False, "error": "Scan result is not structured data."}

    expected_root = str(WORKSPACE)
    if scan.get("target_root") != expected_root:
        return {
            "success": False,
            "error": "Scan target root does not match the configured root.",
        }

    errors = scan.get("errors", [])
    matched_files = scan.get("matched_files", [])
    missing = [
        relative
        for relative in matched_files
        if not _workspace_path(relative).is_file()
    ]
    if errors or missing:
        return {
            "success": False,
            "error": "Filesystem scan verification failed.",
            "scan_errors": errors,
            "missing_files": missing,
        }

    return {
        "success": True,
        "target_root": expected_root,
        "files_verified": len(matched_files),
    }


def read_file(file_path: str) -> str:
    """Read a text file."""

    path = safe_path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    return path.read_text(encoding="utf-8")


def search_code(directory: str, query: str) -> list[dict]:
    """Search text files for a query."""

    path = _workspace_path(directory)

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    results = []

    for file in path.rglob("*"):
        if not file.is_file():
            continue

        try:
            content = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1,
        ):
            if query.lower() in line.lower():
                results.append(
                    {
                        "file": str(file),
                        "line": line_number,
                        "content": line.strip(),
                    }
                )

    return results


def create_file(
    file_path: str,
    content: str,
) -> str:
    """Create a new file."""

    path = safe_path(file_path)

    if path.exists():
        raise FileExistsError(f"File already exists: {file_path}")

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
        raise FileNotFoundError(f"File does not exist: {file_path}")

    content = path.read_text(encoding="utf-8")

    if old_text not in content:
        raise ValueError(f"The requested old_text was not found in {file_path}")

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

    project_root = WORKSPACE

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


def git_diff() -> dict:
    """Return the current unstaged and staged Git diff."""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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
            "error": ("Workspace is not a Git repository.\n" + status.stderr),
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
        source = path.read_text(encoding="utf-8")

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
                functions.append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "async": isinstance(
                            node,
                            ast.AsyncFunctionDef,
                        ),
                    }
                )

        return {
            "success": True,
            "file": file_path,
            "functions": functions,
        }

    except SyntaxError as error:
        return {
            "success": False,
            "error": (f"Python syntax error: {error}"),
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
        content = path.read_text(encoding="utf-8")

        occurrences = content.count(old_text)

        if occurrences == 0:
            return {
                "success": False,
                "error": ("The requested section was not found in the file."),
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
        results = search_index(query, WORKSPACE)

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
            "content": path.read_text(encoding="utf-8"),
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
            existing = path.read_text(encoding="utf-8")
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
                existing.rstrip() + "\n\n" + heading + "\n\n" + content.strip() + "\n"
            )

        path.write_text(
            updated,
            encoding="utf-8",
        )

        return {
            "success": True,
            "message": (f"Memory section '{section}' updated."),
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
        }


def edit_file(
    file_path: str,
    old_text: str,
    new_text: str,
) -> dict:

    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "action": "edit_file",
        }

    try:
        content = path.read_text(encoding="utf-8")

        occurrences = content.count(old_text)

        if occurrences == 0:
            return {
                "success": False,
                "error": (f"The requested old_text was not found in {file_path}."),
                "hint": (
                    "Read the file again and use an exact "
                    "section from the actual file contents."
                ),
                "action": "edit_file",
                "file_path": file_path,
            }

        if occurrences > 1:
            return {
                "success": False,
                "error": (
                    f"The requested old_text occurs {occurrences} times in {file_path}."
                ),
                "hint": ("Use a larger, unique section of the file or use patch_file."),
                "action": "edit_file",
                "file_path": file_path,
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
            "action": "edit_file",
            "file_path": file_path,
            "message": "File edited successfully.",
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error),
            "action": "edit_file",
            "file_path": file_path,
        }
