from pathlib import Path
import ast


IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
}


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".sql",
}


def build_index(root="workspace"):
    root = Path(root)

    index = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if any(
            part in IGNORED_DIRS
            for part in path.parts
        ):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative = path.relative_to(root)

        entry = {
            "file": str(relative),
            "extension": path.suffix.lower(),
        }

        if path.suffix.lower() == ".py":

            try:
                source = path.read_text(
                    encoding="utf-8"
                )

                tree = ast.parse(source)

                entry["functions"] = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    )
                ]

                entry["classes"] = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        ast.ClassDef,
                    )
                ]

            except Exception:
                entry["functions"] = []
                entry["classes"] = []

        index.append(entry)

    return index


def search_index(query, root="workspace"):
    query = query.lower()

    index = build_index(root)

    results = []

    for entry in index:

        score = 0

        filename = entry["file"].lower()

        if query in filename:
            score += 10

        for function in entry.get(
            "functions",
            [],
        ):

            if query in function.lower():
                score += 8

        for class_name in entry.get(
            "classes",
            [],
        ):

            if query in class_name.lower():
                score += 8

        if score > 0:

            result = dict(entry)
            result["score"] = score

            results.append(result)

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results
