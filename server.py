#!/usr/bin/env python3
import json
import os
import subprocess
import shutil
import io
import contextlib
import json
import queue
import threading
from pathlib import Path
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    Response,
    stream_with_context,
)
from flask_cors import CORS

import config
from agent import brain, tools as agent_tools

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

WORKSPACE = Path(config.PROJECT_WORKSPACE).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)


def set_workspace(path):
    """Update all runtime components that operate on the selected project."""
    global WORKSPACE
    selected = Path(path).expanduser().resolve()
    if not selected.is_dir():
        raise ValueError("Selected workspace is not a directory")
    WORKSPACE = selected
    config.PROJECT_WORKSPACE = str(selected)
    agent_tools.WORKSPACE = selected
    brain.PROJECT_WORKSPACE = str(selected)


# ------------------- Helpers -------------------
def get_file_list():
    try:
        return agent_tools.list_files(".")
    except Exception:
        return []


def get_git_info():
    try:
        status = agent_tools.git_status()
        if not status.get("success"):
            return {"branch": "main", "changes": 0}
        output = status.get("stdout", "").strip()
        changes = len([line for line in output.splitlines() if line.strip()])
        try:
            branch = (
                subprocess.check_output(
                    ["git", "branch", "--show-current"],
                    cwd=WORKSPACE,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                or "main"
            )
        except:
            branch = "main"
        return {"branch": branch, "changes": changes}
    except:
        return {"branch": "main", "changes": 0}


def _safe_path(path_str):
    full = (WORKSPACE / path_str).resolve()
    try:
        full.relative_to(WORKSPACE)
    except ValueError:
        raise PermissionError(f"Access denied: {path_str} outside workspace")
    return full


# ------------------- Routes -------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "status": "active",
            "workspace": str(WORKSPACE),
        }
    )


@app.route("/api/project")
def project_info():
    try:
        files = get_file_list()
        git = get_git_info()
        return jsonify(
            {
                "workspace": str(WORKSPACE),
                "file_count": len(files),
                "files": files,
                "git": git,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/workspace/select", methods=["POST"])
def select_workspace():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Choose AetherIDE workspace",
            initialdir=str(WORKSPACE),
        )
        root.destroy()
        if not selected:
            return jsonify({"ok": True, "cancelled": True, "workspace": str(WORKSPACE)})
        set_workspace(selected)
        return jsonify({"ok": True, "workspace": str(WORKSPACE)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/file")
def read_file():
    path = request.args.get("path")
    if not path:
        return jsonify({"ok": False, "error": "Missing path"}), 400
    try:
        content = agent_tools.read_file(path)
        return jsonify({"ok": True, "content": content})
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "File not found"}), 404
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/file", methods=["POST"])
def write_file():
    data = request.get_json()
    path = data.get("path")
    content = data.get("content", "")
    if not path:
        return jsonify({"ok": False, "error": "Missing path"}), 400
    try:
        full = _safe_path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return jsonify({"ok": True, "message": f"File {path} written"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/file", methods=["DELETE"])
def delete_file():
    path = request.args.get("path")
    if not path:
        return jsonify({"ok": False, "error": "Missing path"}), 400
    try:
        full = _safe_path(path)
        if full.is_dir():
            shutil.rmtree(full)
        else:
            full.unlink()
        return jsonify({"ok": True, "message": f"Deleted {path}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/file/rename", methods=["POST"])
def rename_file():
    data = request.get_json()
    old_path = data.get("old_path")
    new_path = data.get("new_path")
    if not old_path or not new_path:
        return jsonify({"ok": False, "error": "Missing old_path or new_path"}), 400
    try:
        old_full = _safe_path(old_path)
        new_full = _safe_path(new_path)
        old_full.rename(new_full)
        return jsonify({"ok": True, "message": f"Renamed {old_path} to {new_path}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/directory", methods=["POST"])
def create_directory():
    data = request.get_json()
    path = data.get("path")
    if not path:
        return jsonify({"ok": False, "error": "Missing path"}), 400
    try:
        full = _safe_path(path)
        full.mkdir(parents=True, exist_ok=True)
        return jsonify({"ok": True, "message": f"Directory {path} created"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/terminal", methods=["POST"])
def run_terminal_command():
    data = request.get_json()
    command = data.get("command", "")
    timeout = data.get("timeout", 30)
    if not command:
        return jsonify({"ok": False, "error": "Missing command"}), 400
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return jsonify(
            {
                "ok": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        )
    except subprocess.TimeoutExpired:
        return jsonify(
            {"ok": False, "error": f"Command timed out after {timeout}s"}
        ), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/search")
def search_project():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"ok": False, "error": "Missing query"}), 400
    try:
        # Use the existing search_project tool
        result = agent_tools.search_project(query)
        if result.get("success"):
            return jsonify({"ok": True, "results": result.get("results", [])})
        else:
            return jsonify(
                {"ok": False, "error": result.get("error", "Search failed")}
            ), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/git/status")
def git_status_api():
    try:
        result = agent_tools.git_status()
        branch = (
            subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=WORKSPACE,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            or "main"
        )
        changes = len(
            [line for line in result.get("stdout", "").splitlines() if line.strip()]
        )
        return jsonify(
            {
                "ok": True,
                "branch": branch,
                "changes": changes,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/git/diff")
def git_diff_api():
    path = request.args.get("path")
    try:
        cmd = ["git", "diff", "HEAD"]
        if path:
            cmd.append("--")
            cmd.append(path)
        result = subprocess.run(
            cmd,
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return jsonify(
            {
                "ok": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/git/commit", methods=["POST"])
def git_commit():
    data = request.get_json()
    message = data.get("message", "Agent commit")
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=WORKSPACE,
            capture_output=True,
            check=True,
        )
        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return jsonify({"ok": False, "error": result.stderr}), 500
        return jsonify({"ok": True, "stdout": result.stdout, "stderr": result.stderr})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/git/reset", methods=["POST"])
def git_reset():
    data = request.get_json()
    hard = data.get("hard", True)
    try:
        if hard:
            result = subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(
                ["git", "reset", "HEAD"],
                cwd=WORKSPACE,
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            return jsonify({"ok": False, "error": result.stderr}), 500
        return jsonify({"ok": True, "stdout": result.stdout, "stderr": result.stderr})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/task", methods=["POST"])
def run_task():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "error": "Empty message"}), 400
    trace_buffer = io.StringIO()
    try:
        with (
            contextlib.redirect_stdout(trace_buffer),
            contextlib.redirect_stderr(trace_buffer),
        ):
            result = brain.ask_model(message, verbose=True)
        return jsonify(
            {"ok": True, "result": str(result), "trace": trace_buffer.getvalue()}
        )
    except Exception as e:
        trace = trace_buffer.getvalue()
        return jsonify({"ok": False, "error": str(e), "trace": trace}), 500


@app.route("/api/task/stream", methods=["POST"])
def stream_task():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    events = queue.Queue()
    complete = object()

    class QueueWriter:
        def write(self, value):
            if value:
                events.put({"type": "log", "value": value})
            return len(value)

        def flush(self):
            return None

    def run_agent():
        writer = QueueWriter()
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result = brain.ask_model(message, verbose=True)
            events.put({"type": "result", "value": str(result)})
        except Exception as error:
            events.put({"type": "error", "value": str(error)})
        finally:
            events.put(complete)

    threading.Thread(target=run_agent, daemon=True).start()

    @stream_with_context
    def event_stream():
        while True:
            event = events.get()
            if event is complete:
                yield 'data: {"type":"done"}\n\n'
                break
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    print(f"AetherIDE server running on http://127.0.0.1:8000")
    print(f"📁 Workspace: {WORKSPACE}")
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
