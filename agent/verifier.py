import subprocess
from config import PROJECT_WORKSPACE


def run_verification(command):

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=PROJECT_WORKSPACE,
            capture_output=True,
            text=True,
            timeout=120,
        )

        return {
            "success": result.returncode == 0,
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "command": command,
            "error": "Verification timed out.",
        }

    except Exception as error:
        return {
            "success": False,
            "command": command,
            "error": str(error),
        }
