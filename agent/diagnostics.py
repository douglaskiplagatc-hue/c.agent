def diagnose_failure(result):

    if not result:
        return "Unknown failure."

    stderr = result.get("stderr", "")
    stdout = result.get("stdout", "")

    if stderr:
        return stderr[-6000:]

    if result.get("error"):
        return result["error"]

    if stdout:
        return stdout[-6000:]

    return "Command failed without diagnostic output."
