# agent/reporting.py

class Reporter:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def debug(self, *args, **kwargs):
        """Internal reasoning, tool calls, reflections – only shown in verbose mode."""
        if self.verbose:
            print(*args, **kwargs)

    def phase(self, message):
        """Phase changes – optionally shown to the user as progress."""
        # Always show phases, or show only if verbose
        if self.verbose:
            print(f"[PHASE] {message}")
        # If you want a minimal progress indicator for the user:
        # else: print(".", end="", flush=True)

    def user(self, message):
        """Final output or user‑facing messages – always shown."""
        print(message)

    def error(self, message):
        """Errors – always shown."""
        print(f"[ERROR] {message}")
