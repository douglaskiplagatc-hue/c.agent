class TaskState:
    def __init__(self, objective):

        self.objective = objective
        self.plan = ""

        self.attempts = 0
        self.max_attempts = 5

        self.completed = False
        self.success = False

        self.modified_files = []
        self.inspected_files = []

        self.commands = []
        self.errors = []
        self.verifications = []
        self.llm_verification_count = 0

        self.tool_history = []
        self.failed_actions = []

        self.memory_loaded = False
        self.checkpoint_created = False
        self.requires_modification = False
        self.functional_verification_passed = False

    # -------------------------
    # FILE TRACKING
    # -------------------------

    def record_inspected_file(self, file_path):

        if file_path not in self.inspected_files:
            self.inspected_files.append(file_path)

    def record_inspection(self, file_path):

        self.record_inspected_file(file_path)

    def record_modified_file(self, file_path):

        if file_path not in self.modified_files:
            self.modified_files.append(file_path)

    def record_modification(self, file_path):

        self.record_modified_file(file_path)

    # -------------------------
    # COMMAND TRACKING
    # -------------------------

    def record_command(self, command):

        self.commands.append(command)

    # -------------------------
    # ERRORS
    # -------------------------

    def record_error(self, error):

        self.errors.append(str(error))

    # -------------------------
    # VERIFICATION
    # -------------------------

    def record_verification(self, description, success):

        self.verifications.append(
            {
                "description": description,
                "success": success,
            }
        )

    # -------------------------
    # TOOL HISTORY
    # -------------------------

    def record_tool(
        self,
        tool_name,
        arguments,
        result,
    ):

        self.tool_history.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )

    # -------------------------
    # FAILED ACTIONS
    # -------------------------

    def record_failed_action(
        self,
        tool_name,
        arguments,
        error,
    ):

        self.failed_actions.append(
            {
                "tool": tool_name,
                "arguments": arguments,
                "error": str(error),
            }
        )

    def recent_failures(self):

        return self.failed_actions[-5:]

    # -------------------------
    # REPAIR
    # -------------------------

    def can_retry(self):

        return self.attempts < self.max_attempts

    def repair_context(self):

        failures = self.recent_failures()

        if not failures:
            return ""

        lines = [
            "PREVIOUS FAILED ACTIONS:",
            "",
        ]

        for failure in failures:
            lines.append(f"Tool: {failure['tool']}")

            lines.append(f"Arguments: {failure['arguments']}")

            lines.append(f"Error: {failure['error']}")

            lines.append("")

        lines.extend(
            [
                "IMPORTANT:",
                "Do not repeat the failed action blindly.",
                "Use the failure as evidence.",
                "Inspect the relevant code before repairing.",
            ]
        )

        return "\n".join(lines)

    # -------------------------
    # SUMMARY
    # -------------------------

    def summary(self):

        return {
            "objective": self.objective,
            "success": self.success,
            "attempts": self.attempts,
            "inspected_files": self.inspected_files,
            "modified_files": self.modified_files,
            "commands": len(self.commands),
            "errors": self.errors,
            "verifications": self.verifications,
        }
