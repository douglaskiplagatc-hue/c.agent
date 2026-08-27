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

        self.errors = []
        self.verifications = []

        self.tool_history = []

        self.memory_loaded = False

        self.checkpoint_created = False

    def record_tool(
        self,
        tool_name,
        arguments,
        result,
    ):

        self.tool_history.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
        })

    def record_error(self, error):

        self.errors.append(str(error))

    def record_verification(
        self,
        description,
        success,
    ):

        self.verifications.append({
            "description": description,
            "success": success,
        })

    def record_inspection(self, file_path):

        if file_path not in self.inspected_files:
            self.inspected_files.append(file_path)

    def record_modification(self, file_path):

        if file_path not in self.modified_files:
            self.modified_files.append(file_path)

    def can_retry(self):

        return self.attempts < self.max_attempts

    def summary(self):

        return {
            "objective": self.objective,
            "success": self.success,
            "attempts": self.attempts,
            "inspected_files": self.inspected_files,
            "modified_files": self.modified_files,
            "errors": self.errors,
            "verifications": self.verifications,
        }
