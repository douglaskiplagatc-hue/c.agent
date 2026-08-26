from dataclasses import dataclass, field


@dataclass
class TaskState:

    objective: str

    plan: str = ""

    inspected_files: list[str] = field(
        default_factory=list
    )

    modified_files: list[str] = field(
        default_factory=list
    )

    commands: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    completed: bool = False

    attempts: int = 0

    max_attempts: int = 8

    def record_error(self, error: str):
        self.errors.append(error)

    def record_command(self, command: str):
        self.commands.append(command)

    def record_modified_file(self, file_path: str):
        if file_path not in self.modified_files:
            self.modified_files.append(file_path)

    def record_inspected_file(self, file_path: str):
        if file_path not in self.inspected_files:
            self.inspected_files.append(file_path)
