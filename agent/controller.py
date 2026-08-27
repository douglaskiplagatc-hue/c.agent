from enum import Enum

from agent.tools import (
    git_status,
    git_checkpoint,
    git_rollback,
)


class TaskPhase(Enum):
    PLANNING = "planning"
    MEMORY = "memory"
    CHECKPOINT = "checkpoint"
    INSPECTING = "inspecting"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    DIAGNOSING = "diagnosing"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


class AgentController:

    def __init__(self, state):
        self.state = state
        self.phase = TaskPhase.PLANNING

    def set_phase(self, phase):

        # Don't print the same phase twice.
        if self.phase == phase:
            return

        self.phase = phase

        print(
            f"\n[PHASE → {phase.value.upper()}]"
        )

    def prepare_workspace(self):

        # Read-only tasks do not need a Git checkpoint.
        if not self.state.requires_modification:
            return True

        self.set_phase(TaskPhase.CHECKPOINT)

        status = git_status()

        if not status.get("success"):
            print(
                "[Git] Workspace is not a Git repository."
            )
            return False

        checkpoint = git_checkpoint(
            "Agent pre-task checkpoint"
        )

        if checkpoint.get("success"):
            self.state.checkpoint_created = True

            print(
                "[Git] Checkpoint ready."
            )

            return True

        print(
            "[Git] No new checkpoint needed."
        )

        self.state.checkpoint_created = True

        return True

    def begin_inspection(self):
        self.set_phase(TaskPhase.INSPECTING)

    def begin_implementation(self):
        self.set_phase(TaskPhase.IMPLEMENTING)

    def begin_verification(self):
        self.set_phase(TaskPhase.VERIFYING)

    def begin_diagnosis(self):
        self.set_phase(TaskPhase.DIAGNOSING)

    def begin_repair(self):

        self.state.attempts += 1

        self.set_phase(TaskPhase.REPAIRING)

        print(
            f"[Repair] Attempt "
            f"{self.state.attempts}/"
            f"{self.state.max_attempts}"
        )

    def can_repair(self):
        return self.state.can_retry()

    def rollback(self):

        if not self.state.checkpoint_created:
            print(
                "[Git] No checkpoint available."
            )
            return False

        self.set_phase(TaskPhase.ROLLBACK)

        result = git_rollback()

        if result.get("success"):
            print(
                "[Git] Rollback successful."
            )
            return True

        print(
            "[Git] Rollback failed."
        )

        return False

    def complete(self):

        self.state.success = True
        self.state.completed = True

        self.set_phase(TaskPhase.COMPLETED)

    def fail(self):

        self.state.success = False
        self.state.completed = True

        self.set_phase(TaskPhase.FAILED)
