from enum import Enum

from agent.tools import (
    git_status,
    git_checkpoint,
    git_rollback,
)


class TaskPhase(Enum):
    PLANNING = "planning"
    CHECKPOINT = "checkpoint"
    INSPECTING = "inspecting"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentController:

    def __init__(self, state):
        self.state = state
        self.phase = TaskPhase.PLANNING
        self.checkpoint_created = False

    def set_phase(self, phase):

        self.phase = phase

        print(
            f"\n[PHASE → {phase.value.upper()}]"
        )

    def prepare_workspace(self):

        self.set_phase(
            TaskPhase.CHECKPOINT
        )

        status = git_status()

        if not status.get("success", False):

            print(
                "[Git] Workspace is not a Git repository."
            )

            return False

        print(
            "[Git] Workspace status checked."
        )

        checkpoint = git_checkpoint(
            "Agent pre-task checkpoint"
        )

        if checkpoint.get("success", False):

            self.checkpoint_created = True

            print(
                "[Git] Checkpoint ready."
            )

            return True

        print(
            "[Git] Checkpoint could not be created."
        )

        return False

    def begin_inspection(self):

        self.set_phase(
            TaskPhase.INSPECTING
        )

    def begin_implementation(self):

        self.set_phase(
            TaskPhase.IMPLEMENTING
        )

    def begin_verification(self):

        self.set_phase(
            TaskPhase.VERIFYING
        )

    def begin_repair(self):

        self.set_phase(
            TaskPhase.REPAIRING
        )

        self.state.attempts += 1

    def can_repair(self):

        return (
            self.state.attempts
            < self.state.max_attempts
        )

    def rollback(self):

        if not self.checkpoint_created:

            print(
                "[Git] No agent checkpoint available."
            )

            return False

        print(
            "[Git] Rolling back workspace..."
        )

        result = git_rollback()

        if result.get("success", False):

            print(
                "[Git] Rollback successful."
            )

            return True

        print(
            "[Git] Rollback failed."
        )

        return False

    def complete(self):

        self.set_phase(
            TaskPhase.COMPLETED
        )

        self.state.completed = True

    def fail(self):

        self.set_phase(
            TaskPhase.FAILED
        )

        self.state.completed = True
