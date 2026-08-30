from enum import Enum

from agent.tools import (
    git_status,
    git_checkpoint,
    git_rollback,
)
from config import SKIP_GIT_CHECKPOINT


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
    def __init__(self, state, reporter):
        self.state = state
        self.reporter = reporter
        self.phase = TaskPhase.PLANNING

    def set_phase(self, phase):

        # Don't print the same phase twice.
        if self.phase == phase:
            return

        self.phase = phase

        if self.reporter:
            self.reporter.phase(f"{phase.value.upper()}")

    def prepare_workspace(self):

        if SKIP_GIT_CHECKPOINT:
            self.reporter.debug("[Git] Checkpoint skipped by configuration.")
            self.state.checkpoint_created = False
            return True

        self.set_phase(TaskPhase.CHECKPOINT)

        status = git_status()

        if not status.get("success"):
            self.reporter.error("[Git] Workspace is not a Git repository.")
            return False

        checkpoint = git_checkpoint("Agent pre-task checkpoint")

        if checkpoint.get("success"):
            self.state.checkpoint_created = True
            self.reporter.debug("[Git] Checkpoint ready.")
            return True

        self.reporter.debug("[Git] No checkpoint available.")
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

        self.reporter.debug(f"[Repair] Attempt {self.state.attempts}/{self.state.max_attempts}")

    def can_repair(self):
        return self.state.can_retry()

    def rollback(self):

        if not self.state.checkpoint_created:
            self.reporter.debug("[Git] No checkpoint available.")
            return False

        self.set_phase(TaskPhase.ROLLBACK)

        result = git_rollback()

        if result.get("success"):
            self.reporter.debug("[Git] Rollback successful.")
            return True

        self.reporter.error("[Git] Rollback failed.")

        return False

    def complete(self):

        self.state.success = True
        self.state.completed = True

        self.set_phase(TaskPhase.COMPLETED)

    def fail(self):

        self.state.success = False
        self.state.completed = True

        self.set_phase(TaskPhase.FAILED)
