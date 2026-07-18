"""
Verification finite state machine.

This is the heart of the structural defense. The state lives HERE, in Python,
never in the conversation. No amount of persuasive text can move it — only a
`verify_identity` call whose arguments actually match the patient record.

States
------
UNVERIFIED : start state. Disclosure tools refused. verify_identity allowed;
             a mismatch increments the failure counter.
VERIFIED   : disclosure tools allowed.
LOCKED     : reached after MAX_VERIFY_ATTEMPTS failures. Everything refused
             (terminal for the session).

Transitions
-----------
UNVERIFIED --verify PASS------------------> VERIFIED
UNVERIFIED --verify FAIL (< max attempts)-> UNVERIFIED   (counter += 1)
UNVERIFIED --verify FAIL (== max attempts)-> LOCKED
VERIFIED   --session end------------------> UNVERIFIED   (re-lock)
"""

from __future__ import annotations

from enum import Enum

import config


class State(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    LOCKED = "LOCKED"


class VerificationFSM:
    def __init__(self, max_attempts: int = config.MAX_VERIFY_ATTEMPTS) -> None:
        self.state = State.UNVERIFIED
        self.failed_attempts = 0
        self.max_attempts = max_attempts
        # Which patient (if any) the caller successfully verified as.
        self.verified_patient_name: str | None = None

    # -- queries ----------------------------------------------------------- #
    @property
    def is_verified(self) -> bool:
        return self.state is State.VERIFIED

    @property
    def is_locked(self) -> bool:
        return self.state is State.LOCKED

    def can_disclose(self) -> bool:
        return self.state is State.VERIFIED

    # -- transitions ------------------------------------------------------- #
    def record_verification(self, success: bool, patient_name: str | None = None) -> State:
        """
        Apply the outcome of a verify_identity attempt. Returns the new state.
        Attempts made while LOCKED (or already VERIFIED) do not change state.
        """
        if self.state is State.LOCKED:
            return self.state
        if self.state is State.VERIFIED:
            return self.state

        if success:
            self.state = State.VERIFIED
            self.verified_patient_name = patient_name
            return self.state

        # failure
        self.failed_attempts += 1
        if self.failed_attempts >= self.max_attempts:
            self.state = State.LOCKED
        return self.state

    def end_session(self) -> None:
        """Re-lock a verified session (used between benign 'calls')."""
        if self.state is State.VERIFIED:
            self.state = State.UNVERIFIED
            self.verified_patient_name = None

    def snapshot(self) -> dict:
        """Serializable view for transcript logging."""
        return {
            "state": self.state.value,
            "failed_attempts": self.failed_attempts,
            "verified_patient": self.verified_patient_name,
        }
