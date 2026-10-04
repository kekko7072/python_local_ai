"""Exception hierarchy shared by every python_local_ai entry point."""

from __future__ import annotations

from typing import Optional

__all__ = [
    "LocalAiError",
    "UnavailableError",
    "SessionBusyError",
    "SessionClosedError",
    "ModelClosedError",
    "UnsupportedCapabilityError",
    "InvalidConfigError",
    "BackendError",
    "GenerationCancelledError",
]


class LocalAiError(Exception):
    """Base class. ``code`` is a stable, machine-readable category."""

    code: str = "generation_failed"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnavailableError(LocalAiError):
    """No usable backend. ``reason`` matches :attr:`Availability.reason`."""

    code = "unavailable"

    def __init__(self, message: str, *, reason: Optional[str] = None) -> None:
        super().__init__(message)
        self.reason = reason


class SessionBusyError(LocalAiError):
    """The session already has a generation in progress."""

    code = "busy"


class SessionClosedError(LocalAiError):
    code = "session_closed"


class ModelClosedError(LocalAiError):
    code = "model_closed"


class UnsupportedCapabilityError(LocalAiError):
    """The active backend does not provide ``capability``."""

    code = "unsupported"

    def __init__(self, message: str, *, capability: Optional[str] = None) -> None:
        super().__init__(message)
        self.capability = capability


class InvalidConfigError(LocalAiError, ValueError):
    code = "invalid_config"


class BackendError(LocalAiError):
    """The native backend failed. ``backend`` is the :attr:`BackendInfo.kind`."""

    code = "backend"

    def __init__(self, message: str, *, backend: Optional[str] = None) -> None:
        super().__init__(message)
        self.backend = backend


class GenerationCancelledError(LocalAiError):
    """The generation was cancelled with :meth:`LocalAiSession.cancel`.

    Distinct from :class:`asyncio.CancelledError`, which cancels the awaiting task.
    """

    code = "cancelled"
