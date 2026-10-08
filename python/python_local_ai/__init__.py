"""OS-native, on-device AI for Python, powered by rust_local_ai.

>>> import python_local_ai as lai
>>> model = lai.detect()
>>> if model.availability_sync():
...     with model.open_session_sync("Be brief.") as session:
...         print(session.generate_sync("What is on-device AI?").text)
"""

from ._errors import (
    BackendError,
    GenerationCancelledError,
    InvalidConfigError,
    LocalAiError,
    ModelClosedError,
    SessionBusyError,
    SessionClosedError,
    UnavailableError,
    UnsupportedCapabilityError,
)
from ._native import (
    AiResponse,
    Availability,
    BackendInfo,
    Capabilities,
    LocalAiModel,
    LocalAiSession,
    ResponseStream,
    detect,
    inference_snap,
    openai_compatible,
)

__version__ = "0.1.0"

__all__ = [
    "AiResponse",
    "Availability",
    "BackendError",
    "BackendInfo",
    "Capabilities",
    "GenerationCancelledError",
    "InvalidConfigError",
    "LocalAiError",
    "LocalAiModel",
    "LocalAiSession",
    "ModelClosedError",
    "ResponseStream",
    "SessionBusyError",
    "SessionClosedError",
    "UnavailableError",
    "UnsupportedCapabilityError",
    "detect",
    "inference_snap",
    "openai_compatible",
]
