"""OS-native, on-device AI for Python, powered by rust_local_ai.

>>> import python_local_ai as lai
>>> model = lai.detect()
>>> availability = model.availability_sync()
>>> isinstance(availability.available, bool)
True
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

__version__ = "0.1.1"

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
