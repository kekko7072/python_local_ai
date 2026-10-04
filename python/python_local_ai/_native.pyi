from collections.abc import AsyncIterator, Iterator
from types import TracebackType
from typing import Any, Awaitable, Literal, Mapping, Optional, Union

from typing_extensions import Self, final

_BackendKind = Literal[
    "apple_foundation_models",
    "windows_ai",
    "ubuntu_inference_snap",
    "linux_provider",
    "fake",
    "unsupported",
    "unknown",
]
_AvailabilityReason = Literal[
    "unsupported_operating_system",
    "unsupported_os_version",
    "unsupported_hardware",
    "system_feature_disabled",
    "model_not_ready",
    "provider_not_installed",
    "backend_failure",
    "unknown",
]
_ResponseFormat = Union[Literal["text", "json"], Mapping[str, Any], None]

__all__ = [
    "detect",
    "LocalAiModel",
    "LocalAiSession",
    "ResponseStream",
    "Availability",
    "Capabilities",
    "BackendInfo",
    "AiResponse",
    "FakeBackend",
    "RUST_LOCAL_AI_REVISION",
]

RUST_LOCAL_AI_REVISION: str

@final
class BackendInfo:
    @property
    def kind(self) -> _BackendKind: ...
    @property
    def name(self) -> str: ...
    @property
    def system_managed_model(self) -> bool: ...

@final
class Availability:
    @property
    def available(self) -> bool: ...
    @property
    def reason(self) -> Optional[_AvailabilityReason]: ...
    @property
    def detail(self) -> Optional[str]: ...
    def __bool__(self) -> bool: ...

@final
class Capabilities:
    def __new__(
        cls,
        *,
        text_generation: bool = False,
        streaming: bool = False,
        structured_output: bool = False,
        tool_calling: bool = False,
        vision: bool = False,
        token_counting: bool = False,
        cancellation: bool = False,
        concurrent_sessions: bool = False,
        system_managed_model: bool = False,
    ) -> Capabilities: ...
    @property
    def text_generation(self) -> bool: ...
    @property
    def streaming(self) -> bool: ...
    @property
    def structured_output(self) -> bool: ...
    @property
    def tool_calling(self) -> bool: ...
    @property
    def vision(self) -> bool: ...
    @property
    def token_counting(self) -> bool: ...
    @property
    def cancellation(self) -> bool: ...
    @property
    def concurrent_sessions(self) -> bool: ...
    @property
    def system_managed_model(self) -> bool: ...

@final
class AiResponse:
    @property
    def text(self) -> str: ...
    @property
    def input_tokens(self) -> Optional[int]: ...
    @property
    def output_tokens(self) -> Optional[int]: ...
    @property
    def finish_reason(self) -> Optional[str]: ...

def detect() -> LocalAiModel: ...

@final
class LocalAiModel:
    @property
    def backend(self) -> BackendInfo: ...
    @property
    def closed(self) -> bool: ...
    def availability(self) -> Awaitable[Availability]: ...
    def availability_sync(self) -> Availability: ...
    def capabilities(self) -> Awaitable[Capabilities]: ...
    def capabilities_sync(self) -> Capabilities: ...
    def open_session(self, instructions: Optional[str] = None) -> Awaitable[LocalAiSession]: ...
    def open_session_sync(self, instructions: Optional[str] = None) -> LocalAiSession: ...
    def close(self) -> None: ...

@final
class LocalAiSession:
    @property
    def capabilities(self) -> Capabilities: ...
    @property
    def closed(self) -> bool: ...
    def add_query_chunk(self, chunk: str) -> Awaitable[None]: ...
    def add_query_chunk_sync(self, chunk: str) -> None: ...
    def generate(
        self,
        prompt: Optional[str] = None,
        *,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: _ResponseFormat = None,
    ) -> Awaitable[AiResponse]: ...
    def generate_sync(
        self,
        prompt: Optional[str] = None,
        *,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: _ResponseFormat = None,
    ) -> AiResponse: ...
    def stream(
        self,
        prompt: Optional[str] = None,
        *,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        response_format: _ResponseFormat = None,
    ) -> ResponseStream: ...
    def cancel(self) -> Awaitable[None]: ...
    def cancel_sync(self) -> None: ...
    def count_tokens(self, text: str) -> Awaitable[int]: ...
    def count_tokens_sync(self, text: str) -> int: ...
    def close(self) -> Awaitable[None]: ...
    def close_sync(self) -> None: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool: ...

@final
class ResponseStream(AsyncIterator[str], Iterator[str]):
    def __aiter__(self) -> Self: ...
    def __anext__(self) -> Awaitable[str]: ...
    def __iter__(self) -> Self: ...
    def __next__(self) -> str: ...
    def close(self) -> None: ...

@final
class FakeBackend:
    def __new__(cls, capabilities: Optional[Capabilities] = None) -> FakeBackend: ...
    def model(self) -> LocalAiModel: ...
    def set_availability(
        self,
        available: bool,
        reason: Optional[_AvailabilityReason] = None,
        detail: Optional[str] = None,
    ) -> None: ...
    def push_response(
        self,
        text: str,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
    ) -> None: ...
    def push_backend_error(self, message: str) -> None: ...
    def set_stream_chunks(self, chunks: list[str]) -> None: ...
    def fail_next_closes(self, count: int) -> None: ...
    def set_generation_blocked(self, blocked: bool) -> None: ...
