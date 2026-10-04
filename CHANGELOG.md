# Changelog

All notable changes to `python_local_ai` are documented here. The project
follows [Semantic Versioning](https://semver.org); while it is `0.x`, minor
versions may contain breaking changes.

## 0.1.0

First release: PyO3 bindings over `rust_local_ai` 0.2.

### Added
- `detect()`, `LocalAiModel`, `LocalAiSession` and `ResponseStream`, with
  awaitable methods and `_sync` twins that release the GIL while they wait.
- Backends from `rust_local_ai` 0.2: Apple Foundation Models on macOS and
  Windows Phi Silica (both beta), Ubuntu inference snaps on Linux, and any
  local OpenAI-compatible server through `openai_compatible()`.
  `inference_snap()` picks a specific snap; `prepare()` is the only call that
  may start a model download.
- Streams that work with both `async for` and `for`; sessions as `async with`
  and `with` context managers.
- `response_format` for JSON and JSON-schema output where the backend
  supports it.
- A typed `LocalAiError` hierarchy with stable `code` values.
- `python_local_ai.testing.FakeBackend`, the deterministic backend from Rust.
- abi3 wheels for CPython 3.9+ on Linux (glibc and musl), macOS and Windows,
  plus an sdist.
