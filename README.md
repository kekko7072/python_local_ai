# python_local_ai

OS-native, on-device AI for Python. It is a thin [PyO3](https://pyo3.rs)
binding over [`rust_local_ai`](https://github.com/kekko7072/rust_local_ai), so
every backend, capability check and safety rule is implemented once in Rust.
Part of [Universal Local AI](https://github.com/kekko7072/universal_local_ai)
([vezz.io](https://vezz.io)).

> Early `0.x`, **not yet on PyPI**. What a backend can do depends entirely on
> the bound `rust_local_ai` revision. Today that means Apple Foundation Models
> (beta) on macOS and an explicit `unavailable` everywhere else. This package
> never downloads a model or installs a provider.

## Install (from source for now)

You need Rust stable and Python 3.9+. On macOS the Swift toolchain from Xcode
builds the Apple bridge.

```sh
pip install maturin
maturin develop            # into the active virtualenv
# or: maturin build --release && pip install target/wheels/*.whl
```

Each wheel is a single `abi3` build that works on every CPython from 3.9
onwards.

## Quick start (blocking)

```python
import python_local_ai as lai

model = lai.detect()                     # never raises, never downloads
availability = model.availability_sync()
if not availability:
    print("unavailable:", availability.reason, availability.detail)
else:
    with model.open_session_sync("You are a helpful assistant.") as session:
        print(session.generate_sync("Explain reinforcement learning simply.").text)
```

## asyncio

Every I/O method is awaitable. Each one also has a `_sync` twin that releases
the GIL while it waits, so other threads keep running.

```python
import asyncio
import python_local_ai as lai

async def main():
    model = lai.detect()
    if not await model.availability():
        return
    async with await model.open_session("Be brief.") as session:
        reply = await session.generate("Why does on-device AI matter?", temperature=0.3)
        print(reply.text, reply.output_tokens)

        if session.capabilities.streaming:
            async for chunk in session.stream("Now say it as a haiku."):
                print(chunk, end="", flush=True)

asyncio.run(main())
```

`session.stream(...)` supports both `async for` and plain `for`. The
generation starts on the first iteration. Call `stream.close()` to stop early
and free the session for the next turn.

## API

| Rust (`rust_local_ai`) | Python |
|---|---|
| `detect().await` | `detect()` |
| `model.availability()` / `capabilities()` | `await model.availability()` / `model.availability_sync()`, same for `capabilities` |
| `model.open_session(instructions)` | `await model.open_session(instructions)` / `open_session_sync` |
| `session.add_query_chunk` + `generate(config)` | `await session.generate(prompt, *, max_output_tokens, temperature, top_p, seed, response_format)` |
| `session.generate_stream(config)` | `session.stream(prompt, ...)`, a sync or async iterator of `str` |
| `cancel`, `count_tokens`, `close` | same names, plus `_sync` twins, `async with` / `with` |

- `response_format` takes `"text"` (the default), `"json"`, or a JSON-schema
  `dict`. Anything other than text requires `capabilities.structured_output`.
- `Availability` is truthy only when a session can be opened. Its `reason` is
  one of `unsupported_operating_system`, `unsupported_os_version`,
  `unsupported_hardware`, `system_feature_disabled`, `model_not_ready`,
  `provider_not_installed` or `backend_failure`.
- One generation runs per session at a time. A rejected call never adds its
  prompt to the conversation.
- The package is fully typed (`py.typed`, with stubs checked against the
  compiled module by `stubtest` in CI).

## Errors

Every error subclasses `LocalAiError` and carries a stable `code`:

| Exception | `code` | Extra attribute |
|---|---|---|
| `UnavailableError` | `unavailable` | `reason` |
| `UnsupportedCapabilityError` | `unsupported` | `capability` |
| `SessionBusyError` | `busy` | |
| `SessionClosedError` / `ModelClosedError` | `session_closed` / `model_closed` | |
| `InvalidConfigError` (also a `ValueError`) | `invalid_config` | |
| `BackendError` | `backend` | `backend` (the backend kind) |
| `GenerationCancelledError` | `cancelled` | |

`GenerationCancelledError` means `session.cancel()` stopped the generation.
It is deliberately distinct from `asyncio.CancelledError`.

## Testing without OS AI

`python_local_ai.testing.FakeBackend` is the deterministic backend from Rust's
`testing` module, so your tests exercise the same session rules as production:

```python
from python_local_ai.testing import FakeBackend

fake = FakeBackend()
fake.push_response("hello", output_tokens=1)
fake.set_stream_chunks(["a", "b"])
session = fake.model().open_session_sync()
assert session.generate_sync("hi").text == "hello"
assert list(session.stream("go")) == ["a", "b"]
```

It can also simulate unavailability (`set_availability`), backend errors
(`push_backend_error`), close failures (`fail_next_closes`), and a generation
that blocks (`set_generation_blocked`) so you can test busy and cancel paths.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest mypy typing_extensions
maturin develop
pytest                                    # unit tests and doctests
python -m mypy.stubtest python_local_ai._native
cargo fmt --check && cargo clippy --all-targets -- -D warnings
```

CI runs these steps on Linux, macOS and Windows with Python 3.9 and 3.13, runs
the examples, and builds abi3 wheels and an sdist as artifacts. CI never
publishes.

The Rust core is pinned to a specific `rust_local_ai` git revision in
`Cargo.toml` (it is also exposed as
`python_local_ai._native.RUST_LOCAL_AI_REVISION`). Bump it deliberately
whenever the core changes.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), the
same as `rust_local_ai`.
