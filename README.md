# python_local_ai

OS-native, on-device AI for Python. It is a thin [PyO3](https://pyo3.rs)
binding over [`rust_local_ai`](https://github.com/kekko7072/rust_local_ai), so
every backend, capability check and safety rule is implemented once in Rust.
Part of [Universal Local AI](https://github.com/kekko7072/universal_local_ai)
([vezz.io](https://vezz.io)).

> Early `0.x`, built on `rust_local_ai` 0.2. What a backend can do depends
> entirely on that core: see the table below. This package
> never downloads a model or installs a provider.

## Install

```sh
pip install python_local_ai
```

Wheels are published for CPython 3.9+ (one `abi3` wheel per platform):

| Platform | Wheels | Backend |
|---|---|---|
| macOS 12+ (Apple silicon, Intel) | `arm64`, `x86_64` | Apple Foundation Models on macOS 26+ with Apple Intelligence; otherwise reports why it is unavailable |
| Linux (glibc and musl) | `x86_64`, `aarch64` | Ubuntu inference snaps (for example `qwen3`), if one is installed |
| Windows | `x64`, `arm64` | Phi Silica (beta): needs a Copilot+ PC or supported GPU, Windows 11 25H2+, and an app with package identity, so a plain `python.exe` reports `provider_not_installed` |
| Any | — | `openai_compatible()`: a local llama.cpp, Ollama, LM Studio or Foundry Local server |

"beta" means the Rust core builds and type-checks this backend in CI, but it
hasn't yet been run against a real model.

The macOS wheels are built with the macOS 26 SDK, and the release fails if
the Foundation Models bridge is missing. Other platforms fall back to the
sdist, which needs Rust stable (and Xcode on macOS).

### From source

```sh
pip install maturin
maturin develop            # into the active virtualenv
```

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

## Choosing a backend

```python
model = lai.detect()                                  # the OS-native backend
model = lai.openai_compatible("http://localhost:11434/v1")      # e.g. Ollama
model = lai.openai_compatible("http://127.0.0.1:8080/v1", "qwen3")  # pick a model
model = lai.inference_snap("qwen3")                   # a specific Ubuntu snap (Linux)
```

`openai_compatible` only accepts plain `http://` URLs on this machine
(`localhost`, `127.0.0.0/8`, `::1`), so prompts never leave it. Without a
model name it uses the first one the server lists.

`model.prepare()` (or `prepare_sync()`) explicitly asks the backend to get its
model ready. On Windows this may start a large Phi Silica download, so ask the
user first. No other call ever downloads anything.

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

The Rust core comes from crates.io (`rust_local_ai = "0.2.0"` in
`Cargo.toml`, locked in `Cargo.lock`). The resolved version is exposed as
`python_local_ai._native.RUST_LOCAL_AI_VERSION`. To move to a new core,
release `rust_local_ai` first, then bump it here and run the tests.

## Releasing

`.github/workflows/release.yml` releases from `main`. It builds every wheel
and the sdist, tests the installed wheels on Linux, macOS and Windows,
uploads them to PyPI through Trusted Publishing, and creates the GitHub
release `v<version>` with the `CHANGELOG.md` entry and every distribution
attached.

1. One-time setup: on pypi.org, add a *pending trusted publisher* for the
   project `python_local_ai` (owner `kekko7072`, repository
   `python_local_ai`, workflow `release.yml`, environment `pypi`). Do the
   same on test.pypi.org with the environment `testpypi` for dry runs. Then
   create the `pypi` and `testpypi` environments in the GitHub repository
   settings. Adding required reviewers to `pypi` is recommended: every
   release then waits for your approval.
2. Bump the version in `pyproject.toml`, `Cargo.toml` and
   `python/python_local_ai/__init__.py`, and add a `## <version>` entry to
   `CHANGELOG.md`. The workflow refuses mismatched versions.
3. Optional dry run: Actions → Release → Run workflow → `testpypi`.
4. Merge to `main`. A version that is already on PyPI or already tagged is
   skipped, so ordinary merges never publish anything.

Publishing a GitHub release by hand (tag `v<version>`) also works: the
distributions are published and attached to that release.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), the
same as `rust_local_ai`.
