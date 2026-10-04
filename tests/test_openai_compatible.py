"""End-to-end tests through rust_local_ai's OpenAI-compatible backend.

A tiny local server stands in for llama.cpp / Ollama / LM Studio, so these
exercise the real Python -> Rust -> HTTP path on every CI platform.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import python_local_ai as lai


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def _json(self, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            self._json({"object": "list", "data": [{"id": "tiny-model", "object": "model"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(request)
        reply = "echo: " + request["messages"][-1]["content"]
        if not request.get("stream"):
            self._json(
                {
                    "choices": [{"message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                }
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in reply.split(" "):
            chunk = {"choices": [{"delta": {"content": word + " "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.requests = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_only_local_http_urls_are_accepted():
    with pytest.raises(lai.InvalidConfigError):
        lai.openai_compatible("https://api.example.com/v1")


def test_unreachable_service_reports_unavailable():
    # Port 9 (discard) has nothing listening on CI runners.
    model = lai.openai_compatible("http://127.0.0.1:9/v1")
    assert model.backend.kind == "openai_compatible"
    availability = model.availability_sync()
    assert not availability
    assert availability.reason in ("provider_not_installed", "backend_failure")


def test_generate_and_stream_against_a_local_service(server: str):
    model = lai.openai_compatible(server)
    availability = model.availability_sync()
    assert availability, availability
    capabilities = model.capabilities_sync()
    assert capabilities.text_generation and capabilities.streaming

    with model.open_session_sync("Be brief.") as session:
        assert session.generate_sync("hello").text == "echo: hello"
        streamed = "".join(session.stream("again please"))
        assert streamed.strip() == "echo: again please"

    first, second = _Handler.requests
    assert first["model"] == "tiny-model"
    assert first["messages"][0] == {"role": "system", "content": "Be brief."}
    # The session keeps the conversation: the second turn carries the first.
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert second["stream"] is True


def test_async_against_a_local_service(server: str):
    async def main() -> None:
        model = lai.openai_compatible(server, "tiny-model")
        async with await model.open_session() as session:
            reply = await session.generate("async")
            assert reply.text == "echo: async"
            chunks = [chunk async for chunk in session.stream("streamed")]
            assert "".join(chunks).strip() == "echo: streamed"

    asyncio.run(main())


def test_inference_snap_is_linux_only():
    import sys

    if sys.platform.startswith("linux"):
        model = lai.inference_snap("qwen3")
        assert isinstance(model.availability_sync(), lai.Availability)
    else:
        with pytest.raises(lai.UnavailableError):
            lai.inference_snap("qwen3")


def test_reports_the_bound_rust_core_version():
    from python_local_ai import _native

    assert _native.RUST_LOCAL_AI_VERSION.startswith("0.2.")
