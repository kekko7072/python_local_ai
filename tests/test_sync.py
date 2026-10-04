import pytest

import python_local_ai as lai
from python_local_ai.testing import FakeBackend


def test_detect_never_raises_and_reports_availability():
    model = lai.detect()
    availability = model.availability_sync()
    assert isinstance(availability, lai.Availability)
    assert isinstance(model.backend, lai.BackendInfo)
    if not availability:
        assert availability.reason is not None
        with pytest.raises(lai.UnavailableError) as info:
            model.open_session_sync()
        assert info.value.code == "unavailable"
        assert info.value.reason == availability.reason


def test_generate_echoes_prompt_and_uses_queued_responses():
    fake = FakeBackend()
    with fake.model().open_session_sync("instructions") as session:
        assert session.generate_sync("hello").text == "hello"
        fake.push_response("queued", input_tokens=3, output_tokens=1, finish_reason="stop")
        response = session.generate_sync("ignored")
        assert response == session_response("queued", 3, 1, "stop", response)
        assert str(response) == "queued"
    assert session.closed


def session_response(text, inp, out, finish, actual):
    assert (actual.text, actual.input_tokens, actual.output_tokens, actual.finish_reason) == (text, inp, out, finish)
    return actual


def test_query_chunks_accumulate():
    session = FakeBackend().model().open_session_sync()
    session.add_query_chunk_sync("one ")
    assert session.generate_sync("two").text == "one two"


def test_sync_stream():
    fake = FakeBackend()
    fake.set_stream_chunks(["a", "b", "c"])
    session = fake.model().open_session_sync()
    assert list(session.stream("go")) == ["a", "b", "c"]
    # The lease is released once the stream ends.
    assert session.generate_sync("next").text


def test_capabilities_and_unsupported_errors():
    fake = FakeBackend(lai.Capabilities(text_generation=True))
    model = fake.model()
    caps = model.capabilities_sync()
    assert caps.text_generation and not caps.streaming
    session = model.open_session_sync()
    assert session.capabilities == caps
    with pytest.raises(lai.UnsupportedCapabilityError) as info:
        list(session.stream("x"))
    assert info.value.capability == "streaming"
    with pytest.raises(lai.UnsupportedCapabilityError):
        session.count_tokens_sync("a b")
    with pytest.raises(lai.UnsupportedCapabilityError) as info:
        session.generate_sync("x", response_format="json")
    assert info.value.capability == "structured_output"


def test_structured_output_and_config_validation():
    session = FakeBackend().model().open_session_sync()
    session.generate_sync("x", response_format={"type": "object"})
    session.generate_sync("x", response_format="json", temperature=0.5, top_p=0.9, seed=1, max_output_tokens=10)
    with pytest.raises(lai.InvalidConfigError):
        session.generate_sync("x", temperature=-1.0)
    with pytest.raises(ValueError):  # InvalidConfigError is also a ValueError
        session.generate_sync("x", top_p=2.0)
    with pytest.raises(lai.InvalidConfigError):
        session.generate_sync("x", max_output_tokens=0)
    with pytest.raises(TypeError):
        session.generate_sync("x", response_format="yaml")


def test_unavailable_backend():
    fake = FakeBackend()
    fake.set_availability(False, "model_not_ready", "still downloading")
    model = fake.model()
    availability = model.availability_sync()
    assert not availability
    assert (availability.reason, availability.detail) == ("model_not_ready", "still downloading")
    with pytest.raises(lai.UnavailableError) as info:
        model.open_session_sync()
    assert info.value.reason == "model_not_ready"


def test_backend_errors_and_close_retry():
    fake = FakeBackend()
    session = fake.model().open_session_sync()
    fake.push_backend_error("boom")
    with pytest.raises(lai.BackendError) as info:
        session.generate_sync("x")
    assert info.value.backend == "fake"
    assert isinstance(info.value, lai.LocalAiError)

    fake.fail_next_closes(1)
    with pytest.raises(lai.BackendError):
        session.close_sync()
    assert not session.closed
    session.close_sync()
    session.close_sync()  # idempotent
    with pytest.raises(lai.SessionClosedError):
        session.generate_sync("x")


def test_model_close_and_token_counting():
    model = FakeBackend().model()
    session = model.open_session_sync()
    assert session.count_tokens_sync("one two three") == 3
    model.close()
    assert model.closed
    with pytest.raises(lai.ModelClosedError):
        model.open_session_sync()
    # Existing sessions keep working.
    assert session.generate_sync("still").text == "still"


def test_rejected_turn_does_not_change_the_conversation():
    fake = FakeBackend()
    fake.set_stream_chunks(["x", "y"])
    session = fake.model().open_session_sync()
    stream = session.stream("first ")
    assert next(stream) == "x"
    # The stream holds the turn: a second turn is rejected before its prompt is added.
    with pytest.raises(lai.SessionBusyError):
        session.generate_sync("rejected ")
    stream.close()
    assert list(stream) == []
    assert session.generate_sync("second").text == "first second"


def test_repr_and_equality():
    caps = lai.Capabilities(streaming=True)
    assert caps == lai.Capabilities(streaming=True)
    assert "streaming=True" in repr(caps)
    assert repr(FakeBackend().model().backend).startswith("BackendInfo(kind=\"fake\"")
