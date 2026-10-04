"""Deterministic backend for tests that must not depend on OS AI hardware.

>>> from python_local_ai.testing import FakeBackend
>>> fake = FakeBackend()
>>> fake.push_response("hello")
>>> session = fake.model().open_session_sync()
>>> session.generate_sync("hi").text
'hello'
"""

from ._native import FakeBackend

__all__ = ["FakeBackend"]
