import asyncio

import pytest

import python_local_ai as lai
from python_local_ai.testing import FakeBackend


def run(coro):
    return asyncio.run(coro)


def test_async_generate_and_context_manager():
    async def main():
        fake = FakeBackend()
        model = fake.model()
        assert (await model.availability()).available
        assert (await model.capabilities()).streaming
        async with await model.open_session("sys") as session:
            await session.add_query_chunk("a ")
            assert (await session.generate("b")).text == "a b"
            assert await session.count_tokens("x y") == 2
        assert session.closed

    run(main())


def test_async_stream():
    async def main():
        fake = FakeBackend()
        fake.set_stream_chunks(["Hel", "lo"])
        session = await fake.model().open_session()
        chunks = [chunk async for chunk in session.stream("hi")]
        assert chunks == ["Hel", "lo"]
        await session.close()

    run(main())


def test_busy_and_cancel():
    async def main():
        fake = FakeBackend()
        session = await fake.model().open_session()
        fake.set_generation_blocked(True)
        pending = asyncio.ensure_future(session.generate("slow"))
        await asyncio.sleep(0.05)
        with pytest.raises(lai.SessionBusyError):
            await session.generate("second")
        with pytest.raises(lai.SessionBusyError):
            await session.close()
        await session.cancel()
        fake.set_generation_blocked(False)
        with pytest.raises(lai.GenerationCancelledError) as info:
            await pending
        assert info.value.code == "cancelled"
        assert (await session.generate("after")).text == "slow" + "after"
        await session.close()

    run(main())


def test_sync_calls_release_the_gil():
    # A blocked generate_sync in a worker thread must not freeze the event loop.
    async def main():
        fake = FakeBackend()
        session = fake.model().open_session_sync()
        fake.set_generation_blocked(True)
        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(None, session.generate_sync, "threaded")
        await asyncio.sleep(0.05)
        assert not worker.done()
        fake.set_generation_blocked(False)
        assert (await worker).text == "threaded"

    run(main())
