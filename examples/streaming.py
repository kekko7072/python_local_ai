"""Stream a reply with asyncio, falling back to one-shot generation."""

import asyncio

import python_local_ai as lai


async def main() -> None:
    model = lai.detect()
    if not await model.availability():
        print("Local AI is unavailable on this device.")
        return

    capabilities = await model.capabilities()
    async with await model.open_session("Answer in three sentences.") as session:
        prompt = "Why does on-device AI matter?"
        if capabilities.streaming:
            async for chunk in session.stream(prompt):
                print(chunk, end="", flush=True)
            print()
        else:
            print((await session.generate(prompt)).text)


if __name__ == "__main__":
    asyncio.run(main())
