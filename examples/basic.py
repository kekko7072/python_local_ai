"""Generate one reply with the OS-native model, or explain why it is unavailable."""

import python_local_ai as lai


def main() -> None:
    model = lai.detect()
    print(f"Backend: {model.backend.name} ({model.backend.kind})")

    availability = model.availability_sync()
    if not availability:
        print(f"Local AI unavailable: {availability.reason} — {availability.detail}")
        return

    with model.open_session_sync("You are a helpful assistant.") as session:
        print(session.generate_sync("Explain reinforcement learning simply.").text)


if __name__ == "__main__":
    main()
