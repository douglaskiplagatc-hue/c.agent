from agent.brain import ask_model


def main():

    print()
    print("================================")
    print("        MY CODING AGENT")
    print("================================")
    print()

    while True:

        task = input("What should I do? ").strip()

        if not task:
            continue

        if task.lower() in {
            "exit",
            "quit",
        }:
            print("Agent stopped.")
            break

        print()
        print("Agent is working...")
        print()

        try:

            result = ask_model(task)

            print()
            print("================================")
            print("AGENT RESULT")
            print("================================")
            print()
            print(result)
            print()

        except Exception as error:

            print()
            print("AGENT ERROR")
            print("------------")
            print(error)
            print()


if __name__ == "__main__":
    main()
