from DLM import DLM

canContinue = True

print(f"{'\033[31m'}Welcome to Dynamic Learning Model Bot (DLM Bot){'\033[0m'}")
print(f"{'\033[31m'}DLM Bot gets smarter for every query asked because it either knows it or learns it{'\033[0m'}")
userChoice = input("\nAre you ready to ask it questions (Y/N): ")

bot = DLM()

while (userChoice.lower() != "y" and userChoice.lower() != "n"):
    userChoice = input("\nPlease type either 'Y' or 'N' to proceed: ")

while canContinue:
    bot.ask()
    choice = input("Continue (Y/N): ")
    while (choice.lower() != "y" and choice.lower() != "n"):
        userChoice = input("\nPlease type either 'Y' or 'N' to proceed: ")
    canContinue = choice.lower() == "y"
