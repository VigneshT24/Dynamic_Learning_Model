from DLM import DLM

canContinue = True

print(f"{'\033[31m'}Welcome to Dynamic Learning Model Bot (DLM Bot). This bot can be trained for any purposes.{'\033[0m'}")
print(f"{'\033[31m'}DLM Bot gets smarter for every query asked because it either knows it or learns it for next time{'\033[0m'}")
userChoice = input("\nThere are two options: Train the DLM Bot (type 'T') or use it as is (type 'A'): ")

bot = DLM()

while (userChoice.lower() != "t" and userChoice.lower() != "a"):
    userChoice = input("\nPlease type either 'Y' or 'N' to proceed: ")

while canContinue:
    bot.ask(userChoice.lower() == "t")
    choice = input("Continue (Y/N): ")
    while (choice.lower() != "y" and choice.lower() != "n"):
        choice = input("\nPlease type either 'Y' or 'N' to proceed: ")
    canContinue = choice.lower() == "y"
