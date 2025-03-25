from DLM import DLM

canContinue = True

print(f"{'\033[31m'}Welcome to Dynamic Learning Model Bot (DLM Bot){'\033[0m'}")
print(f"{'\033[31m'}DLM Bot gets smarter for every query asked because it either knows it or learns it{'\033[0m'}")
userChoice = input("\nDo you want to solely train DLM or ask DLM with questions (type 'train' or 'ask'): ")

bot = DLM()

while canContinue:
    bot.ask(userChoice == "train")
    choice = input("Continue (Y/N): ")
    canContinue = choice.lower() == "y"
