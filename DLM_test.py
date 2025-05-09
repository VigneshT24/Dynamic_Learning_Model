from DLM import DLM
import time

def loadingAnimation(input):
    for seconds in range(0, 3):
        print(f"{input}{'.' * (seconds + 1)}   ", end = " ")
        time.sleep(0.5)

canContinue = True
devPassword = "12345"

# introduction and disclaimers
print(f"{'\033[31m'}Welcome to Dynamic Learning Model Bot (DLM Bot). This bot can be trained for any purposes.{'\033[0m'}")
print(f"{'\033[31m'}DLM Bot gets smarter for every query asked because it either knows it or learns it for next time{'\033[0m'}")
print(f"{'\033[31m'}NOTICE: DLM Bot may sometimes misinterpret input or provide inaccurate responses. Please verify important information independently.{'\033[0m'}")
userChoice = input("\nThere are two options: Train the DLM Bot (type 'T') or use it as is (type 'A'): ")

while (userChoice.lower() != "t" and userChoice.lower() != "a"):  # if userChoice is not the expected response, keep asking until it is
    userChoice = input("\nPlease type either 'T' or 'A' to proceed: ")

password = None

if (userChoice.lower() == "t"):
    password = input("Enter the password to enter Training Mode: ")
    while (password != devPassword):
        password = input("Password is incorrect, try again or type 'stop' to enter in commercial mode instead: ")
        if (password.lower() == "stop"):
            userChoice = "A"
            break

if (password == devPassword):
    loadingAnimation("Logging in as Trainer")
else:
    loadingAnimation("Logging in as Commercial User")

print("\n")
bot = DLM("stored_data.txt")  # stored_data.txt is the knowledge base that bot will be using

while canContinue:
    bot.ask(userChoice.lower() == "t")
    choice = input("Continue (Y/N): ")
    while (choice.lower() != "y" and choice.lower() != "n"):  # if choice is not the expected response, keep asking until it is
        choice = input("\nPlease type either 'Y' or 'N' to proceed: ")
    canContinue = choice.lower() == "y"
