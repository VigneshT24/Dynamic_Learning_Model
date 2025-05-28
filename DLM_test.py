from DLM import DLM
import time

# ANSI escape for moving the cursor up N lines
def move_cursor_up(lines):
    print(f"\033[{lines}A", end="")

# loading animation utilizes "move_cursor_up"
def loadingAnimation(input):
    for seconds in range(0, 3):
        print(f"\r{input}{'.' * (seconds + 1)}   ", end="", flush=True)
        time.sleep(0.5)
    print()

canContinue = True
trainingPwd = "371507"

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
    while (password != trainingPwd):
        password = input("Password is incorrect, try again or type 'stop' to enter in commercial mode instead: ")
        if (password.lower() == "stop"):
            userChoice = "A"
            break

if (password == trainingPwd):
    # trainers must understand these rules as DLM can generate bad responses if these instructions are neglected
    print(f"\n\n{'\033[31m'}MAKE SURE TO UNDERSTAND THE FOLLOWING ANSWER FORMAT EXPECTED FOR EACH CATEGORY FOR THE BOT TO LEARN ACCURATELY:{'\033[0m'}\n")
    print("*'yesno': Make sure to start your answer responses with \"yes\" or \"no\" ONLY")
    print("*'process': Each answer must have three steps for your responses, separated by \";\" (semicolon)")
    print("*'definition': Make sure to not mention the WORD/PHRASE to be defined & always start your response here with \"the\" only")
    print("*'deadline': Only include the deadline date, as an example, \"March 31st 2025\"")
    print("*'location': Mention the location only, nothing else. For example, \"The FAFSA.Gov website\"")
    print("*'generic': Format doesn't matter for this, give your answer in any comprehensive format")
    print("*'eligibility': Make sure to ONLY start the response with a pronoun like \"you\", \"they\", \"he\", \"she\", etc\n\n")

    confirmation = input("Make sure to understand and note these instructions somewhere as the generated responses would get corrupt otherwise.\nType 'Y' to continue: ")
    while confirmation.lower() != "y": # trainers must understand the instructions above
        confirmation = input("You cannot proceed to train without understanding the instructions aforementioned. Type 'Y' to continue: ")
    loadingAnimation("Logging in as Trainer")
else:
    loadingAnimation("Logging in as Commercial User")

print("\n")
bot = DLM()  # SQL dlm_knowledge.db is the knowledge base that bot will be using

while canContinue:
    bot.ask(password)
    choice = input("Continue (Y/N): ")
    while (choice.lower() != "y" and choice.lower() != "n"):  # if choice is not the expected response, keep asking until it is
        choice = input("\nPlease type either 'Y' or 'N' to proceed: ")
    canContinue = choice.lower() == "y"
