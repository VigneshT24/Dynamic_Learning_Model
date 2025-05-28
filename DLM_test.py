from DLM import DLM
import time
import threading, itertools, sys      # std-lib helpers for the spinner

# ANSI escape for moving the cursor up N lines
def move_cursor_up(lines):
    print(f"\033[{lines}A", end="")

# loading animation WITH threading for parallel processing
def loadingAnimation(message, stop_event):
    dots = itertools.cycle([".", "..", "..."])
    while not stop_event.is_set():
        sys.stdout.write(f"\r{message}{next(dots)}   ")
        sys.stdout.flush()
        time.sleep(0.5)
    sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")

canContinue = True

# introduction and disclaimers
print(f"\033[31mWelcome to Dynamic Learning Model Bot (DLM Bot). This bot can be trained for any purposes.\033[0m")
print(f"\033[31mDLM Bot gets smarter for every query asked because it either knows it or learns it for next time\033[0m")
print(f"\033[31mNOTICE: DLM Bot may sometimes misinterpret input or provide inaccurate responses. Please verify important information independently.\033[0m")

userChoice = input("\nThere are two options: Train the DLM Bot (type 'T') or use it as is (type 'A'): ")
while userChoice.lower() not in ("t", "a"):
    userChoice = input("\nPlease type either 'T' or 'A' to proceed: ")

# Start the spinner before creating the DLM object
stop_spinner = threading.Event()
spinner = threading.Thread(
    target=loadingAnimation,
    args=("Starting SQL Server", stop_spinner),
    daemon=True)
spinner.start()

bot = DLM()

stop_spinner.set()     # signal the spinner to finish
spinner.join()         # wait until it cleans the line
print("SQL Server ready!\n")

while canContinue:
    bot.ask(userChoice)
    choice = input("Continue (Y/N): ")
    while choice.lower() not in ("y", "n"):
        choice = input("\nPlease type either 'Y' or 'N' to proceed: ")
    canContinue = (choice.lower() == "y")
