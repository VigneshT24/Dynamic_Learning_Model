from DLM import DLM

canContinue = True

while canContinue:
    bot = DLM()
    bot.ask()
    choice = input("Continue (Y/N): ")
    if (choice.lower() != "y"): canContinue = False
    else: canContinue = True
