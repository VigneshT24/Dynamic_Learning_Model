# Dynamic-Learning Model (DLM) bot that learns how to respond to questions by learning from user input/expectations
import difflib
class DLM:
    __filename = "stored_data.txt"
    __query = None
    __expectation = None

    # private helper that handles the learning aspect of the bot; if Q/A pair doesn't exist in database, add it there
    def __learn(self, query, expectation):
        with open(self.__filename, "a") as file:
            file.write("\n" + query + ">>" + expectation)

    # public method that handles the ask me anything (AMA) aspect of the bot; this is where the bot either learns or knows queries
    def ask(self):
        self.__query = input("DLM Bot here, ask away: ")
        with open(self.__filename, "r") as file: # go through database to see if question fuzzily matches with anything, if so, answer the question, else, learn the question
            for line in file:
                match = difflib.SequenceMatcher(None, self.__query.lower(), line.strip().split(">>")[0].lower())
                if match.ratio() > 0.85: # returns a ratio representing how much the current question matches a specific question in database
                    print(f"\n{'\033[34m'}" + line.split(">>", 1)[1].strip() + f"{'\033[0m'}\n") # if 90% match, give the answer
                    self.__expectation = input("Is this what you expected (Y/N): ")
                    while not self.__expectation: # expectation must not be empty
                        self.__expectation = input("Empty input is not acceptable. Is this what you expected (Y/N): ")
                    if self.__expectation.lower() == "y":
                        print("Great!")
                        return
                    break # bot doesn't know, do it executes the bottom code
        self.__expectation = input("I don't know the answer. What was the expected response (training mode): ") # train DLM
        while not self.__expectation:
            print("Nothing learnt. Moving on.")
            return
        self.__learn(self.__query, self.__expectation) # learn this new question and answer pair and add to stored_data.txt
        print("I learned something new!") # confirmation that it went through the whole process
