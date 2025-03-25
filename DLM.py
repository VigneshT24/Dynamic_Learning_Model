# Dynamic-Learning Model that learns how to respond to questions by learning from user input/expectations
import difflib
class DLM:
    __filename = "stored_data.txt"
    __file = None
    __query = None
    __expectation = None

    # constructor that opens the database file
    def __init__(self):
        self.__file = open(self.__filename, "a")

    # checks to see if given query already exists in the database (avoids redundant data)
    def data_exists(self):
        with open(self.__filename, "r") as file:
            for line in file:
                # don't do anything if pair already exists in database
                if self.__query == line.strip().split(">>")[0]: return True
        return False

    # handles the learning aspect of the bot; if Q/A pair doesn't exist in database, add it there
    def learn(self, query, expectation):
        if (not self.data_exists()):
            self.__file.write("\n" + query + ">>" + expectation)

    # only method developer/user can use to train DLM or get answers to known queries
    def ask(self, trainingOnly):
        print("Training Only") if trainingOnly else print("AMA Only (with training ability)")
        self.__query = input("DML Bot here, ask away: ")
        if (not trainingOnly):
            with open(self.__filename, "r") as file: # go through database to see if question fuzzily matches with anything, if so, answer the question, else, learn the question
                for line in file:
                    match = difflib.SequenceMatcher(None, self.__query, line.strip().split(">>")[0])
                    if match.ratio() > 0.85: # returns a ratio representing how much the current question matches a specific question in database
                        print(f"\n{'\033[34m'}" + line.split(">>", 1)[1].strip() + f"{'\033[0m'}\n") # if 85% match, give the answer
                        self.__expectation = input("Is this what you expected (Y/N): ")
                        if self.__expectation.lower() == "y":
                            print("Great!")
                            return
                        break
        if (not self.data_exists()):
            self.__expectation = input("I don't know the answer. What was the expected response (training mode): ") # train DLM
            self.learn(self.__query, self.__expectation) # learn this new question and answer pair and add to stored_data.txt
            print("I learned something new!") # confirmation that it went through the whole process
        else:
            print("I already know this! Try another query.")
