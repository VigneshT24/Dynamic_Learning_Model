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

    # private helper called inside public method "ask" to simply add question/answer pair in specific text file
    def __learn(self, query, expectation):
        with open(self.__filename, "r") as file:
            for line in file:
                # don't do anything if pair already exists in database
                if (query + ">>" + expectation) == line: return
        self.__file.write("\n" + query + ">>" + expectation)

    # only method developer/user can use to train DLM or get answers to known queries
    def ask(self):
        self.__query = input("DML Bot here, ask away: ")
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
        self.__expectation = input("I don't know the answer. What was the expected response (training mode): ") # train DLM
        self.__learn(self.__query, self.__expectation) # learn this new question and answer pair and add to stored_data.txt
        print("I learned something new!") # confirmation that it went through the whole process
