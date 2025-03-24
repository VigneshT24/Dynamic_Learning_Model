# Dynamic-Learning Model that learns how to respond to questions by learning from user input/expectations
# STEPS TO DEVELOP:
#   * allow user to ask an input
#   * if the input is not recognizable (not stored in stored_data.txt), then ask user what the response should be, store the question-answer pair in file
#   * keep doing this until a recognizable query comes up, and in that situation, give the response stored in key-value pair
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
        self.__file.write(query + ">>" + expectation + "\n")

    # only method developer/user can use to train DLM or get answers to known queries
    def ask(self):
        self.__query = input("Query: ")
        with open(self.__filename, "r") as file: # go through database to see if question fuzzily matches with anything, if so, answer the question, else, learn the question
            for line in file:
                match = difflib.SequenceMatcher(None, self.__query, line.strip().split(">>")[0])
                if match.ratio() > 0.75: # returns a ratio representing how much the current question matches a specific question in database
                    print(line.split(">>", 1)[1].strip()) # if 75% match, return the answer
                    return
        self.__expectation = input("What was the expected response (training mode): ") # train DLM
        self.__learn(self.__query, self.__expectation) # learn this new question and answer pair and add to stored_data.txt
        print("I learned something new!") # confirmation that it went through the whole process
