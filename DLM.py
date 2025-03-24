# Dynamic-Learning Model that learns how to respond to questions by learning from user input/expectations
# STEPS TO DEVELOP:
#   * allow user to ask an input
#   * if the input is not recognizable (not stored in stored_data.txt), then ask user what the response should be, store the question-answer pair in file
#   * keep doing this until a recognizable query comes up, and in that situation, give the response stored in key-value pair
import os
import difflib
class DLM:
    __filename = "stored_data.txt"
    __file = None
    __query = None
    __expectation = None

    # matcher = difflib.SequenceMatcher(None, "hello", "capitol")
    # print(matcher.ratio())

    def __init__(self):
        self.__file = open(self.__filename, "a")

    def __learn(self, query, expectation):
        self.__file.write(query + ">>" + expectation + "\n")

    def ask(self):
        self.__query = input("Query: ")

        # TO DO: bot needs to go through stored_data.txt to see if it can find a question that matches
        # if match has been found, respond with that question's answer
        # ask the user if that is the answer they were looking for: if YES, great, if NO, then learn that new question and expected answer user inputted

        self.__expectation = input("I don't know the answer to this question. What was the expected response from me: ")
        self.__learn(self.__query, self.__expectation) # learn this new question and answer pair and add to stored_data.txt
