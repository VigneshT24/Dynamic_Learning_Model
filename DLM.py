# Dynamic-Learning Model (DLM) bot that learns how to respond to questions by learning from user input/expectations
import difflib
import string
import random

class DLM:
    __filename = "stored_data.txt" # database
    __query = None # user-inputted query
    __expectation = None # user-inputted expected answer to query

    # words to be filtered from user input for better accuracy and less distractions
    __filler_words = [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if",
        "in", "into", "is", "it", "of", "on", "or", "such", "that", "the",
        "their", "then", "there", "these", "they", "this", "to", "was", "will",
        "with", "about", "after", "again", "against", "all", "am", "any", "because",
        "before", "being", "between", "both", "during", "each", "few", "further",
        "had", "has", "have", "he", "her", "here", "hers", "him", "himself", "his",
        "how", "i", "into", "itself", "me", "more", "most", "my", "myself", "no",
        "nor", "not", "now", "of", "off", "on", "once", "only", "other", "our",
        "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
        "some", "such", "than", "too", "under", "until", "up", "very", "was", "we",
        "were", "what", "when", "where", "which", "while", "who", "whom", "why",
        "with", "you", "your", "yours", "yourself", "yourselves"
    ]

    def __learn(self, query, expectation):
        """ Stores the new query and expectation pair in stored_data.txt """
        with open(self.__filename, "a") as file:
            file.write("\n" + query + ">>" + expectation)

    def __isIncomplete(self, userInput):
        """ if the filtered userInput is empty, that would mean that there were only filler words, therefore denoting as incomplete """
        if not isinstance(userInput, str):
            raise TypeError("Expected a string input.")

        # gives personalized message to user when message is incomplete
        messages = [
                "It looks like your thought isn't finished. Did you mean to continue?",
                "Your sentence is incomplete. Do you want to add something?",
                "Hmm, that seems unfinished. What were you about to say next?",
                "Your input stops abruptly. What were you trying to express?",
                "It sounds like something is missing. Want to complete your thought?",
                "That feels incomplete. Can you clarify what you meant?",
                "Your sentence ends weirdly. Were you about to add more?",
                "That seems like it's missing a part. What comes after?",
                "It sounds like you were going to say something else. Want to continue?"]
        return random.choice(messages) if len(userInput) == 0 else None

    def __filtered_input (self, userInput):
        """ filter all the words using 'filler_words' list """
        # Tokenize user input (split into words)
        words = userInput.lower().split()

        # Remove filler words
        filtered_words = [word for word in words if word not in self.__filler_words]

        # Join the remaining words back into a string
        return " ".join(filtered_words)

    def ask(self):
        """ Main method in which the user is able to ask any query and DLM will either answer it or learn it """
        self.__query = input("DLM Bot here, ask away: ")

        # storing the user-query (filtered and lower-case)
        filtered_query = self.__filtered_input(self.__query.lower().translate(str.maketrans('', '', string.punctuation)))
        with open(self.__filename, "r") as file:
            for line in file:

                # storing the database line's query
                stored_question = line.strip().split(">>")[0].lower()
                similarity = difflib.SequenceMatcher(None, stored_question, filtered_query).ratio()

                # Only accept a match if similarity is 87% or more
                if similarity >= 0.87:
                    print(f"\n{'\033[34m'}" + line.split(">>", 1)[1].strip() + f"{'\033[0m'}\n")
                    self.__expectation = input("Is this what you expected (Y/N): ")

                    while not self.__expectation:
                        self.__expectation = input("Empty input is not acceptable. Is this what you expected (Y/N): ")

                    if self.__expectation.lower() == "y":
                        print("Great!")
                        return
                    break  # If incorrect, allow learning

        if self.__isIncomplete(self.__query) != None:
            print(str(self.__isIncomplete(self.__query)))
            return

        self.__expectation = input("I'm not sure. What was the expected response (training mode): ") # train DLM

        while not self.__expectation:
            print("Nothing learnt. Moving on.")
            return

        self.__learn(filtered_query, self.__expectation) # learn this new question and answer pair and add to stored_data.txt
        print("I learned something new!") # confirmation that it went through the whole process
