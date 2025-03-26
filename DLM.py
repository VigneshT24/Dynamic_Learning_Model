# Dynamic-Learning Model (DLM) bot that learns how to respond to questions by learning from user input/expectations
import difflib
import string
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
        with (open(self.__filename, "r") as file): # go through database to see if question fuzzily matches with anything, if so, answer the question, else, learn the question
            for line in file:
                stored_question = line.strip().split(">>")[0].lower()
                query_lower = self.__query.lower()

                # Split the query and stored question into two halves
                mid_query = len(query_lower) // 2
                mid_stored = len(stored_question) // 2

                query_first_half, query_second_half = query_lower[:mid_query].translate(str.maketrans('', '', string.punctuation)),query_lower[mid_query:].translate(str.maketrans('', '', string.punctuation))
                stored_first_half, stored_second_half = stored_question[:mid_stored].translate(str.maketrans('', '', string.punctuation)), stored_question[mid_stored:].translate(str.maketrans('', '', string.punctuation))

                # Compare first half and second half separately
                first_half_match = difflib.SequenceMatcher(None, query_first_half, stored_first_half).ratio()
                second_half_match = difflib.SequenceMatcher(None, query_second_half, stored_second_half).ratio()
                if first_half_match > 0.85 and second_half_match > 0.60: # returns a ratio representing how much the current question matches a specific question in database
                    print(f"\n{'\033[34m'}" + line.split(">>", 1)[1].strip() + f"{'\033[0m'}\n") # if 80% match, give the answer
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
