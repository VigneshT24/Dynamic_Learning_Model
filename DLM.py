# Dynamic-Learning Model (DLM) bot that learns how to respond to questions by learning from user input/expectations
import difflib
import string
import random

class DLM:
    __filename = "stored_data.txt"
    __query = None
    __expectation = None

    # following 3 lists are used to detect specific types of incomplete sentences
    __conjunctions = [
        "if", "but", "and", "so", "because", "or", "then", "although", "though", "whereas", "while", "unless", "until",
        "for", "nor", "yet", "after", "as", "as if", "as long as", "as much as", "as soon as", "as though",
        "before", "even if", "even though", "if only", "in order that", "once",
        "provided that", "rather than", "since", "so that", "than", "that",
        "where", "wherever", "whenever", "whether", "why"
    ]

    __auxiliary_verbs = [
        "is", "was", "were", "am", "are", "be", "being", "been", "will", "shall", "should", "would",
        "can", "could", "may", "might", "must", "do", "does", "did", "has", "have", "had",
        "ought", "need", "dare", "used",
    ]

    __prepositions = [
        "in", "on", "at", "for", "with", "about", "of", "by", "to", "from", "under", "over",
        "between", "into", "onto", "without", "through", "among", "beside", "around", "before",
        "after", "against", "during", "within", "beyond", "beneath", "behind", "above", "below",
        "towards", "along", "across", "throughout", "into", "upon", "through", "out", "up"
    ]

    def __learn(self, query, expectation):
        """ Stores the new query and expectation pair in stored_data.txt """
        with open(self.__filename, "a") as file:
            file.write("\n" + query + ">>" + expectation)

    def __isIncomplete(self, userInput):
        """
            __isIncomplete Method {private}
            =======================

            Description:
            Checks if userInput is incomplete by analyzing its structure and last word.

            Parameters:
            userInput: A string input by the user that will be checked for possible incompleteness.

            Returns:
            result: A message notifying the user that their input might be incomplete.

            Raises:
            TypeError: If 'userInput' is not a string.
        """
        if not isinstance(userInput, str):
            raise TypeError("Expected a string input.")

        userInput = userInput.translate(str.maketrans("", "", string.punctuation.replace("?", ""))).strip()
        words = userInput.split()

        if not words:
            return None  # Empty input is not incomplete, just ignored.

        lastWord = words[-1].lower()

        # Common question starters to detect valid questions
        question_starters = {"who", "what", "when", "where", "why", "how", "which", "whose", "whom", "did", "does",
                             "do", "is", "are", "can", "could", "will", "would", "shall", "should", "may", "might",
                             "was", "were", "has", "have", "had"}

        # If it's a question (ends with "?" or starts with a question word), it's valid
        if userInput.endswith("?") or words[0].lower() in question_starters:
            return None

        messages = {
            "conjunction": [
                "It looks like your thought isn't finished. Did you mean to continue?",
                "Your sentence ends with a conjunction. Do you want to add something?",
                "Hmm, that seems unfinished. What were you about to say next?"
            ],
            "auxiliary_verb": [
                "Your input stops at an auxiliary verb. What were you trying to express?",
                "It sounds like something is missing. Want to complete your thought?",
                "That feels incomplete. Can you clarify what you meant?"
            ],
            "preposition": [
                "Your sentence ends with a preposition. Were you about to add more?",
                "That seems like it's missing a part. What comes after?",
                "It sounds like you were going to say something else. Want to continue?"
            ]
        }

        # Check if last word matches one of the incomplete categories
        if any(lastWord == word.lower() for word in self.__conjunctions):
            return random.choice(messages["conjunction"])
        elif any(lastWord == word.lower() for word in self.__auxiliary_verbs):
            return random.choice(messages["auxiliary_verb"])
        elif any(lastWord == word.lower() for word in self.__prepositions):
            return random.choice(messages["preposition"])

        return None  # If none of the conditions apply, it's a complete sentence

    def __tokenize(self, userInput):
        """ Tokenizes the sentence and removes punctuation """
        return userInput.translate(str.maketrans('', '', string.punctuation)).lower().split()

    def ask(self):
        """ Main method in which the user is able to ask any query and DLM will either answer it or learn it """
        self.__query = input("DLM Bot here, ask away: ")
        with open(self.__filename, "r") as file:
            for line in file:
                stored_question = line.strip().split(">>")[0].lower()
                query_lower = self.__query.lower()

                # Tokenize query and stored question into words
                query_tokens = self.__tokenize(query_lower)
                stored_tokens = self.__tokenize(stored_question)

                # Calculate similarity based on word importance
                keyword_match_count = sum(1 for word in query_tokens if word in stored_tokens)
                total_unique_words = len(set(query_tokens + stored_tokens))

                # Compute weighted similarity score
                keyword_similarity = keyword_match_count / total_unique_words  # How many words match?
                overall_similarity = difflib.SequenceMatcher(None, query_lower, stored_question).ratio()  # String-based match

                # Only accept a match if BOTH word-based and overall similarity are high enough
                if keyword_similarity > 0.5 and overall_similarity > 0.85:
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
        self.__learn(self.__query, self.__expectation) # learn this new question and answer pair and add to stored_data.txt
        print("I learned something new!") # confirmation that it went through the whole process
