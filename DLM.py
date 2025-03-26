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
        # Articles & Determiners
        "a", "an", "the", "some", "any", "each", "every", "either", "neither", "this", "that", "these", "those",

        # Pronouns
        "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves", "who", "whom", "whose", "which", "that",

        # Auxiliary Verbs (Helping Verbs)
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "shall", "should", "will", "would", "can", "could", "may", "might", "must",

        # Conjunctions
        "and", "but", "or", "nor", "so", "for", "yet", "although", "though", "because", "since", "unless",
        "while", "whereas", "either", "neither", "both", "whether", "not", "if",

        # Prepositions
        "about", "above", "across", "after", "against", "along", "among", "around", "as", "at",
        "before", "behind", "below", "beneath", "beside", "between", "beyond", "by",
        "despite", "down", "during", "except", "for", "from", "in", "inside", "into",
        "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past",
        "since", "through", "throughout", "till", "to", "toward", "under", "underneath",
        "until", "up", "upon", "with", "within", "without",

        # Common Adverbs (that don’t add meaning)
        "again", "already", "also", "always", "ever", "never", "just", "now", "often",
        "once", "only", "quite", "rather", "really", "seldom", "sometimes", "soon",
        "still", "then", "there", "therefore", "thus", "too", "very", "well",

        # Question Words
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",

        # Informal/Common Fillers
        "gonna", "wanna", "gotta", "lemme", "dunno", "kinda", "sorta", "aint", "ya", "yeah", "nah",

        # Verbs Commonly Used in Questions (but don’t change meaning)
        "do", "does", "did", "can", "could", "should", "shall", "will", "would", "may", "might", "must",

        # Additional Filler Phrases
        "tell", "please", "say", "let", "know", "consider", "find", "show", "explain", "define", "describe",
        "list", "give", "provide", "help", "make", "see", "like", "mean",

        # Contracted Forms (Common in Speech & Casual Writing)
        "i'd", "i'll", "i'm", "i've", "you'd", "you'll", "you're", "you've", "he'd", "he'll", "he's",
        "she'd", "she'll", "she's", "we'd", "we'll", "we're", "we've", "they'd", "they'll", "they're", "they've",
        "it's", "that's", "who's", "what's", "where's", "when's", "why's", "how's", "there's", "here's", "let's",

        # Miscellaneous Fillers
        "actually", "basically", "seriously", "literally", "obviously", "honestly", "frankly", "clearly",
        "apparently", "probably", "definitely", "certainly", "mostly", "mainly", "typically", "essentially",
        "generally", "approximately", "virtually", "kind", "sort", "type", "whatever", "however",

        # Excess Words That Don’t Change Sentence Meaning
        "thing", "stuff", "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody",
        "people", "person", "something", "anything", "everything", "nothing",

        # Placeholder & Non-Descriptive Words
        "thingy", "whatchamacallit", "doohickey", "thingamajig", "thingamabob",

        # Conversational Phrases That Don’t Add Meaning
        "you know", "i mean", "you see", "by the way", "sort of", "kind of", "more or less",
        "as far as i know", "in my opinion", "to be honest", "to be fair", "just saying",
        "at the end of the day", "if you ask me", "truth be told", "the fact is", "long story short"
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
        return random.choice(messages) if (len(userInput.split()) == 0) else None

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

                # only accept a match if similarity is 87% or more
                if similarity >= 0.87:
                    print(f"\n{'\033[34m'}" + line.split(">>", 1)[1].strip() + f"{'\033[0m'}\n")
                    self.__expectation = input("Is this what you expected (Y/N): ")

                    while not self.__expectation:
                        self.__expectation = input("Empty input is not acceptable. Is this what you expected (Y/N): ")

                    if self.__expectation.lower() == "y":
                        print("Great!")
                        return
                    break  # if incorrect, allow learning

        incompleteness = self.__isIncomplete(filtered_query)
        if incompleteness != None:
            print(str(incompleteness))
            return

        self.__expectation = input("I'm not sure. What was the expected response (training mode): ") # train DLM

        while not self.__expectation:
            print("Nothing learnt. Moving on.")
            return

        self.__learn(filtered_query, self.__expectation) # learn this new question and answer pair and add to stored_data.txt
        print("I learned something new!") # confirmation that it went through the whole process
