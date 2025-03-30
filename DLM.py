# Dynamic-Learning Model (DLM) bot that learns how to respond to questions by learning from user input/expectations
import difflib
import string
import random
import spacy

class DLM:
    __filename = "stored_data.txt"  # knowledge-base
    __query = None  # user-inputted query
    __expectation = None  # user-inputted expected answer to query
    __nlp = spacy.load("en_core_web_md") # Spacy NLP analysis

    # personalized responses to let the user know that the bot doesn't know the answer
    __fallback_responses = [
        "Hmm, that's a great question! I might need more context or details to answer it.",
        "I'm still training my brain on that topic. Could you clarify what you mean?",
        "Oops! That one's not in my database yet, or maybe it's phrased in a way I don't recognize!",
        "You got me this time! Could you try rewording it so I can understand better?",
        "That's a tough one! I might need a bit more information to figure it out.",
        "I don't have the answer just yet, but I bet it’s out there somewhere! Could you rephrase it?",
        "Hmm... I’ll have to hit the books for that one! Or maybe I just need a little more context?",
        "I haven't learned that yet, but I'm constantly improving! Maybe try a different wording?",
        "You just stumped me! But no worries, I’m always evolving—maybe I misinterpreted the question?",
        "That’s outside my knowledge base for now, or maybe I'm just not parsing it right!",
        "I wish I had the answer! If it’s incomplete, could you add more details?",
        "I’m not sure about that one. Maybe try breaking it down into smaller parts?",
        "Hmm, I don’t have an answer yet. Could you reword or give more details?",
        "Still learning this one! If something’s missing, feel free to add more context.",
        "I don’t have that in my knowledge bank yet, or maybe I'm missing part of the question!"
    ]

    # words to be filtered from user input for better accuracy and fewer distractions
    __filler_words = [

        # Articles & Determiners (Words that don't add meaning to sentence)
        "a", "an", "the", "some", "any", "each", "every", "either", "neither", "this", "that", "these", "those",
        "certain", "another", "such", "whatsoever", "whichever", "whomever", "whatever", "all",

        # Pronouns (General pronouns that don’t change meaning)
        "i", "me", "my", "mine",
        "myself", "you", "your", "yours", "yourself", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves", "who", "whom", "whose", "which", "that",
        "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody", "people", "person",
        "whoever", "wherever", "whenever", "whosoever", "others", "oneself",

        # Auxiliary (Helping) Verbs (Do not contribute meaning)
        "get", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "best", "do",
        "does",
        "did", "doing", "shall", "should", "will", "would", "can", "could", "may", "might", "must", "bad", "dare",
        "need", "want",
        "used", "shallnt", "shouldve", "wouldve", "couldve", "mustve", "mightve", "mustnt", "good",

        # Conjunctions (Connectors that do not change meaning)
        "and", "but", "or", "gotten",
        "nor", "so", "for", "yet", "although", "though", "because", "since", "unless",
        "while", "whereas", "either", "neither", "both", "whether", "not", "if", "even if", "even though", "common",
        "as long as",
        "provided that", "whereas", "therefore", "thus", "hence", "meanwhile", "besides", "furthermore",

        # Prepositions (Location/Relation words that are often unnecessary)
        "about", "above", "across", "after", "against", "along", "among", "around", "as", "at",
        "before", "behind", "below", "beneath", "beside", "between", "beyond", "by", "low", "high", "despite", "down",
        "during", "happen",
        "except", "for", "from", "in", "inside", "into",
        "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past",
        "since", "through", "throughout", "till", "to", "toward", "under", "underneath",
        "until", "up", "upon", "with", "within", "without", "aside from", "concerning", "regarding",

        # Common Adverbs (Time words and intensity words that add fluff)
        "way", "ways", "again", "already", "also", "always", "ever", "never", "just", "now", "often",
        "once", "only", "quite", "rather", "really", "seldom", "sometimes", "soon", "got",
        "still", "then", "there", "therefore", "thus", "too", "very", "well", "anytime",
        "hardly", "barely", "scarcely", "seriously", "truly", "frankly", "honestly", "basically", "literally",
        "definitely", "obviously", "surely", "likely", "probably", "certainly", "clearly", "undoubtedly",

        # Question Words (Words that do not impact search meaning)
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "whichever", "whomever", "whenever", "wherever", "whosoever", "however", "whence",

        # Informal/Common Fillers (Spoken language fillers)
        "gonna", "wanna", "gotta", "lemme", "dunno", "kinda", "sorta", "aint", "ya", "yeah", "nah",
        "um", "uh", "hmm", "huh", "mmm", "uhh", "ahh", "err", "ugh", "tsk", "like", "okay", "ok", "alright",
        "yo", "bruh", "dude", "bro", "sis", "mate", "fam", "nah", "yup", "nope", "welp",

        # Verbs Commonly Used in Questions (but don’t change meaning)
        "go", "do", "dont", "does", "did", "can", "can't", "could", "couldnt", "should", "shouldnt", "shall", "will", "would", "wouldnt", "may", "might", "must", "use", "tell",
        "please", "say", "let", "know", "consider", "find", "show", "explain", "define", "describe", "take",
        "list", "give", "provide", "make", "see", "mean", "understand", "point out", "stay", "look", "care",

        # Contracted Forms (Casual writing contractions),
        "ill", "im", "ive", "youd", "youll", "youre", "youve", "hed", "hell", "hes",
        "shed", "shell", "shes", "wed", "well", "were", "weve", "theyd", "theyll", "theyre", "theyve",
        "its", "thats", "whos", "whats", "wheres", "whens", "whys", "hows", "theres", "heres", "lets",

        # Conversational Fillers (Unnecessary words in casual speech)
        "actually", "basically", "seriously", "literally", "obviously", "honestly", "frankly", "clearly",
        "apparently", "probably", "definitely", "certainly", "most", "mostly", "mainly", "typically", "essentially",
        "generally", "approximately", "virtually", "kind", "sort", "type", "whatever", "however",
        "you know", "i mean", "you see", "by the way", "sort of", "kind of", "more or less",
        "as far as i know", "in my opinion", "to be honest", "to be fair", "just saying",
        "at the end of the day", "if you ask me", "truth be told", "the fact is", "long story short",

        # Internet Slang, Misspellings, and Shortcuts
        "lol", "lmao", "rofl", "omg", "idk", "fyi", "btw", "imo", "smh", "afk", "ttyl", "brb",
        "thx", "pls", "ppl", "u", "ur", "r", "cuz", "coz", "cause", "gimme", "lemme", "wassup", "sup",

        # Placeholder & Non-Descriptive Words
        "thing", "stuff", "thingy", "whatchamacallit", "doohickey", "thingamajig", "thingamabob",

        # Words That Don’t Add Meaning
        "important", "necessary", "specific", "certain", "particular", "special", "exactly", "precisely",
        "recently", "currently", "today", "tomorrow", "yesterday", "soon", "later", "eventually", "sometime",

        # Overused Transitions
        "so", "then", "therefore", "thus", "anyway", "besides", "moreover", "furthermore", "meanwhile"
    ]

    def __semantic_similarity(self, userInput, knowledgebaseData):
        """ semantically analyzes user input and database's best match to see if they can still match using Spacy """
        # break parameter inputs into list of words
        UI_list = userInput.lower().split()
        KB_list = knowledgebaseData.lower().split()

        # Initialize match count
        match_count = 0

        for user_word in UI_list:
            user_token = self.__nlp(user_word)
            for knowledge_word in KB_list:
                knowledge_token = self.__nlp(knowledge_word)
                # Use Spacy's similarity function to check semantic similarity
                if user_token.similarity(knowledge_token) > 0.7: # similarity must be over 70%
                    match_count += 1
                    break  # Stop checking this word as it has found a match

        # Calculate the match ratio
        match_ratio = match_count / len(KB_list)
        if (match_ratio >= 75):
            return True
        return False

    def __learn(self, query, expectation):
        """ Stores the new query and expectation pair in stored_data.txt """
        with open(self.__filename, "a") as file:
            file.write("\n" + query + ">>" + expectation)

    def __filtered_input(self, userInput):
        """ filter all the words using 'filler_words' list """
        # Tokenize user input (split into words)
        words = userInput.lower().split()

        # Remove filler words
        filtered_words = [word for word in words if word.lower() not in self.__filler_words]

        # Join the remaining words back into a string
        return " ".join(filtered_words)

    def ask(self, trainingMode):
        """ Main method in which the user is able to ask any query and DLM will either answer it or learn it """
        print("\nTRAINING MODE") if (trainingMode == True) else print("\nCOMMERCIAL MODE")
        self.__query = input("DLM Bot here, ask away: ")

        # storing the user-query (filtered and lower-case)
        filtered_query = self.__filtered_input(self.__query.lower().translate(str.maketrans('', '', string.punctuation)))

        highest_similarity = 0
        best_match_answer = None  # stores the best answer after o(n) iterations
        with open(self.__filename, "r") as file:
            for line in file:
                # storing both the question and answer from database
                stored_question, stored_answer = line.strip().split(">>", 1)
                stored_question = stored_question.lower()

                # Calculate similarity
                similarity = difflib.SequenceMatcher(None, stored_question, filtered_query).ratio()

                # Keep track of the best match
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_answer = stored_answer.strip()

        # accept a match if highest_similarity is 65% or more, or if semantic similarity is recognized
        if (highest_similarity >= 0.65 or (best_match_answer and self.__semantic_similarity(filtered_query, best_match_answer))):
            print(f"\n{'\033[34m'}" + best_match_answer + f"{'\033[0m'}\n")
            if trainingMode:
                self.__expectation = input("Is this what you expected (Y/N): ")

                while not self.__expectation:
                    self.__expectation = input("Empty input is not acceptable. Is this what you expected (Y/N): ")

                if self.__expectation.lower() == "y":
                    print("Great!")
                    return
            else:
                return

        # only executes if training option is TRUE
        if (trainingMode):
            self.__expectation = input("I'm not sure. What was the expected response (training mode): ")  # train DLM

            while not self.__expectation:
                print("Nothing learnt. Moving on.")
                return

            self.__learn(filtered_query,
                         self.__expectation)  # learn this new question and answer pair and add to stored_data.txt
            print("I learned something new!")  # confirmation that it went through the whole process
        else:
            print(random.choice(self.__fallback_responses))
