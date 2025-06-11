# Dynamic-Learning Model (DLM) bot that learns how to respond to questions by learning from user input/expectations, as well as computationally solve arithmetics
import difflib
import string
import random
import spacy
import time
import sqlite3
import re
import nltk
from better_profanity import profanity
from nltk.corpus import names
from word2number import w2n


class DLM:
    __filename = None  # knowledge-base (SQL)
    __query = None  # user-inputted query
    __expectation = None  # trainer-inputted expected answer to query
    __category = None  # categorizes each question for efficient retrieval and basic NLG in SQL DB
    __nlp = None  # Spacy NLP analysis
    __tone = None  # sentimental tone of user query
    __trainingPwd = "371507"  # password to enter training mode
    __mode = None  # either "training", "commercial", or "experimental"
    __singlePassthrough = True  # used to prevent multiple iterations of training prompt
    __unsure_while_thinking = False  # if uncertain while thinking, then it will let the user know that
    __nlp_similarity_value = None  # saves the similarity value by doing SpaCy calculation (for debugging)
    __special_stripped_query = None  # saves query without any special words for reduced interference while vector calculating
    __nltk_names = set(name.lower() for name in names.words())

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

        # articles & determiners (words that don't add meaning to sentence)
        "the", "some", "any", "many", "every", "each", "either", "neither", "this", "that", "these", "those",
        "certain", "another", "such", "whatsoever", "whichever", "whomever", "whatever", "all", "something", "possible",

        # pronouns (general pronouns that don’t change meaning)
        "i", "me", "my", "mine", "here",
        "myself", "you", "your", "yours", "yourself", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves", "who", "whom", "whose", "which", "that",
        "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody", "people", "person",
        "whoever", "wherever", "whenever", "whosoever", "others", "oneself",

        # auxiliary (helping) verbs (do not contribute meaning)
        "get", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "best", "do",
        "does",
        "did", "doing", "shall", "should", "will", "would", "can", "could", "may", "might", "must", "bad", "dare",
        "need", "want",
        "used", "shallnt", "shouldve", "wouldve", "couldve", "mustve", "mightve", "mustnt", "good",

        # conjunctions (connectors that do not change meaning)
        "and", "but", "or", "gotten",
        "nor", "so", "for", "yet", "although", "though", "because", "since", "unless",
        "while", "whereas", "either", "neither", "both", "whether", "not", "if", "even if", "even though", "common",
        "as long as",
        "provided that", "whereas", "therefore", "thus", "hence", "meanwhile", "besides", "furthermore",

        # prepositions (location/relation words that are often unnecessary)
        "about", "above", "across", "after", "against", "along", "among", "around", "as", "at",
        "before", "behind", "below", "beneath", "beside", "between", "beyond", "by", "low", "high", "despite", "down",
        "during", "happen",
        "except", "for", "from", "in", "inside", "into",
        "like", "near", "off", "on", "onto", "out", "outside", "over", "past",
        "since", "through", "throughout", "till", "to", "toward", "under", "underneath",
        "until", "up", "upon", "with", "within", "without", "aside from", "concerning", "regarding",

        # common adverbs (time words and intensity words that add fluff)
        "way", "ways", "again", "already", "also", "always", "ever", "never", "just", "now", "often",
        "once", "only", "quite", "rather", "really", "seldom", "sometimes", "soon", "got",
        "still", "then", "there", "therefore", "thus", "too", "very", "well", "anytime",
        "hardly", "barely", "scarcely", "seriously", "truly", "frankly", "honestly", "basically", "literally",
        "definitely", "obviously", "surely", "likely", "probably", "certainly", "clearly", "undoubtedly",

        # question words (words that do not impact search meaning)
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "whichever", "whomever", "whenever", "wherever", "whosoever", "however", "whence",

        # informal/common fillers (spoken language fillers)
        "gonna", "wanna", "gotta", "lemme", "dunno", "kinda", "sorta", "aint", "ya", "yeah", "nah",
        "um", "uh", "hmm", "huh", "mmm", "uhh", "ahh", "err", "ugh", "tsk", "like", "okay", "ok", "alright",
        "yo", "bruh", "dude", "bro", "sis", "mate", "fam", "nah", "yup", "nope", "welp",

        # verbs commonly used in questions (but don’t change meaning)
        "go", "do", "dont", "does", "did", "can", "can't", "could", "couldnt", "should", "shouldnt", "shall", "will",
        "would", "wouldnt", "may", "might", "must", "use", "tell", "thinking",
        "please", "say", "let", "know", "consider", "find", "show", "take", "working",
        "list", "give", "provide", "make", "see", "mean", "understand", "point out", "stay", "look", "care", "work",

        # contracted forms (casual writing contractions),
        "ill", "im", "ive", "youd", "youll", "youre", "youve", "hed", "hell", "hes",
        "shed", "shell", "shes", "wed", "well", "were", "weve", "theyd", "theyll", "theyre", "theyve",
        "its", "thats", "whos", "whats", "wheres", "whens", "whys", "hows", "theres", "heres", "lets",

        # conversational fillers (unnecessary words in casual speech)
        "actually", "basically", "seriously", "literally", "obviously", "honestly", "frankly", "clearly",
        "apparently", "probably", "definitely", "certainly", "most", "mostly", "mainly", "typically", "essentially",
        "generally", "approximately", "virtually", "kind", "sort", "type", "whatever", "however",
        "you know", "i mean", "you see", "by the way", "sort of", "kind of", "more or less",
        "as far as i know", "in my opinion", "to be honest", "to be fair", "just saying",
        "at the end of the day", "if you ask me", "truth be told", "the fact is", "long story short",

        # internet slang, misspellings, and shortcuts
        "lol", "lmao", "rofl", "omg", "idk", "fyi", "btw", "imo", "smh", "afk", "ttyl", "brb",
        "thx", "pls", "ppl", "u", "ur", "r", "cuz", "coz", "cause", "gimme", "lemme", "wassup", "sup",

        # placeholder & non-descriptive words
        "thing", "stuff", "thingy", "whatchamacallit", "doohickey", "thingamajig", "thingamabob",

        # words that don’t add meaning
        "important", "necessary", "specific", "certain", "particular", "special", "exactly", "precisely",
        "recently", "currently", "today", "tomorrow", "yesterday", "soon", "later", "eventually", "sometime",

        # overused transitions
        "so", "then", "therefore", "thus", "anyway", "besides", "moreover", "furthermore", "meanwhile"
    ]

    # used for Chain-of-Thought (CoT) feature
    __exception_fillers = [
        "who", "whom", "whose", "what",
        "which", "when", "where", "why",
        "is", "are", "am", "was",
        "were", "do", "does", "did",
        "have", "has", "had", "can",
        "could", "will", "would", "shall",
        "should", "may", "might", "must",
        "show", "list", "give", "how", "i"
    ]

    # special words that the bot can mention while in "CoT"
    __special_exception_fillers = ["define", "explain", "describe", "compare", "calculate", "translate", "mean"]

    # advanced CoT computation identifiers
    __computation_identifiers = {
        # add
        "+": [
            "add", "plus", "sum", "total", "combined", "together",
            "in all", "in total", "more", "increased by", "gain",
            "got", "collected", "received", "add up", "accumulate",
            "bring to", "rise by", "grow by", "earned", "pick", "+"
        ],
        # subtract
        "-": [
            "subtract", "minus", "less", "difference", "left", "−",
            "remain", "remaining", "take away", "remove", "lost",
            "gave", "spent", "give away", "deduct", "decrease by",
            "fell by", "drop by", "leftover", "popped", "ate", "paid",
            "sold", "sells", "used", "use", "took", "absent", "broke off"
        ],
        # multiply
        "*": [
            "multiply", "times", "multiplied by", "product",
            "each", "every", "such", "per box", "per row", "per hour",
            "per week", "half", "double", "triple", "quadruple", "quartet", "twice as many",
            "thrice as many", "x", "such box", "*", "×"
        ],
        # divide
        "/": [
            "divide", "divided by", "split", "shared equally",
            "per", "share", "shared", "equal parts", "equal groups",
            "ratio", "quotient", "for each", "out of",
            "for every", "into", "average", "/", "÷"
        ],
        # convert
        "=": [
            "inch", "inches",
            "foot", "feet", "ft",
            "yard", "yards", "yd",
            "cm", "centimeter", "centimeters",
            "m", "meter", "meters",
            "mm", "millimeter", "millimeters",
            "week", "weeks",
            "second", "seconds", "minute", "minutes", "min",
            "hour", "hours", "day", "days", "month", "months",
            "year", "years", "yr",
            "km", "kilometer", "kilometers",
            "mile", "miles",
            "ml", "milliliter", "milliliters",
            "l", "liter", "liters",
            "mg", "milligram", "milligrams",
            "g", "gram", "grams",
            "kg", "kilogram", "kilograms",
            "lb", "pound", "pounds",
            "oz", "ounce", "ounces",
            "gallon", "gallons",
            "quart", "quarts",
            "pint", "pints",
            "cup", "cups",
            "dollar", "dollars",
            "cent", "cents",
            "penny", "pennies",
            "nickel", "nickels",
            "dime", "dimes",
            "quarter", "quarters"
        ]
    }

    # for SI conversion and CoT
    __units = {
        # distance units (base = meters)
        "inch": 0.0254, "inches": 0.0254,
        "foot": 0.3048, "feet": 0.3048, "ft": 0.3048,
        "yard": 0.9144, "yards": 0.9144, "yd": 0.9144,
        "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
        "m": 1.0, "meter": 1.0, "meters": 1.0,
        "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
        "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "mile": 1609.344, "miles": 1609.344,

        # time units (base = seconds)
        "second": 1.0, "seconds": 1.0,
        "minute": 60.0, "minutes": 60.0,
        "hour": 3600.0, "hours": 3600.0,
        "day": 86400.0, "days": 86400.0,
        "week": 604800.0, "weeks": 604800.0,
        "month": 2592000.0, "months": 2592000.0,
        "year": 31536000.0, "years": 31536000.0, "yr": 31536000.0,

        # mass units (base = kg)
        "mg": 0.000001, "milligram": 0.000001, "milligrams": 0.000001,
        "g": 0.001, "gram": 0.001, "grams": 0.001,
        "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
        "lb": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
        "oz": 0.0283495231, "ounce": 0.0283495231, "ounces": 0.0283495231,

        # volume units (base = liter)
        "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
        "l": 1.0, "L": 1.0, "liter": 1.0, "liters": 1.0,
        "gallon": 3.78541, "gallons": 3.78541,
        "quart": 0.946353, "quarts": 0.946353,
        "pint": 0.473176, "pints": 0.473176,
        "cup": 0.236588, "cups": 0.236588,

        # currency units (base = dollar)
        "dollar": 1.0, "dollars": 1.0,
        "cent": 0.01, "cents": 0.01,
        "penny": 0.01, "pennies": 0.01,
        "nickel": 0.05, "nickels": 0.05,
        "dime": 0.10, "dimes": 0.10,
        "quarter": 0.25, "quarters": 0.25
    }

    def __init__(self, db_filename):  # initializes SQL database & SpaCy NLP
        """
        Initialize the Dynamic-Learning Model (DLM) chatbot.

        Parameters:
            db_filename (str): The SQLite database file used to train and retrieve
                               question-answer-category triples.

        Behavior:
            - Loads the SpaCy NLP model ('en_core_web_lg').
            - Loads Better-Profanity for profane phrase sensing.
            - Connects to the specified SQLite database file.
            - Ensures the required table structure exists (creates if missing).
        """
        self.__nlp = spacy.load("en_core_web_lg")
        profanity.load_censor_words()
        self.__filename = db_filename
        self.__create_table_if_missing()

    def __create_table_if_missing(self):  # no return, void
        """
        Ensure the existence of the 'knowledge_base' table in the SQLite database; create or modify it if necessary.

        Behavior:
            - Establishes a connection to the SQLite database specified by self.__filename.
            - Creates the 'knowledge_base' table if it does not exist, with the following columns:
                - id (INTEGER, PRIMARY KEY, AUTOINCREMENT)
                - question (TEXT, NOT NULL, UNIQUE)
                - answer (TEXT, NOT NULL)
                - category (TEXT, NOT NULL)
            - If the table already exists but is missing the 'category' column, the method adds it with a default empty string.
            - Used exclusively within the class constructor to ensure the database schema is properly initialized.
        """
        conn = sqlite3.connect(self.__filename)
        c = conn.cursor()
        # Create table with identifier column if it doesn't exist
        c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            question    TEXT    NOT NULL UNIQUE,
            answer      TEXT    NOT NULL,
            category  TEXT    NOT NULL
        )
        """)
        # If the table existed already without identifier, add it now
        c.execute("PRAGMA table_info(knowledge_base)")
        cols = [row[1] for row in c.fetchall()]
        if 'category' not in cols:
            c.execute("""
            ALTER TABLE knowledge_base
            ADD COLUMN category TEXT NOT NULL DEFAULT ''
            """)
        conn.commit()
        conn.close()

    def __get_category(self, exact_question):  # returns category as a string
        """
        Retrieve the category (question type) associated with a specific question from the SQLite knowledge base.

        Parameters:
            exact_question (str): The exact question text used to search the database.

        Returns:
            str or None: The associated category if found (e.g., 'yesno', 'definition'); otherwise, None.

        Behavior:
            - Connects to the SQLite database.
            - Performs a lookup for the given question.
            - Returns the corresponding category tag if a match exists.
        """
        conn = sqlite3.connect(self.__filename)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT category FROM knowledge_base WHERE question = ?",
            (exact_question,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]  # this is the category/question_type
        else:
            return None  # question not found

    def __get_specific_question(self, exact_answer):  # returns question as a string
        """
        Retrieve the original question associated with a given answer from the SQLite knowledge base.

        Parameters:
            exact_answer (str): The exact answer text used to search the database.

        Returns:
            str or None: The corresponding question string if found; otherwise, None.

        Behavior:
            - Connects to the SQLite database.
            - Searches for a question where the answer matches exactly.
            - Returns the first matching question, or None if no match exists.
        """
        conn = sqlite3.connect(self.__filename)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question FROM knowledge_base WHERE answer = ?",
            (exact_answer,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return row[0]  # this is the category/question_type
        else:
            return None  # question not found

    @staticmethod
    # ANSI escape for moving the cursor up N lines
    def __move_cursor_up(lines):  # no return, void
        print(f"\033[{lines}A", end="")

    @staticmethod
    # loading animation for bot thought process
    def __loadingAnimation(user_input, duration):  # no return, void
        for seconds in range(0, 3):
            print(f"{'\033[33m'}\r{user_input}{'.' * (seconds + 1)}   {'\033[0m'}", end="", flush=True)
            time.sleep(duration)
        print("\n")

    def __filtered_input(self, userInput):  # returns filtered string
        """
        Filter out filler words from the user input while preserving important context.

        Parameters:
            userInput (str): The raw, lowercase-converted user query string.

        Returns:
            str: A filtered version of the input string with filler words removed and duplicates eliminated.

        Behavior:
            - Tokenizes the input into words.
            - Removes filler words unless:
                - It's the first word and part of the exception list.
                - The current mode is 'experimental' and the word is a computation keyword.
            - Preserves word order while removing duplicates.
            - Joins the remaining words back into a single filtered string.
        """
        # tokenize user input (split into words)
        words = userInput.lower().split()

        # remove filler words
        filtered_words = []
        for i, word in enumerate(words):
            word_lowered = word.lower()

            # allow exceptions ONLY in first position:
            if i == 0 and word_lowered in self.__exception_fillers:
                filtered_words.append(word)

            # otherwise, only keep non-fillers
            else:
                # In experimental mode, keep any computation keyword even if it's also a filler
                is_computation_kw = any(word_lowered == kw.lower()
                                        for kws in self.__computation_identifiers.values()
                                        for kw in kws)

                if word_lowered not in self.__filler_words or (self.__mode == "experimental" and is_computation_kw):
                    filtered_words.append(word)

        # remove duplicates while preserving order (numbers excluded)
        seen = set()
        unique_words = []
        for word in filtered_words:
            try:
                float(word)  # Try to treat as number
                unique_words.append(word)  # Keep numeric strings (duplicates allowed)
            except ValueError:
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)

        # join the remaining words back into a string
        return " ".join(unique_words)

    def __set_sentiment_tone(self, orig_input):  # no return, void
        """
        Analyze the original user input and assign an appropriate emotional tone.

        Parameters:
            orig_input (str): The raw, unfiltered user query.

        Behavior:
            - Detects aggressive language using profanity filtering.
            - Analyzes punctuation and casing to infer emotional tone such as:
                - 'angry aggressive' for profane content
                - 'angry frustrated' for all-uppercase text
                - 'angry confused' for combined "?" and "!"
                - 'angry excited' for "!" only
                - 'confused unclear' for "?" only
                - 'doubtful uncertain' for ellipses ("..." or "..")
            - Stores the result in self.__tone as a string label.
        """
        is_profane = profanity.contains_profanity(orig_input)
        if is_profane:
            self.__tone = "angry aggressive"
        elif orig_input == orig_input.upper():
            self.__tone = "angry frustrated"
        elif orig_input.__contains__("?") and orig_input.__contains__("!"):
            self.__tone = "angry confused"
        elif orig_input.__contains__("!"):
            self.__tone = "angry excited"
        elif orig_input.__contains__("?"):
            self.__tone = "confused unclear"
        elif orig_input.__contains__("...") or orig_input.__contains__(".."):
            self.__tone = "doubtful uncertain"
        else:
            self.__tone = ""

    def __perform_advanced_CoT(self, filtered_query):  # no return, void
        """
        Perform advanced Chain-of-Thought (CoT) reasoning to solve arithmetic or unit conversion problems.

        Parameters:
            filtered_query (str): The cleaned user input, expected to be a math- or logic-based question.

        Behavior:
            - Simulates step-by-step reasoning to solve arithmetic word problems without relying on memorized answers.
            - Extracts entities including person names, items, numbers, and operations using SpaCy, NLTK, and regex.
            - Detects arithmetic operations via lexical and semantic matching with predefined keyword sets.
            - Handles both numeric digits and text-based numbers (e.g., "three", "double").
            - Supports simple arithmetic expressions and unit conversions (e.g., inches to cm).
            - Prints the interpreted steps, logical inferences, and the final computed result with contextual explanations.
            - Displays fallback messages if the query is incomplete or too ambiguous to solve.
        """
        print(
            f"{'\033[33m'}I am presented with a more involved query asking me to do some form of computation{'\033[0m'}")
        self.__loadingAnimation("Let me think about this carefully and break it down so that I can solve it", 0.8)
        self.__loadingAnimation(
            f"I’ve trimmed away any extra words so I’m focusing on \"{filtered_query.title()}\" now", 0.8)
        persons_mentioned = []
        items_mentioned = []
        keywords_mentioned = []
        num_mentioned = []
        operands_mentioned = []
        arithmetic_ending_phrases = [
            "total", "all", "left", "leftover", "remaining", "altogether", "together", "each", "spend", "per",
            "sum", "combined", "add up", "accumulate", "bring to", "rise by", "grow by", "earned", "in all", "in total",
            "difference", "deduct", "decrease by", "fell by", "drop by", "ate",
            "multiply", "times", "product", "received", "pick", "paid", "gave", "pay",
            "split", "shared equally", "equal parts", "equal groups", "ratio", "quotient", "average", "out of", "into"
        ]
        filtered_query = filtered_query.title()
        doc = self.__nlp(filtered_query)

        # Have the bot pick out names mentioned (in order) using SpaCy and NLTK (for maximum coverage)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                cleaned = re.sub(r'\d+', "", ent.text).strip()
                if cleaned:
                    persons_mentioned.append(cleaned)

        tokens = nltk.word_tokenize(filtered_query)
        for tok in tokens:
            cleaned = re.sub(r"[^a-zA-Z]", "", tok).lower()
            if cleaned in self.__nltk_names:
                persons_mentioned.append(cleaned.capitalize())
        persons_mentioned = {name for name in set(persons_mentioned) if len(name.split()) == 1}
        persons_mentioned = set(persons_mentioned)

        # Have the bot pick out item names (in order) using SpaCy
        for token in doc:
            if token.pos_ == "PROPN":
                cleaned = re.sub(r'\d+', "", token.text).strip()
                if cleaned and cleaned not in persons_mentioned:
                    items_mentioned.append(cleaned)
        items_mentioned = set(items_mentioned)

        # Now have the bot pick out numbers (in order)
        re_pattern = re.compile(r"\d+(\.\d+)?")
        for match in re_pattern.finditer(filtered_query):
            num_mentioned.append(float(match.group(0)).__str__())

        tokens_lower = filtered_query.lower().split()
        last_two = set(tokens_lower[-2:])  # only the final 2 words from filtered input

        # Then have it find all operand indicating keywords
        found_operand = False
        for fq in filtered_query.split():
            fq_l = fq.lower()
            # If this word is one of the ending phrases and sits among the last five, skip it
            if fq_l in arithmetic_ending_phrases and fq_l in last_two:
                continue
            if fq_l in {"+", "-", "*", "/"}:
                operands_mentioned.append(fq_l)
                continue  # move on to the next token
            for operand, keywords in self.__computation_identifiers.items():
                for kw in keywords:
                    p1 = self.__nlp(kw)
                    p2 = self.__nlp(fq_l)
                    word_num_surrounded = re.search(rf'\d+\s*{fq.lower()}\s*\d+', filtered_query.lower())

                    # Direct match or lemma match
                    if (kw.lower() == fq.lower()) or p1[0].lemma_ == p2[0].lemma_:
                        keywords_mentioned.append(kw.title())
                        if kw.lower() == "average":
                            operands_mentioned.append("+")
                        elif kw.lower() == "out of":
                            if word_num_surrounded:
                                operands_mentioned.append(operand)
                                found_operand = True
                                break  # only break if 'out of' condition is satisfied
                            continue  # skip adding 'out of' if not surrounded by numbers
                        else:
                            operands_mentioned.append(operand)
                            found_operand = True
                            break

                    # Vector + string similarity
                    if p1.vector_norm != 0 and p2.vector_norm != 0 and (
                            p1.similarity(p2) > 0.80 and difflib.SequenceMatcher(None, kw, fq_l).ratio() > 0.40):
                        keywords_mentioned.append(kw.title())
                        if kw.lower() == "average":
                            operands_mentioned.append("+")
                        elif kw.lower() == "out of":
                            if word_num_surrounded:
                                operands_mentioned.append(operand)
                                found_operand = True
                                break
                            continue
                        else:
                            operands_mentioned.append(operand)
                            found_operand = True
                            break

                    # Fallback: high string similarity
                    elif difflib.SequenceMatcher(None, kw, fq_l).ratio() > 0.80:
                        keywords_mentioned.append(kw.title())
                        if kw.lower() == "average":
                            operands_mentioned.append("+")
                        elif kw.lower() == "out of":
                            if word_num_surrounded:
                                operands_mentioned.append(operand)
                                found_operand = True
                                break
                            continue
                        else:
                            operands_mentioned.append(operand)
                            found_operand = True
                            break

                if found_operand:
                    found_operand = False
                    break

        # If no operands were found in the main pass, check ending phrases as a last resort
        if not operands_mentioned:
            for fq in filtered_query.split():
                p_fq = self.__nlp(fq)

                # Replace exact matching with a spaCy similarity check against ending_phrases
                matched_ep = None
                for ep in arithmetic_ending_phrases:
                    p_ep = self.__nlp(ep)
                    if p_ep.vector_norm != 0 and p_fq.vector_norm != 0 and p_ep.similarity(p_fq) > 0.50:
                        matched_ep = ep
                        break

                if not matched_ep:
                    continue

                # Now matched_ep roughly corresponds to an ending phrase; find its operand via spaCy
                for operand, keywords in self.__computation_identifiers.items():
                    for kw in keywords:
                        p_kw = self.__nlp(kw)
                        if p_kw.vector_norm != 0 and p_fq.vector_norm != 0 and p_kw.similarity(p_fq) > 0.70:
                            keywords_mentioned.append(kw.title())
                            operands_mentioned.append(operand)
                            break
                    if operands_mentioned:
                        break
                if operands_mentioned:
                    break
        keywords_mentioned = list(dict.fromkeys(keywords_mentioned))
        # additionally, "double" "triple" "quadruple" also count as numbers, in addition to text numbers (e.g. "three")
        text_nums = ["a", "an", "half", "double", "triple", "quadruple"]
        a_an_detected = False
        for match in filtered_query.lower().split():
            try:
                num = w2n.word_to_num(match)
                if str(float(num)) not in num_mentioned:
                    num_mentioned.append(str(float(num)))
                continue
            except ValueError:
                pass
            for t in text_nums:
                p1 = self.__nlp(match)
                p2 = self.__nlp(t)
                if p1[0].lemma_ == p2[0].lemma_:
                    if t == "double":
                        num_mentioned.append(float(2).__str__())
                    elif t == "triple":
                        num_mentioned.append(float(3).__str__())
                    elif t == "half":
                        num_mentioned.append(float(0.5).__str__())
                    elif ("=" in operands_mentioned) and (t == "a" or t == "an"):
                        a_an_detected = True
                        num_mentioned.append(float(1.0).__str__())
                    elif t == "quadruple":
                        num_mentioned.append(float(4).__str__())

        if a_an_detected and len(num_mentioned) > 1:
            num_mentioned.remove("1.0")
        if ('=' in operands_mentioned) and (len(num_mentioned) < 2):
            operands_mentioned.clear()
            operands_mentioned.append('=')
        else:
            if '=' in operands_mentioned:
                operands_mentioned = [op for op in operands_mentioned if op != '=']
        operands_mentioned = list(dict.fromkeys(operands_mentioned))

        print("\n")
        if any(not lst for lst in (num_mentioned, operands_mentioned)) or (
                '=' not in operands_mentioned and num_mentioned.__len__() < 2):  # don't compute if parts are missing
            print(
                f"{self.__loadingAnimation('Hmm', 0.8) or ''}{'\033[34m'}It looks like some essential details are missing, so I can’t complete this calculation right now.{'\033[0m'}")
        else:  # else, the bot needs to explain what it has tokenized
            self.__loadingAnimation(
                f"1.) I see {', '.join(persons_mentioned) if persons_mentioned.__len__() >= 1 else 'no one'} mentioned as a person name; "
                f"{'they’re likely key to this problem' if persons_mentioned.__len__() >= 1 else 'moving on'}", 0.2)
            self.__loadingAnimation(
                f"2.) Moreover, I see {', '.join(items_mentioned) if items_mentioned.__len__() >= 1 else 'no items'} mentioned as proper nouns; "
                f"{'this might be a key thing to this problem' if items_mentioned.__len__() >= 1 else 'moving on'}",
                0.2)
            self.__loadingAnimation(
                f"3.) I’ve also identified the numbers {' and '.join(num_mentioned)} that I need to compute with", 0.2)
            self.__loadingAnimation(
                f"4.) I see that I need to perform a \"{'\" and \"'.join(operands_mentioned)}\" operation for this query; I’ll use that to guide my calculation",
                0.2)
            self.__loadingAnimation("Now I have the parts, so let me put it all together and solve", 0.3)
            # Finally compute it and then give the response (if there is any)

            # move "originally" numbers to the front
            indicators = {"original", "originally", "initial", "initially", "at first", "to begin with", "had",
                          "savings", "saving", "of"}

            tokens = filtered_query.split()
            temp = None
            # lowercase copy for matching
            lower_tokens = [t.lower() for t in tokens]

            for idx, token in enumerate(lower_tokens):
                if token in indicators:
                    # check token before
                    if idx > 0 and token != "of":
                        candidate = lower_tokens[idx - 1]
                        try:
                            temp = (w2n.word_to_num(candidate))
                        except ValueError:
                            pass
                    # check token after
                    if idx < len(tokens) - 1:
                        candidate = lower_tokens[idx + 1]
                        try:
                            temp = (w2n.word_to_num(candidate))
                        except ValueError:
                            pass
            if temp is not None:
                if str(float(temp)) in num_mentioned:
                    num_mentioned.remove(str(float(temp)))
                num_mentioned.insert(0, str(float(temp)))

            # conversion problem
            if len(num_mentioned) == 1 and len(operands_mentioned) == 1:
                try:
                    tokens = filtered_query.lower().split()
                    num0 = float(num_mentioned[0])
                    num_idx = None

                    # redefine text_nums to be a dictionary instead
                    text_nums = {
                        "a": 1.0,
                        "an": 1.0,
                        "half": 0.5,
                        "double": 2.0,
                        "triple": 3.0,
                        "quadruple": 4.0
                    }

                    # Find index of the numeric token (either digit or w2n‐convertible)
                    for i, tok in enumerate(tokens):
                        lower_tok = tok.lower()

                        # Check if tok is one of the special words
                        if lower_tok in text_nums:
                            if text_nums[lower_tok] == num0:
                                num_idx = i
                                break
                            else:
                                continue  # skip parsing this token further

                        # Otherwise try parsing as a standard float
                        try:
                            if float(tok) == num0:
                                num_idx = i
                                break
                        except ValueError:
                            # If that fails, try converting via w2n.word_to_num
                            try:
                                if float(w2n.word_to_num(tok)) == num0:
                                    num_idx = i
                                    break
                            except ValueError:
                                continue

                    source_key = None
                    target_key = None

                    # Look for the first unit immediately after the number for source key
                    if num_idx is not None:
                        for tok in tokens[num_idx + 1:]:
                            for key, val in self.__units.items():
                                p1 = self.__nlp(tok)
                                p2 = self.__nlp(key)
                                if p1[0].lemma_ == p2[0].lemma_:
                                    source_key = key
                                    break
                            if source_key:
                                break

                    # 3) Now scan the entire sentence for the target-key
                    for tok in tokens:
                        for key, val in self.__units.items():
                            p1 = self.__nlp(tok)
                            p2 = self.__nlp(key)
                            p3 = self.__nlp(source_key)
                            if (p1[0].lemma_ == p2[0].lemma_) and (p2[0].lemma_ != p3[0].lemma_):
                                target_key = key
                                break
                        if target_key:
                            break

                    # 4) Compute only if we have both source_key and target_key
                    if source_key and target_key:
                        result = (num0 * self.__units[source_key]) / self.__units[target_key]
                        self.__loadingAnimation(
                            f"I need to take {num0} and multiply it by {self.__units[source_key]}. Finally, I divide by {self.__units[target_key]} and I got my answer",
                            0.2)
                        expr = f"{num_mentioned[0]} {source_key}(s) ==> {round(result, 2)} {target_key}(s)"
                        print(f"{'\033[34m'}Conversion Answer: {expr} {'\033[0m'}")
                    else:
                        print(f"{'\033[33m'}Could not identify both source and target units.{'\033[0m'}")
                except SyntaxError:
                    print("\033[33mOops! I still mix up conversions and arithmetic sometimes. Working on it!\033[0m")
            # regular arithmetic operations
            elif len(num_mentioned) >= 2 and (
                    len(operands_mentioned) == (len(num_mentioned) - 1) or len(operands_mentioned) == 1):
                # Build a string like "n0 op0 n1 op1 n2 ... op_{N-2} n_{N-1}"
                parts = []
                for i, num in enumerate(num_mentioned):
                    parts.append(str(num))
                    if i < (len(num_mentioned) - 1) and ("average" in filtered_query.lower()):
                        parts.append("+")
                    elif i < (len(num_mentioned) - 1) and (len(operands_mentioned) == 1):
                        parts.append(operands_mentioned[0])
                    elif i < len(operands_mentioned):
                        parts.append(operands_mentioned[i])
                expr = " ".join(parts)

                try:
                    result = eval(expr)
                    if "average" in filtered_query.lower():
                        expr = "(" + expr + ") / " + str(len(num_mentioned))
                        result /= len(num_mentioned)
                    print(f"{'\033[34m'}Arithmetic Answer: {expr} = {result}{'\033[0m'}")
                except SyntaxError:
                    print(
                        f"{'\033[34mAh'}, something about that stumped me. I’ll need to learn more to handle it properly.{'\033[0m'}")
            else:
                print(f"{'\033[34m'}{random.choice(self.__fallback_responses)}{'\033[0m'}")
                print(
                    f"{'\033[34m'}However, while I was trying to understand the math, I ran into \"{'" and "'.join(keywords_mentioned)}\", which I use to connect keywords to math operations.{'\033[0m'}")
                print(
                    f"{'\033[34m'}That might've confused me a bit, maybe try leaving one of those out or rephrase it to make it clearer?{'\033[0m'}")

    def __generate_thought(self, filtered_query, best_match_question, best_match_answer,
                           highest_similarity):  # no return, void
        """
        Simulate a Chain-of-Thought (CoT) reasoning process by printing the bot's internal analysis.

        Parameters:
            filtered_query (str): The cleaned version of the user's question, stripped of filler or trigger words.
            best_match_question (str): The closest matching question found in the knowledge base.
            best_match_answer (str): The corresponding answer to the matched question.
            highest_similarity (float): The calculated string similarity score (0 to 1) for the match.

        Behavior:
            - Outputs step-by-step reasoning in a conversational format (e.g., interpreting the question's structure and tone).
            - In experimental mode, calls advanced reasoning (e.g., math parsing or CoT decomposition).
            - Identifies the question's tone, topic, and potential intent based on interrogative words and SpaCy similarity.
            - Displays confidence based on similarity metrics and sets flags for uncertain answers.
            - Uses colorized terminal output and a loading animation to simulate reflective thought.
        """
        print("\nThought Process (Yellow):")
        if filtered_query is None or filtered_query == "":
            print(
                f"{'\033[33m'}I couldn't pick out any context or clear topic. If I see a match in my database I will respond with that, or else I have no clue!{'\033[0m'}")
        else:
            sentiment_tone = self.__tone.split()

            if self.__tone != "":
                print(
                    f"{'\033[33m'}Right off the bat, the user seems quite {sentiment_tone[0]} or {sentiment_tone[1]} by their query tone. Hopefully I won't disappoint!{'\033[0m'}")
            if self.__mode == "experimental":
                self.__perform_advanced_CoT(filtered_query)
            else:
                interrogative_start = filtered_query.split()[0]
                identifier = filtered_query
                special_start = ["definition", "explanation", "description", "comparison", "calculation", "translation",
                                 "meaning"]  # special word in different form
                for word in special_start:
                    identifier = identifier.replace(word, "")
                # collapse any extra spaces
                identifier = " ".join(identifier.split())
                identifier = identifier.split()

                if " ".join(identifier) == "":
                    print(
                        f"{'\033[33m'}The user starts their query with \"{interrogative_start.title()}\", but I couldn't pick out a clear topic or context.{'\033[0m'}")
                else:
                    print(
                        f"{'\033[33m'}The user starts their query with \"{interrogative_start.title()}\" and they are asking about \"{" ".join(identifier).title()}\".{'\033[0m'}")
                self.__loadingAnimation("Let me think about this carefully", 0.8)

                for s in special_start:
                    for u in filtered_query.split():
                        s_input = self.__nlp(s)
                        u_input = self.__nlp(u)
                        if (s_input.vector_norm != 0 and u_input.vector_norm != 0) and (
                                s_input.similarity(u_input) > 0.60):
                            print(
                                f"{'\033[33m'}It seems like they want a {s} of \"{" ".join(identifier).title()}\".{'\033[0m'}")

                self.__semantic_similarity(self.__special_stripped_query, best_match_question)
                spacy_proceed = self.__nlp_similarity_value is not None
                if (best_match_answer is None) or (highest_similarity < 0.65 and (spacy_proceed and self.__nlp_similarity_value < 0.85)):
                    print(f"{'\033[33m'}The closest match is only {int(highest_similarity * 100)}% similar when I used sequence matching.{'\033[0m'}")
                    if spacy_proceed:
                        print(f"{'\033[33m'}Furthermore, an in-depth vector analysis revealed a similarity percentage of {int(self.__nlp_similarity_value * 100)}%.{'\033[0m'}")
                    print(f"{self.__loadingAnimation("Hmm", 0.8) or ''}{'\033[33m'}I don't think I know the answer, so I am going to let the user know that.{'\033[0m'}")
                    self.__unsure_while_thinking = True
                else:
                    self.__unsure_while_thinking = False
                    DB_identifier = self.__get_specific_question(best_match_answer)
                    print(f"{'\033[33m'}Yes! I do remember learning about \"{DB_identifier}\" and I might have the right answer!")
                    print(f"This is because when I did a sequence similarity calculation to one of the closest match in my database, I found it to be {int(highest_similarity * 100)}% similar.")
                    if spacy_proceed:
                        print(f"Additionally, doing a more in-depth vector NLP analysis resulted in {int(self.__nlp_similarity_value * 100)}% similarity. Although there are room for error, we will see.{'\033[0m'}")
                    self.__loadingAnimation("Let me recall that answer", 0.8)
        print("\n")

    def __generate_response(self, best_match_answer, best_match_question):  # no return, void
        """
        Generate a dynamic natural language response based on the answer's category.

        Parameters:
            best_match_answer (str): The stored answer retrieved from the knowledge base.
            best_match_question (str): The matched user question used to derive category context.

        Behavior:
            - Determines the question's category using internal tagging (e.g., 'yesno', 'process', etc.).
            - Selects a category-specific response template to simulate Natural Language Generation (NLG).
            - Reformats the answer with human-like phrasing and prints it in stylized terminal output.
            - Handles special formatting for categories like definitions, processes, and deadlines.
            - Gracefully handles unrecognized categories or missing data.
        """
        identifier = self.__get_category(best_match_question)
        BLUE = '\033[34m'
        RESET = '\033[0m'

        if identifier is None:
            print("Sorry, I encountered an error on my end. Please try again later.")
            return

        if identifier == "generic":
            print(f"\n{BLUE}{best_match_answer}{RESET}\n")

        elif identifier == "yesno":
            affirmative_templates = [
                "Yes, {}",
                "Absolutely, {}",
                "Certainly, {}",
                "Indeed, {}",
                "That's right, {}",
                "Correct, {}",
                "You got it, {}",
                "Sure thing, {}",
                "Of course, {}",
                "Definitely, {}",
                "Without a doubt, {}",
                "That's true, {}",
                "Affirmative, {}",
                "Right on, {}",
                "You're spot on, {}",
                "Exactly, {}",
                "Totally, {}",
                "No question about it, {}",
                "100%, {}",
                "I agree, {}"
            ]
            negative_templates = [
                "No, {}",
                "Not at all, {}",
                "Unfortunately, {}",
                "Of course not, {}",
                "That's not correct, {}",
                "Actually, no, {}",
                "I'm afraid not, {}",
                "Nope, {}",
                "Sorry, but no, {}",
                "That’s not the case, {}",
                "Negative, {}",
                "Not quite, {}",
                "That’s incorrect, {}",
                "I'm sorry, {}",
                "Absolutely not, {}",
                "Nah, {}",
                "Doesn’t seem so, {}",
                "I wouldn't say that, {}",
                "No way, {}",
                "That’s a no, {}"
            ]

            ans = best_match_answer.strip().lower()
            if ans.startswith(("no", "not", "don't", "do not", "never", "cannot")):
                template = random.choice(negative_templates)
                # remove instances of "negative" words to remove redundancy
                if ans.__contains__("no, "):
                    best_match_answer = best_match_answer.replace("no, ", "", 1)
                else:
                    best_match_answer = best_match_answer.replace("no ", "", 1)
            else:
                template = random.choice(affirmative_templates)
                # remove instances of "affirmative" words to remove redundancy
                if ans.__contains__("yes, "):
                    best_match_answer = best_match_answer.replace("yes, ", "", 1)
                else:
                    best_match_answer = best_match_answer.replace("yes ", "", 1)
            response = template.format(best_match_answer)
            print(f"\n{BLUE}{response}{RESET}\n")

        elif identifier == "process":  # when training, make sure there are only 3 steps for "process"
            templates = [
                "To get started, {}. Then, {}. Finally, {}",
                "First, {}. Next, {}. Lastly, {}",
                "Begin by {}. After that, {}. Don't forget to {}.",
                "Start with {}. Continue by {}. Finish by {}.",
                "Initially, {}. Then proceed to {}. End with {}.",
                "Kick things off by {}. Follow it up with {}. Conclude by {}.",
                "Your first step is to {}. The second step is to {}. The final step is to {}.",
                "Commence by {}. Subsequently, {}. Ultimately, {}.",
                "Start off by {}. Then move on to {}. Finally, make sure you {}.",
                "Begin with {}. Then take care of {}. Lastly, ensure you {}."
            ]
            steps = best_match_answer.split("; ")  # steps must be separated by a semicolon
            response = random.choice(templates).format(*steps[:3])
            print(f"\n{BLUE}{response}{RESET}\n")

        elif identifier == "definition":
            # extract just the term by filtering out common definition triggers
            raw = best_match_question  # e.g. "what definition fafsa"
            triggers = {
                "what", "definition", "define", "meaning", "interpret",
                "what's", "whats", "what is", "what does", "mean", "means",
                "could", "you", "explain", "describe", "clarify", "tell",
                "me", "give", "the", "of", "in", "other", "words"
            }
            # split on whitespace, drop any trigger words (case‐insensitive)
            term_words = [w for w in raw.split() if w.lower() not in triggers]
            term = " ".join(term_words).strip()

            templates = [
                "\"{0}\" refers to {1}",
                "By definition, \"{0}\" is {1}",
                "In simple terms, \"{0}\" means {1}",
                "\"{0}\" can be described as {1}",
                "The term \"{0}\" stands for {1}",
                "Essentially, \"{0}\" is {1}",
                "\"{0}\" is understood as {1}",
                "In other words, \"{0}\" is {1}",
                "To put it simply, \"{0}\" refers to {1}",
                "\"{0}\" typically means {1}",
                "When we say \"{0}\", we’re talking about {1}",
                "\"{0}\" represents {1}",
                "\"{0}\" is defined as {1}",
                "You can think of \"{0}\" as {1}"
            ]
            response = random.choice(templates).format(term, best_match_answer)
            print(f"\n{BLUE}{response}{RESET}\n")

        elif identifier == "deadline":
            raw = best_match_question
            triggers = {
                "when", "what", "what's", "whats", "when's", "whens",
                "is", "the", "a", "an",
                "deadline", "due", "due date", "cutoff", "closing", "closing date",
                "by", "before", "until",
                "date", "day", "last", "latest", "final", "damn"
            }
            words = raw.split()
            term_words = [w for w in words if w.lower() not in triggers]
            term = " ".join(term_words).strip()

            templates = [
                "The deadline for \"{0}\" is {1}",
                "You need to submit \"{0}\" by {1}",
                "Make sure to complete \"{0}\" by {1}",
                "\"{0}\" is due on {1}",
                "Don’t forget, \"{0}\" must be done by {1}",
                "\"{0}\" has a due date of {1}",
                "Be sure to finish \"{0}\" before {1}",
                "Please submit \"{0}\" no later than {1}",
                "\"{0}\" needs to be turned in by {1}",
                "The final date to complete \"{0}\" is {1}",
                "Submission for \"{0}\" closes on {1}",
                "You have until {1} to complete \"{0}\"",
                "\"{0}\" is expected to be submitted by {1}",
                "\"{0}\" must be handed in by {1}",
                "The cutoff for \"{0}\" is {1}"
            ]
            response = random.choice(templates).format(term, best_match_answer)
            print(f"\n{BLUE}{response}{RESET}\n")

        elif identifier == "location":
            templates = [
                "You can find it at {0}",
                "It’s located at {0}",
                "Head over to {0} for more information",
                "Check it out at {0}",
                "Access it via {0}",
                "You’ll find it here: {0}",
                "It’s available at {0}",
                "Navigate to {0} to view it",
                "You can reach it at {0}",
                "Visit {0} to learn more",
                "Take a look at {0}",
                "More details can be found at {0}",
                "For further info, go to {0}",
                "To see it yourself, just go to {0}"
            ]
            best_match_answer = best_match_answer.lower()
            response = random.choice(templates).format(best_match_answer)
            print(f"\n{BLUE}{response}{RESET}\n")

        elif identifier == "eligibility":
            templates = [
                "Eligibility means {0}",
                "Eligibility requires that {0}",
                "Qualifications are met only if {0}",
                "To be eligible, {0}",
                "Meeting eligibility involves {0}",
                "You qualify only if {0}",
                "Eligibility is based on whether {0}",
                "In order to qualify, {0}",
                "You are eligible when {0}",
                "The requirements are satisfied if {0}",
                "Eligibility depends on {0}",
                "To meet the qualifications, {0}",
                "Being eligible implies that {0}",
                "You're considered eligible if {0}",
                "Eligibility conditions include {0}"
            ]
            best_match_answer = best_match_answer.lower()
            response = random.choice(templates).format(best_match_answer)
            print(f"\n{BLUE}{response}{RESET}\n")

        else:
            print("Cannot retrieve and generate response due to data in unfamiliar category. Please try again later.")

    def __semantic_similarity(self, userInput, knowledgebaseData):  # returns True/False
        """
        Evaluate semantic similarity between user input and a stored question using SpaCy vectors.

        Parameters:
            userInput (str): The cleaned or filtered user query.
            knowledgebaseData (str): A question stored in the knowledge base to compare against.

        Returns:
            bool: True if semantic similarity exceeds 0.50 threshold, False otherwise.

        Behavior:
            - Uses SpaCy's vector-based similarity to compare both texts.
            - Saves the similarity score internally for optional debugging or reporting.
        """
        UI_doc = self.__nlp(userInput)
        KB_doc = self.__nlp(knowledgebaseData)
        if UI_doc.vector_norm != 0 and KB_doc.vector_norm != 0:
            self.__nlp_similarity_value = UI_doc.similarity(KB_doc)
            return self.__nlp_similarity_value > 0.50
        else:
            return False

    def __learn(self, expectation, category):  # no return, void
        """
        Store a new question-answer-category entry in the SQLite knowledge base.

        Parameters:
            expectation (str): The expected answer or response to the current user query.
            category (str): The type of question (e.g., 'yesno', 'definition', 'process', etc.).

        Behavior:
            - Inserts the current stripped user query, along with its answer and category,
              into the SQLite database.
            - Uses 'INSERT OR IGNORE' to prevent duplicate entries.
        """
        conn = sqlite3.connect(self.__filename)
        c = conn.cursor()
        c.execute(
            "INSERT OR IGNORE INTO knowledge_base (question, answer, category) VALUES (?, ?, ?)",
            (self.__special_stripped_query, expectation, category)
        )
        conn.commit()
        conn.close()

    def __login_verification(self, mode):  # no return, void
        """
        Verify and initialize the selected access mode (Training, Commercial, or Experimental).

        Parameters:
            mode (str): The access mode. Options are:
                        't' - Training mode (requires password and shows training guidelines)
                        'c' - Commercial mode (no training features, read-only usage)
                        'e' - Experimental mode (enables reasoning features, no DB writes)

        Behavior:
            - If mode is 't', prompts for a password and displays mandatory training instructions.
            - If mode is 'c', enters Commercial mode without training privileges.
            - If mode is 'e', proceeds with Experimental mode with reasoning capabilities.
        """
        if mode.lower() == "t":
            password = input("Enter the password to enter Training Mode: ")
            while password != self.__trainingPwd:
                password = input(
                    "Password is incorrect, try again or type 'stop' to enter in commercial mode instead: ")
                if password.lower() == "stop":
                    self.__mode = "commercial"
                    print("\n")
                    self.__loadingAnimation("Logging in as Commercial User", 0.6)
                    print("\n")
                    break
            if password == self.__trainingPwd:
                # trainers must understand these rules as DLM can generate bad responses if these instructions are neglected
                print(
                    f"\n\n{'\033[31m'}MAKE SURE TO UNDERSTAND THE FOLLOWING ANSWER FORMAT EXPECTED FOR EACH CATEGORY FOR THE BOT TO LEARN ACCURATELY:{'\033[0m'}\n")
                print("*'yesno': Make sure to start your answer responses with \"yes\" or \"no\" ONLY")
                print(
                    "*'process': Each answer must have three steps for your responses, separated by \";\" (semicolon)")
                print(
                    "*'definition': Make sure to not mention the WORD/PHRASE to be defined & always start your response here with \"the\" only")
                print("*'deadline': Only include the deadline date, as an example, \"March 31st 2025\"")
                print("*'location': Mention the location only, nothing else. For example, \"The FAFSA.Gov website\"")
                print("*'generic': Format doesn't matter for this, give your answer in any comprehensive format")
                print(
                    "*'eligibility': Make sure to ONLY start the response with a pronoun like \"you\", \"they\", \"he\", \"she\", etc\n\n")

                confirmation = input(
                    "Make sure to understand and note these instructions somewhere as the generated responses would get corrupt otherwise.\nType 'Y' if you understood: ")
                while confirmation.lower() != "y":  # trainers must understand the instructions above
                    confirmation = input(
                        "You cannot proceed to train without understanding the instructions aforementioned. Type 'Y' to continue: ")
                self.__mode = "training"
                print("\n")
                self.__loadingAnimation("Logging in as Trainer", 0.6)
                print("\n")
        elif mode.lower() == "c":
            self.__mode = "commercial"
            self.__loadingAnimation("Logging in as Commercial User", 0.6)
            print("Ask Non-Computational Queries (switch to experimental for computational queries)")
        else:
            self.__mode = "experimental"
            self.__loadingAnimation("Logging in as Experimental", 0.6)
            print("Ask Computational Problems (Arithmetics and Conversions)")

    def ask(self, mode):  # no return, void
        """
        Handle a full user interaction loop with the DLM bot.
        NOTICE: To make the bot run continuously, implement a loop in your program

        Parameters:
            mode (str): The access mode. Options:
                        't' for Training mode,
                        'c' for Commercial mode,
                        'e' for Experimental mode.

        Behavior:
            - Prompts the user for input.
            - Detects tone, filters input, searches knowledge base.
            - Performs Chain-of-Thought (CoT) while recalling learnt answer.
            - If match is found, generates a response.
            - If in training mode and answer is incorrect or not found, prompts user to teach the bot.
            - In experimental mode, performs reasoning or arithmetic without using database.
        """
        if self.__singlePassthrough:
            self.__login_verification(mode)
            self.__singlePassthrough = False

        if self.__mode == "training":
            print("\nTRAINING MODE")
        elif self.__mode == "commercial":
            print("\n\nCOMMERCIAL MODE")
        else:
            print("\n\nEXPERIMENTAL MODE")  # for experimental, there are no data saving in DB
        self.__query = input("DLM Bot here, ask away: ")

        while self.__query is None or self.__query == "":
            self.__query = input("Empty input is unacceptable. Please enter something: ")

        self.__set_sentiment_tone(self.__query)  # sets global variable sentiment tone

        # storing the user-query (filtered, lower-case, no punctuation)
        if self.__mode == "experimental":
            # We want to keep the following
            keep = {".", "+", "-", "*", "/", "="}
            to_remove = "".join(ch for ch in string.punctuation if ch not in keep)
        else:
            to_remove = string.punctuation

        translation_table = str.maketrans("", "", to_remove)
        filtered_query = self.__filtered_input(
            self.__query.lower().translate(translation_table)
        )

        # match_query is the query without special words to prevent interference with SpaCy similarity
        self.__special_stripped_query = filtered_query
        special_exceptions = ["definition", "explanation", "description", "comparison", "calculation", "translation",
                              "meaning"]
        for word in special_exceptions:
            self.__special_stripped_query = self.__special_stripped_query.replace(word, "")
        # collapse any extra spaces
        self.__special_stripped_query = " ".join(self.__special_stripped_query.split())

        conn = sqlite3.connect(self.__filename)
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer FROM knowledge_base")
        rows = cursor.fetchall()
        conn.close()

        highest_similarity = 0.0
        best_match_question = None
        best_match_answer = None

        for stored_question, stored_answer in rows:
            # compare against both versions of the user query
            sim_stripped = difflib.SequenceMatcher(None, stored_question, self.__special_stripped_query).ratio()
            sim_filtered = difflib.SequenceMatcher(None, stored_question, filtered_query).ratio()
            # pick the higher
            sim = max(sim_stripped, sim_filtered)

            if sim > highest_similarity:
                highest_similarity = sim
                best_match_question = stored_question
                best_match_answer = stored_answer

        # "Chain of Thought" (CoT) Feature
        self.__generate_thought(filtered_query, best_match_question, best_match_answer, highest_similarity)

        # accept a match if highest_similarity is 65% or more, or if semantic similarity is recognized
        if self.__mode != "experimental":
            if (not self.__unsure_while_thinking) and ((highest_similarity >= 0.65) or (
                    best_match_answer and self.__semantic_similarity(self.__special_stripped_query,
                                                                     best_match_question))):
                self.__unsure_while_thinking = False  # reset this back to default for next iteration
                self.__generate_response(best_match_answer, best_match_question)
                if self.__mode == "training":
                    self.__expectation = input("Is this what you expected (Y/N): ")

                    while not self.__expectation:  # if nothing entered, ask until question answered
                        self.__expectation = input("Empty input is unacceptable. Is this what you expected (Y/N): ")

                    if self.__expectation.lower() == "y":
                        print("Great!")
                        return
                else:
                    return

            # only executes if training option is TRUE
            if self.__mode == "training":
                self.__expectation = input(
                    "I'm not sure. Train me with the expected response: ")  # train DLM with answer
                while not self.__expectation:
                    print("Nothing learnt. Moving on.")
                    return
                self.__category = input(
                    "Which category does that question/answer belong to (yesno, process, definition, deadline, location, generic, eligibility): ").lower()

                # used for generated response template
                category_options = ["yesno", "process", "definition", "deadline", "location", "generic", "eligibility"]

                while not self.__category or self.__category not in category_options:
                    self.__category = input("You MUST give an appropriate category for the question/answer: ").lower()

                self.__learn(self.__expectation,
                             self.__category)  # learn this new question and answer pair and add to knowledgebase
                print("I learned something new!")  # confirmation that it went through the whole process
            else:  # only executes when in commercial mode and bot cannot find the answer
                print(f"{'\033[34m'}{random.choice(self.__fallback_responses)}{'\033[0m'}")
