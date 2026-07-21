import os
import io
import contextlib
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()
import difflib
import string
import random
import spacy
import sqlite3
from DLM_Compute_Model import perform_advanced_CoT
from DLM_Memory_Model import get_category
from DLM_Memory_Model import get_specific_question
from DLM_Memory_Model import learn
import math
from transformers import pipeline
from better_profanity import profanity
from nltk.corpus import names


class DLM:
    # for one-time, shared model loaders so that each object won't load a new model (> 2GB)
    _shared_nlp = None
    _shared_hf = None
    _shared_profanity_loaded = False

    __filename = None  # knowledge-base (SQL)
    __query = None  # user-inputted query
    __nlp = None  # Spacy NLP analysis
    __tone = None  # sentimental tone of user query
    __mode = None  # either "learn" or "apply"
    __unsure_while_thinking = False  # if uncertain while thinking, then it will let the user know that
    __nlp_similarity_value = None  # saves the similarity value by doing SpaCy calculation (for debugging)
    __special_stripped_query = None  # saves query without any special words for reduced interference while vector calculating
    __nltk_names = set(name.lower() for name in names.words()) # list of name corpus to be identified in complex word problems
    __refuse_to_respond = False # if profanity and all caps-lock frustration is detected, refuse to respond and suggest user to rephrase nicely
    __model = None # bot automatically chooses between "compute" or "memory" model based on query type (auto-routing)
    __hf_classifier = None # loading huggingface model to determine the query type for auto_mode
    __successfully_computed = False # for when computation model was able to give an answer to a mathematical problem
    __try_compute = False # if the bot tried "memory" model first then decided to try "compute" model
    __try_memory = False # if the bot tried "compute" model first then decided to try "memory" model

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
            "for every", "into", "/", "÷"
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

    # to solve geometric problems (advanced CoT)
    __geometric_calculation_identifiers = {
        # 2D Shapes – Area
        "triangle": {
            "keywords": ["area", "triangle"],
            "params": ["base", "height"],
            "formula": lambda d: 0.5 * d["base"] * d["height"]
        },
        "rectangle": {
            "keywords": ["area", "rectangle"],
            "params": ["height", "width"],
            "formula": lambda d: d["height"] * d["width"]
        },
        "parallelogram": {
            "keywords": ["area", "parallelogram"],
            "params": ["base", "height"],
            "formula": lambda d: d["base"] * d["height"]
        },
        "square": {
            "keywords": ["area", "square"],
            "params": ["side"],
            "formula": lambda d: math.pow(d["side"], 2)
        },
        "trapezoid": {
            "keywords": ["area", "trapezoid"],
            "params": ["other", "height"],
            "formula": lambda d: 0.5 * (d["other"][0] + d["other"][1]) * d["height"]
        },
        "circle": {
            "keywords": ["area", "circle"],
            "params": ["radius"],
            "formula": lambda d: math.pi * math.pow(d["radius"], 2)
        },
        "ellipse": {
            "keywords": ["area", "ellipse"],
            "params": ["a", "b"],
            "formula": lambda d: math.pi * d["a"] * d["b"]
        },
        "pentagon": {
            "keywords": ["area", "pentagon"],
            "params": ["side"],
            "formula": lambda d: (1 / 4) * math.sqrt(5 * (5 + 2 * math.sqrt(5))) * math.pow(d["side"], 2)
        },

        # 3D Shapes – Volume
        "cube": {
            "keywords": ["volume", "cube"],
            "params": ["side"],
            "formula": lambda d: math.pow(d["side"], 3)
        },
        "rectangular prism": {
            "keywords": ["volume", "rectangular prism"],
            # "params": ["length", "width", "height"],
            # "formula": lambda d: d["length"] * d["width"] * d["height"]
            "params": ["height", "length", "width"],
            "formula": lambda d: d["height"] * d["length"] * d["width"]
        },
        "cylinder": {
            "keywords": ["volume", "cylinder"],
            "params": ["radius", "height"],
            "formula": lambda d: math.pi * math.pow(d["radius"], 2) * d["height"]
        },
        "cone": {
            "keywords": ["volume", "cone"],
            "params": ["radius", "height"],
            "formula": lambda d: (1 / 3) * math.pi * math.pow(d["radius"], 2) * d["height"]
        },
        "sphere": {
            "keywords": ["volume", "sphere"],
            "params": ["radius"],
            "formula": lambda d: (4 / 3) * math.pi * math.pow(d["radius"], 3)
        },
        "pyramid": {
            "keywords": ["volume", "pyramid"],
            "params": ["base_area", "height"],
            "formula": lambda d: (1 / 3) * d["base_area"] * d["height"]
        }
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

    # Response for when user uses profanity and all caps, indicating extreme anger
    __refuse_to_respond_statements = [
        "I understand you may be upset. However, I can’t respond to messages expressed in anger. Please rephrase calmly so I can assist you.",
        "Your message seems written in frustration. For a constructive exchange, I need you to restate it respectfully.",
        "I want to help, but I won’t respond to hostile language. Please rewrite your query in a calmer tone.",
        "I can see this might be frustrating. I can’t respond while the message is written in anger, but if you rephrase, I’ll gladly help.",
        "I know emotions can run high, but I need a calmer phrasing to continue. Please try rewording your question.",
        "It sounds like you’re upset. Let’s take a step back — rephrase your question respectfully and I’ll do my best to answer.",
        "Looks like the tone came across strongly. Please rephrase in a calmer way so I can give you the best answer.",
        "I can’t respond to messages phrased in anger. Try again in a clearer, more respectful tone, and I’ll assist right away.",
        "Let’s reset. Rephrase your question without the frustration, and I’ll be able to help you effectively."
    ]

    def __init__(self, mode, db_filename=None):  # initializes SQL database & SpaCy NLP
        """
        Initialize the Dynamic-Learning Model (DLM) chatbot.

        Parameters:
            mode (str): The access mode. Options:
                        'learn' for training mode (to train the bot with queries).
                        'apply' for a trained model to choose between compute and memory mode.
            db_filename (str, optional): The SQLite database file used to train and retrieve
                               question-answer-category triples.

        Behavior:
            - Loads the SpaCy NLP model ('en_core_web_lg').
            - Loads the HuggingFace model for auto-model detection.
            - Loads Better-Profanity for profane phrase sensing.
            - Connects to the specified SQLite database file.
            - Set appropriate mode value.
            - Verify login information based on mode.
            - Ensures the required table structure exists (creates if missing).
        """
        # lazy load SpaCy
        if DLM._shared_nlp is None:
            DLM._shared_nlp = spacy.load("en_core_web_lg")

        # lazy Load HuggingFace
        if DLM._shared_hf is None:
            DLM._shared_hf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # load profanity filter
        if not DLM._shared_profanity_loaded:
            profanity.load_censor_words()
            DLM._shared_profanity_loaded = True

        self.__nlp = DLM._shared_nlp
        self.__hf_classifier = DLM._shared_hf

        if db_filename is None:
            # Create an absolute path to a hidden folder in the user's home directory
            home_dir = os.path.expanduser("~")
            dlm_dir = os.path.join(home_dir, ".dlm")
            
            # Ensure the directory exists before SQLite tries to connect
            os.makedirs(dlm_dir, exist_ok=True)
            self.__filename = os.path.join(dlm_dir, "dlm_database.db")
        else:
            self.__filename = db_filename
        self.__mode = mode

        try:
            self.__conn = sqlite3.connect(self.__filename, check_same_thread=False)
            self.__cursor = self.__conn.cursor()
        except sqlite3.Error as e:
            print(f"System: Error connecting to database: {e}")
            self.__conn = None
            self.__cursor = None

        self.__create_table_if_missing()

    def __del__(self):
        """
        Destructor: safely closes the database connection when the object is destroyed.
        """
        try:
            # We check if the connection attribute exists and is not None
            if hasattr(self, '_DLM__conn') and self.__conn:
                self.__conn.close()
        except Exception:
            pass  # Suppress errors during destruction to prevent noisy exit

    def __create_table_if_missing(self):
        """
        Ensure the existence of the 'knowledge_base' table in the SQLite database; create or modify it if necessary.

        Behavior:
            - Establishes a connection to the SQLite database specified by self.__filename.
            - Creates the 'knowledge_base' table if it does not exist, with the following columns:
                - id (INTEGER, PRIMARY KEY, AUTOINCREMENT).
                - question (TEXT, NOT NULL, UNIQUE).
                - answer (TEXT, NOT NULL).
                - category (TEXT, NOT NULL).
            - If the table already exists but is missing the 'category' column, the method adds it with a default empty string.
            - Used exclusively within the class constructor to ensure the database schema is properly initialized.
        """
        if not self.__conn:
            return

        self.__cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    question    TEXT    NOT NULL UNIQUE,
                    answer      TEXT    NOT NULL,
                    category    TEXT    NOT NULL
                )
                """)

        self.__cursor.execute("PRAGMA table_info(knowledge_base)")
        cols = [row[1] for row in self.__cursor.fetchall()]

        if 'category' not in cols:
            self.__cursor.execute("""
                    ALTER TABLE knowledge_base
                    ADD COLUMN category TEXT NOT NULL DEFAULT ''
                    """)

        self.__conn.commit()

    def __filtered_input(self, userInput) -> string:
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
                - The current model is 'compute' and the word is a computation keyword.
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
                # In compute mode, keep any computation keyword even if it's also a filler
                is_computation_kw = any(word_lowered == kw.lower()
                                        for kws in self.__computation_identifiers.values()
                                        for kw in kws)

                if word_lowered not in self.__filler_words or (self.__model == "compute" and is_computation_kw):
                    filtered_words.append(word)

        # remove duplicates while preserving order (numbers excluded)
        seen = set()
        unique_words = []
        for word in filtered_words:
            try:
                float(word)  # Try to treat as number
                unique_words.append(word)  # Keep numeric strings (duplicates allowed)
            except ValueError:
                if any(word in keywords for keywords in self.__computation_identifiers.values()):
                    unique_words.append(word)  # Keep identifier words (duplicates allowed)
                elif word not in seen:
                    seen.add(word)
                    unique_words.append(word)

        # join the remaining words back into a string
        return " ".join(unique_words)

    def __set_sentiment_tone(self, orig_input):
        """
        Analyze the original user input and assign an appropriate emotional tone.

        Parameters:
            orig_input (str): The raw, unfiltered user query.

        Behavior:
            - Detects aggressive language using profanity filtering.
            - Analyzes punctuation and casing to infer emotional tone such as:
                - 'angry aggressive' for profane content - refuse to respond.
                - 'angry frustrated' for all-uppercase text - refuse to respond.
                - 'angry confused' for combined "?" and "!".
                - 'angry excited' for "!" only.
                - 'confused unclear' for "?" only.
                - 'doubtful uncertain' for ellipses ("..." or "..").
            - Stores the result in self.__tone as a string label.
        """
        is_profane = profanity.contains_profanity(orig_input)
        if is_profane and orig_input == orig_input.upper(): # too inappropriate to respond
            self.__refuse_to_respond = True
        else:
            self.__refuse_to_respond = False
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

    def __generate_thought(self, filtered_query, best_match_question, best_match_answer, highest_similarity, display_thought):
        """
        Simulate a Chain-of-Thought (CoT) reasoning process by printing the bot's internal analysis.

        Parameters:
            filtered_query (str): The cleaned version of the user's question, stripped of filler or trigger words.
            best_match_question (str): The closest matching question found in the knowledge base.
            best_match_answer (str): The corresponding answer to the matched question.
            highest_similarity (float): The calculated string similarity score (0 to 1) for the match.
            display_thought (bool): "True" if the bot is allowed to print its thought or else "False".

        Behavior:
            - Outputs step-by-step reasoning in a conversational format (e.g., interpreting the question's structure and tone).
            - In compute mode, calls advanced reasoning (e.g., math parsing or CoT decomposition).
            - Identifies the question's tone, topic, and potential intent based on interrogative words and SpaCy similarity.
            - Displays confidence based on similarity metrics and sets flags for uncertain answers.
            - Uses colorized terminal output and a loading animation to simulate reflective thought.
        """
        if display_thought:
            print("\nThought Process:")
            if filtered_query is None or filtered_query == "":
                print(
                    f"I couldn't pick out any context or clear topic. If I see a match in my database I will respond with that, or else I have no clue!")
            else:
                sentiment_tone = self.__tone.split()

                if self.__tone != "":
                    print(
                        f"Right off the bat, the user seems quite {sentiment_tone[0]} or {sentiment_tone[1]} by their query tone. Hopefully I won't disappoint!")
                if self.__model == "compute":
                    perform_advanced_CoT(self, filtered_query, display_thought)
                else:
                    interrogative_start = filtered_query.split()[0]
                    identifier = filtered_query
                    special_start = ["definition", "explanation", "description", "comparison", "calculation",
                                     "translation",
                                     "meaning"]  # special word in different form
                    for word in special_start:
                        identifier = identifier.replace(word, "")
                    # collapse any extra spaces
                    identifier = " ".join(identifier.split())
                    identifier = identifier.split()

                    if " ".join(identifier) == "":
                        print(
                            f"The user starts their query with \"{interrogative_start.title()}\", but I couldn't pick out a clear topic or context.")
                    else:
                        print(
                            f"The user starts their query with \"{interrogative_start.title()}\" and they are asking about \"{" ".join(identifier).title()}\".")
                    print("Let me think about this carefully...")

                    for s in special_start:
                        for u in filtered_query.split():
                            s_input = self.__nlp(s)
                            u_input = self.__nlp(u)
                            if (s_input.vector_norm != 0 and u_input.vector_norm != 0) and (
                                    s_input.similarity(u_input) > 0.60):
                                print(
                                    f"It seems like they want a {s} of \"{" ".join(identifier).title()}\".")

                    self.__semantic_similarity(self.__special_stripped_query, best_match_question)
                    spacy_proceed = self.__nlp_similarity_value is not None
                    if (best_match_answer is None) or (
                            highest_similarity < 0.65 and (spacy_proceed and self.__nlp_similarity_value < 0.85)):
                        print(
                            f"The closest match is only {int(highest_similarity * 100)}% similar when I used sequence matching.")
                        if spacy_proceed:
                            print(
                                f"Furthermore, an in-depth vector analysis revealed a similarity percentage of {int(self.__nlp_similarity_value * 100)}%.")
                        print(
                            f"{'Hmm...' or ''}I don't think I know the answer.")
                        self.__unsure_while_thinking = True
                    else:
                        self.__unsure_while_thinking = False
                        DB_identifier = get_specific_question(self, best_match_answer)
                        print(
                            f"Yes! I do remember learning about \"{DB_identifier}\" and I might have the right answer!")
                        print(
                            f"This is because when I did a sequence similarity calculation to one of the closest match in my database, I found it to be {int(highest_similarity * 100)}% similar.")
                        if spacy_proceed:
                            print(
                                f"Additionally, doing a more in-depth vector NLP analysis resulted in {int(self.__nlp_similarity_value * 100)}% similarity. Although there is room for error, we will see.")
                        print("Let me recall that answer...")
            print("\n")
        elif self.__model == "compute":
            perform_advanced_CoT(self, filtered_query, display_thought)

    def __generate_response(self, best_match_answer, best_match_question):
        """
        Generate a dynamic natural language response based on the answer's category.

        Parameters:
            best_match_answer (str): The stored answer retrieved from the knowledge base.
            best_match_question (str): The matched user question used to derive category context.

        Returns:
            str: The formatted, human-like response string based on the category template.
                 Returns an error message string if the category is unrecognized or missing.
        """
        identifier = get_category(self, best_match_question)

        if identifier is None:
            return "Sorry, I encountered an error on my end. Please try again later."

        if identifier == "generic":
            return best_match_answer

        elif identifier == "yesno":
            affirmative_templates = [
                "Yes, {}", "Absolutely, {}", "Certainly, {}", "Indeed, {}",
                "That's right, {}", "Correct, {}", "You got it, {}", "Sure thing, {}",
                "Of course, {}", "Definitely, {}", "Without a doubt, {}",
                "That's true, {}", "Affirmative, {}", "Right on, {}",
                "You're spot on, {}", "Exactly, {}", "Totally, {}",
                "No question about it, {}", "100%, {}", "I agree, {}"
            ]
            negative_templates = [
                "No, {}", "Not at all, {}", "Unfortunately, {}", "Of course not, {}",
                "That's not correct, {}", "Actually, no, {}", "I'm afraid not, {}",
                "Nope, {}", "Sorry, but no, {}", "That’s not the case, {}",
                "Negative, {}", "Not quite, {}", "That’s incorrect, {}",
                "I'm sorry, {}", "Absolutely not, {}", "Nah, {}",
                "Doesn’t seem so, {}", "I wouldn't say that, {}", "No way, {}",
                "That’s a no, {}"
            ]

            ans = best_match_answer.strip().lower()
            if ans.startswith(("no", "not", "don't", "do not", "never", "cannot")):
                template = random.choice(negative_templates)
                if ans.__contains__("no, "):
                    best_match_answer = best_match_answer.replace("no, ", "", 1)
                else:
                    best_match_answer = best_match_answer.replace("no ", "", 1)
            else:
                template = random.choice(affirmative_templates)
                if ans.__contains__("yes, "):
                    best_match_answer = best_match_answer.replace("yes, ", "", 1)
                else:
                    best_match_answer = best_match_answer.replace("yes ", "", 1)
            
            return template.format(best_match_answer)

        elif identifier == "process":  
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
            steps = best_match_answer.split("; ")  
            return random.choice(templates).format(*steps[:3])

        elif identifier == "definition":
            raw = best_match_question  
            triggers = {
                "what", "definition", "define", "meaning", "interpret",
                "what's", "whats", "what is", "what does", "mean", "means",
                "could", "you", "explain", "describe", "clarify", "tell",
                "me", "give", "the", "of", "in", "other", "words"
            }
            term_words = [w for w in raw.split() if w.lower() not in triggers]
            term = " ".join(term_words).strip()

            templates = [
                "It refers to {1}", "By definition, it is {1}",
                "In simple terms, it means {1}", "It can be described as {1}",
                "Essentially, it is {1}", "It is understood as {1}", 
                "In other words, it is {1}", "To put it simply, it refers to {1}", 
                "It typically means {1}", "It represents {1}",
                "It is defined as {1}", "You can think of it as {1}"
            ]
            return random.choice(templates).format(term, best_match_answer)

        elif identifier == "deadline":
            raw = best_match_question
            triggers = {
                "when", "what", "what's", "whats", "when's", "whens",
                "is", "the", "a", "an", "deadline", "due", "due date", 
                "cutoff", "closing", "closing date", "by", "before", "until",
                "date", "day", "last", "latest", "final", "damn"
            }
            words = raw.split()
            term_words = [w for w in words if w.lower() not in triggers]
            term = " ".join(term_words).strip()

            templates = [
                "The deadline is {1}", "You need to submit it by {1}",
                "Make sure to complete it by {1}", "It is due on {1}",
                "Don’t forget, it must be done by {1}", "It has a due date of {1}",
                "Be sure to finish it before {1}", "Please submit it no later than {1}",
                "It needs to be turned in by {1}", "The final date to complete it is {1}",
                "Submission closes on {1}", "You have until {1} to complete it",
                "It is expected to be submitted by {1}", "It must be handed in by {1}",
                "The cutoff is {1}"
            ]
            return random.choice(templates).format(term, best_match_answer)

        elif identifier == "location":
            templates = [
                "You can find it at {0}", "It’s located at {0}",
                "Head over to {0} for more information", "Check it out at {0}",
                "Access it via {0}", "You’ll find it here: {0}",
                "It’s available at {0}", "Navigate to {0} to view it",
                "You can reach it at {0}", "Visit {0} to learn more",
                "Take a look at {0}", "More details can be found at {0}",
                "For further info, go to {0}", "To see it yourself, just go to {0}"
            ]
            best_match_answer = best_match_answer.lower()
            return random.choice(templates).format(best_match_answer)

        elif identifier == "eligibility":
            templates = [
                "Eligibility means {0}", "Eligibility requires that {0}",
                "Qualifications are met only if {0}", "To be eligible, {0}",
                "Meeting eligibility involves {0}", "You qualify only if {0}",
                "Eligibility is based on whether {0}", "In order to qualify, {0}",
                "You are eligible when {0}", "The requirements are satisfied if {0}",
                "Eligibility depends on {0}", "To meet the qualifications, {0}",
                "Being eligible implies that {0}", "You're considered eligible if {0}",
                "Eligibility conditions include {0}"
            ]
            best_match_answer = best_match_answer.lower()
            return random.choice(templates).format(best_match_answer)

        else:
            return "Cannot retrieve and generate response due to data in unfamiliar category. Please try again later."

    def __semantic_similarity(self, userInput, knowledgebaseData) -> bool:
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
        if userInput is None or knowledgebaseData is None:
            return False
        UI_doc = self.__nlp(userInput)
        KB_doc = self.__nlp(knowledgebaseData)
        if UI_doc.vector_norm != 0 and KB_doc.vector_norm != 0:
            self.__nlp_similarity_value = UI_doc.similarity(KB_doc)
            return self.__nlp_similarity_value > 0.50
        else:
            return False
        
    def teach(self, question, expected_answer, category) -> learn: 
        """
        Public API for training the bot with new question-answer-category triples.

        Make sure the expected_answer adheres to the training rules, as written in the DLM github repo: https://github.com/VigneshT24/Dynamic_Learning_Model

        Category Options:
            - "yesno": make sure to start your answer responses with "yes" or "no" ONLY
            - "process": each answer must have three steps for your responses, separated by ";" (semicolon)
            - "definition": make sure to not mention the WORD/PHRASE to be defined & always start your response here with "the" only
            - "deadline": only include the deadline date, as an example, "March 31st 2025"
            - "location": mention the location only, nothing else. For example, "The xyz.com website"
            - "generic": format doesn't matter for this, give your answer in any comprehensive format
            - "eligibility": Make sure to ONLY start the response with a pronoun like "you", "they", "he", "she", etc
        
        More detail in the Github repo.
        """
        # calls the learn method from memory model file
        return learn(self, question, expected_answer, category)


    def ask(self, query, display_thought) -> dict: 
        """
        Process a user query and return a state-signaling dictionary containing the response and reasoning.

        This method employs an inversion-of-control architecture. It does not block execution 
        or prompt the user for input directly. Instead, it delegates conversational state to the 
        implementor by returning specific status codes. The implementor is responsible for 
        managing user prompts, validations, and calling the `teach()` method when required.

        Parameters:
            query (str): The user's question or statement to be processed.
            display_thought (bool): If True, captures the bot's internal Chain-of-Thought (CoT) 
                                    reasoning and includes it in the returned dictionary.

        Returns:
            dict: A structured response containing the following keys:
                - 'status' (str): The state of the interaction.
                - 'answer' (str): The final generated response, fallback message, or refusal statement.
                - 'thought' (str): The captured CoT analysis (empty if display_thought is False).
                - 'context' (dict): Metadata needed for teaching, including 'special_stripped_query' 
                                    and 'best_match_answer'.

        'status' Codes:
            - 'resolved': The bot successfully answered the query (or executed a fallback response).
            - 'refused': The bot refused to answer due to profanity, aggressive tone, or an empty query.
            - 'confirm_teaching': The bot found a potential answer in 'learn' mode. The implementor 
                                  should ask the user to verify if the answer is expected.
            - 'needs_teaching': The bot could not find a valid answer or compute a result. The implementor 
                                should prompt the user for the correct answer and category, then pass 
                                those to the `teach()` method.
        """
        # initialize return schema
        response_data = {
            "status": "resolved", # 'resolved', 'refused', 'confirm_teaching', 'needs_teaching'
            "answer": "",
            "thought": "",
            "context": {}
        }

        cot_buffer = io.StringIO()
        answer_buffer = io.StringIO()

        self.__query = query
        
        # for implementor to handle empty queries
        if self.__query is None or self.__query.strip() == "":
            response_data["status"] = "refused"
            response_data["answer"] = "Empty input is unacceptable. Please enter something."
            return response_data

        # tone check
        with contextlib.redirect_stdout(answer_buffer):
            self.__set_sentiment_tone(self.__query) 
            if self.__refuse_to_respond:
                print()
                print(random.choice(self.__refuse_to_respond_statements))

        # for implementor to handle 
        if self.__refuse_to_respond:
            response_data["status"] = "refused"
            response_data["answer"] = answer_buffer.getvalue()
            return response_data

        # model routing
        if self.__mode != "learn":
            auto_model_choice = self.__hf_classifier(self.__query, ["mathematical", "not mathematical"])["labels"][0]
            if auto_model_choice == "mathematical":
                self.__model = "compute"
            else:
                self.__model = "memory"
        else:
            self.__model = "memory"

        # filtering
        if self.__model == "compute":
            keep = {".", "+", "-", "*", "/", "="}
            to_remove = "".join(ch for ch in string.punctuation if ch not in keep)
        else:
            to_remove = string.punctuation

        translation_table = str.maketrans("", "", to_remove)
        filtered_query = self.__filtered_input(self.__query.lower().translate(translation_table))

        self.__special_stripped_query = filtered_query
        special_exceptions = ["definition", "explanation", "description", "comparison", "calculation", "translation", "meaning"]
        for word in special_exceptions:
            self.__special_stripped_query = self.__special_stripped_query.replace(word, "")
        self.__special_stripped_query = " ".join(self.__special_stripped_query.split())

        # database search
        if self.__cursor:
            self.__cursor.execute("SELECT question, answer FROM knowledge_base")
            rows = self.__cursor.fetchall()
        else:
            rows = []

        highest_similarity = 0.0
        best_match_question = None
        best_match_answer = None

        for stored_question, stored_answer in rows:
            sim_stripped = difflib.SequenceMatcher(None, stored_question, self.__special_stripped_query).ratio()
            sim_filtered = difflib.SequenceMatcher(None, stored_question, filtered_query).ratio()
            sim = max(sim_stripped, sim_filtered)

            if sim > highest_similarity:
                highest_similarity = sim
                best_match_question = stored_question
                best_match_answer = stored_answer

        if highest_similarity < 0.65 and not self.__semantic_similarity(self.__special_stripped_query, best_match_question):
            best_match_answer = None
            best_match_question = None

        response_data["context"] = {
            "special_stripped_query": self.__special_stripped_query,
            "best_match_answer": best_match_answer
        }

        # primary model attempt
        with contextlib.redirect_stdout(cot_buffer):
            self.__generate_thought(filtered_query, best_match_question, best_match_answer, highest_similarity, display_thought)

        # fallback routing attempt
        if self.__model == "compute" and self.__try_memory:
            with contextlib.redirect_stdout(cot_buffer):
                if display_thought:
                    print("Let me put this into my memory model, maybe it wasn't a mathematical query...")
                self.__model = "memory"
                self.__generate_thought(filtered_query, best_match_question, best_match_answer, highest_similarity, display_thought)
            self.__try_memory = False

        if self.__model == "memory":
            is_valid_match = (not self.__unsure_while_thinking) and ((highest_similarity >= 0.65) or (best_match_answer and self.__semantic_similarity(self.__special_stripped_query, best_match_question)))
            
            if not is_valid_match and self.__mode != "learn":
                self.__model = "compute"
                self.__try_compute = True
                with contextlib.redirect_stdout(cot_buffer):
                    if display_thought:
                        print("Let me put this into my computation model, maybe it was a mathematical query...")
                    self.__generate_thought(filtered_query, best_match_question, best_match_answer, highest_similarity, display_thought)

        # resolution & final answer capture
        with contextlib.redirect_stdout(answer_buffer):
            if self.__model == "memory":
                is_valid_match = (not self.__unsure_while_thinking) and ((highest_similarity >= 0.65) or (best_match_answer and self.__semantic_similarity(self.__special_stripped_query, best_match_question)))
                
                if is_valid_match:
                    self.__unsure_while_thinking = False
                    print(self.__generate_response(best_match_answer, best_match_question))
                    response_data["status"] = "confirm_teaching" if self.__mode == "learn" else "resolved"
                else:
                    if self.__mode != "learn":
                        print(random.choice(self.__fallback_responses))
                    response_data["status"] = "needs_teaching"

            elif self.__model == "compute":
                if self.__successfully_computed:
                    response_data["status"] = "resolved"
                else:
                    print(random.choice(self.__fallback_responses))
                    response_data["status"] = "needs_teaching"
                    
                self.__try_compute = False
                self.__successfully_computed = False

        # extract and clean data
        full_thought = cot_buffer.getvalue()
        final_answer = answer_buffer.getvalue()

        # extract math answers from the thought buffer into the answer buffer
        if self.__model == "compute" and final_answer.strip() == "":
            lines = full_thought.split("\n")
            thought_lines = []
            for line in lines:
                if "Answer:" in line:
                    final_answer += line + "\n"
                else:
                    thought_lines.append(line)
            full_thought = "\n".join(thought_lines)

        response_data["thought"] = full_thought.strip()
        response_data["answer"] = final_answer.strip()

        cot_buffer.close()
        answer_buffer.close()

        return response_data