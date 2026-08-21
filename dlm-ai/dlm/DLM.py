import os
import io
import contextlib
import difflib
import string
import random
import spacy
import sqlite3
import socket
import subprocess
import time
import re
from DLM_Compute_Model import *
from DLM_Memory_Model import *
from better_profanity import profanity
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class DLM:
    """
    Dynamic-Learning Model (DLM) Engine.

    A hybrid intent-routing engine that dynamically routes user queries between a 
    SQLite-backed computational math engine and an memory model. It features 
    Chain-of-Thought (CoT) reasoning, sentiment analysis, and an inversion-of-control 
    architecture for seamless integration into external applications.
    """
    # for one-time, shared model loaders so that each object won't load a new model (> 2GB)
    _shared_nlp = None
    _shared_hf = None
    _shared_profanity_loaded = False
    _shared_router = None

    __filename = None  # knowledge-base (SQL)
    __query = None  # user-inputted query
    __nlp = None  # Spacy NLP analysis
    __tone = None  # sentimental tone of user query
    __mode = None  # either "train_memory", "train_compute", or "apply"
    __unsure_while_thinking = False  # if uncertain while thinking, then it will let the user know that
    __nlp_similarity_value = None  # saves the similarity value by doing SpaCy calculation (for debugging)
    __special_stripped_query = None  # saves query without any special words for reduced interference while vector calculating
    __refuse_to_respond = False # if profanity and all caps-lock frustration is detected, refuse to respond and suggest user to rephrase nicely
    __model = None # bot automatically chooses between "compute" or "memory" model based on query type (auto-routing)
    __computation_feedback = ""
    __computation_state = None

    # personalized responses to let the user know that the bot doesn't know the answer
    __fallback_responses = [
        "Hmm, that's a great question! I might need more context or details to answer it.",
        "I'm still training my brain on that topic. Could you clarify what you mean?",
        "Oops! That one's not in my database yet, or maybe it's phrased in a way I don't recognize!",
        "You got me this time! Could you try rewording it so I can understand better?",
        "That's a tough one! I might need a bit more information to figure it out.",
        "I don't have the answer just yet, but I bet it's out there somewhere! Could you rephrase it?",
        "Hmm... I'll have to hit the books for that one! Or maybe I just need a little more context?",
        "I haven't learned that yet, but I'm constantly improving! Maybe try a different wording?",
        "You just stumped me! But no worries, I'm always evolving—maybe I misinterpreted the question?",
        "That's outside my knowledge base for now, or maybe I'm just not parsing it right!",
        "I wish I had the answer! If it's incomplete, could you add more details?",
        "I'm not sure about that one. Maybe try breaking it down into smaller parts?",
        "Hmm, I don't have an answer yet. Could you reword or give more details?",
        "Still learning this one! If something's missing, feel free to add more context.",
        "I don't have that in my knowledge bank yet, or maybe I'm missing part of the question!"
    ]

    # words to be filtered from user input for better accuracy and fewer distractions
    __filler_words = [

        # articles & determiners (words that don't add meaning to sentence)
        "the", "some", "any", "many", "every", "each", "either", "neither", "this", "that", "these", "those",
        "certain", "another", "such", "whatsoever", "whichever", "whomever", "whatever", "all", "something", "possible",

        # pronouns (general pronouns that don't change meaning)
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

        # verbs commonly used in questions (but don't change meaning)
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

        # words that don't add meaning
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

    # response for when user uses profanity and all caps, indicating extreme anger
    __refuse_to_respond_statements = [
        "I understand you may be upset. However, I can't respond to messages expressed in anger. Please rephrase calmly so I can assist you.",
        "Your message seems written in frustration. For a constructive exchange, I need you to restate it respectfully.",
        "I want to help, but I won't respond to hostile language. Please rewrite your query in a calmer tone.",
        "I can see this might be frustrating. I can't respond while the message is written in anger, but if you rephrase, I'll gladly help.",
        "I know emotions can run high, but I need a calmer phrasing to continue. Please try rewording your question.",
        "It sounds like you're upset. Let's take a step back — rephrase your question respectfully and I'll do my best to answer.",
        "Looks like the tone came across strongly. Please rephrase in a calmer way so I can give you the best answer.",
        "I can't respond to messages phrased in anger. Try again in a clearer, more respectful tone, and I'll assist right away.",
        "Let's reset. Rephrase your question without the frustration, and I'll be able to help you effectively."
    ]

    def __init__(self, mode, db_filename=None):  # initializes SQL database & SpaCy NLP
        """
        Initializes the DLM engine, loading NLP models, connecting to the knowledge base and compute model database.

        Args:
            mode (str): 
                * 'train_memory': training/correcting factual domain-specific queries
                * 'train_compute': training/correcting the computated answers for validity 
                * 'apply': No training, for when DLM is production ready (ensure to train well before using this for production applications)
            db_filename (str, optional): Absolute path to the SQLite memory database. Defaults to '~/.dlm/dlm_database.db'.
        """
        self.__ensure_ollama_running() # ensure router is running
        # lazy load SpaCy
        if DLM._shared_nlp is None:
            try:
                DLM._shared_nlp = spacy.load("en_core_web_lg") # type: ignore
            except OSError:
                print("[SYSTEM]: Downloading required SpaCy NLP model (this will only happen once)...")
                from spacy.cli import download # type: ignore
                download("en_core_web_lg") 
                DLM._shared_nlp = spacy.load("en_core_web_lg")

        # load profanity filter
        if not DLM._shared_profanity_loaded:
            profanity.load_censor_words()
            DLM._shared_profanity_loaded = True

        # lazy load ollama router
        if DLM._shared_router is None:
            DLM._shared_router = ChatOllama(model='llama3.2', base_url='http://localhost:11434')
        self.__router_llm = DLM._shared_router

        self.__nlp = DLM._shared_nlp

        if db_filename is None:
            # create an absolute path to a hidden folder in the user's home directory
            home_dir = os.path.expanduser("~")
            dlm_dir = os.path.join(home_dir, ".dlm")
            
            # ensure the directory exists before SQLite tries to connect
            os.makedirs(dlm_dir, exist_ok=True)
            self.__filename = os.path.join(dlm_dir, "dlm_database.db")
        else:
            self.__filename = db_filename
        self.__mode = mode
        self.__computation_feedback = ""

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
            # we check if the connection attribute exists and is not None
            if hasattr(self, '_DLM__conn') and self.__conn:
                self.__conn.close()
        except Exception:
            pass  # suppress errors during destruction to prevent noisy exit

    def __create_table_if_missing(self) -> None:
        """
        Ensures the SQLite 'knowledge_base' table exists and has the required schema.
        Automatically adds the 'category' column to legacy databases.
        """
        if not self.__conn:
            return

        assert self.__cursor is not None
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

    def __ensure_ollama_running(self) -> None: # pyright: ignore[reportSelfClsParameterName]
        """Silently checks if Ollama is active, and boots it in the background if it is not."""
        port = 11434
        host = '127.0.0.1'
        server_running = False

        # check if the server is already awake
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            try:
                s.connect((host, port))
                server_running = True
            except socket.error:
                pass
        
        # boot the server if it was asleep
        if not server_running:
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0),
                    start_new_session=True
                )
                time.sleep(3)
            except FileNotFoundError:
                print("\n[CRITICAL ERROR]: Ollama is not installed on this system. Please install it from ollama.com to use DLM.")
                return

        # verify the required models exists even if the server was already running
        try:
            required_models = ["llama3.2", "nomic-embed-text"]
            existing_models = subprocess.check_output(["ollama", "list"]).decode("utf-8")

            for model in required_models:
                if model not in existing_models:
                    print(f"\n[SYSTEM]: Downloading and pulling required Ollama model '{model}'. This may take a few minutes...")
                    subprocess.run(["ollama", "pull", model], check=True)
        except FileNotFoundError:
             print("\n[CRITICAL ERROR]: Ollama is not installed on this system. Please install it from ollama.com to use DLM.")

    def __filtered_input(self, userInput) -> str:
        """
        Filter out filler words from the user input while preserving important context.

        Parameters:
            userInput (str): The raw, lowercase-converted user query string.

        Returns:
            str: A filtered version of the input string with filler words removed and duplicates eliminated.
        """
        # tokenize user input (split into words)
        words = userInput.lower().split()

        # remove filler words
        filtered_words = []
        for i, word in enumerate(words):
            word_lowered = word.lower()

            # allow exceptions only in first position:
            if i == 0 and word_lowered in self.__exception_fillers:
                filtered_words.append(word)

            # otherwise, only keep non-fillers
            elif word_lowered not in self.__filler_words:
                filtered_words.append(word)

        # remove duplicates while preserving order (numbers excluded)
        seen = set()
        unique_words = []
        for word in filtered_words:
            try:
                float(word)  # try to treat as number
                unique_words.append(word)  # keep numeric strings (duplicates allowed)
            except ValueError:
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)

        # join the remaining words back into a string
        return " ".join(unique_words)

    def __set_sentiment_tone(self, orig_input) -> None:
        """
        Analyzes punctuation and profanity to determine the user's emotional state.

        Sets the internal `__tone` variable and triggers the `__refuse_to_respond` flag 
        if aggressive or highly inappropriate language is detected.

        Args:
            orig_input (str): The raw, unaltered user query.
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

    def __generate_thought(self, orig_query, filtered_query, best_match_question, best_match_answer, highest_similarity, display_thought) -> None:
        """
        Simulates and captures the Chain-of-Thought (CoT) reasoning process.

        Analyzes tone, extracts context, checks string/vector similarities, and delegates 
        to the advanced CoT computation engine if routed to the math model. Output is 
        redirected to a buffer by the caller.

        Args:
            filtered_query (str): The cleaned user query.
            best_match_question (str): The closest matching question from the database.
            best_match_answer (str): The corresponding database answer.
            highest_similarity (float): The sequence matcher ratio (0.0 to 1.0).
            display_thought (bool): Flag enabling CoT generation.
        """
        if display_thought:
            if filtered_query is None or filtered_query == "":
                print(f"I couldn't pick out any context or clear topic. If I see a match in my database I will respond with that, or else I have no clue.")
            else:
                assert self.__tone is not None
                sentiment_tone = self.__tone.split()

                if self.__tone != "" and self.__model == "memory":
                    print(f"Right off the bat, the user seems quite {sentiment_tone[0]} or {sentiment_tone[1]} by their query tone. Hopefully I won't disappoint!")
                if self.__model == "compute":
                    # save dict generated by invoking question to compute model
                    self.__computation_state = dlm_compute_engine.invoke({"query": orig_query}) # type: ignore
                    route = self.__computation_state.get("route")
                    formula = self.__computation_state.get("formula", "Unknown")
                    template = self.__computation_state.get("formula_template", "Unknown")
                    var_num = self.__computation_state.get("var_num", [])

                    print("Let me break down the math for this...")

                    if not self.__computation_state or "answer" not in self.__computation_state:
                        print("I tried to compute the query but I couldn't formulate a valid mathematical expression.")
                        return

                    print(f"First, I extracted the numerical variables from the query: {var_num}")

                    if route == "apply":
                        print("I searched my computation database and found an exact formula match for this type of problem.")
                        print(f"I applied the saved template: {template}")
                    else:
                        print("I didn't have a saved formula for this scenario, so I sent a request to my computation engine to formulate a new one from scratch.")
                        print(f"It successfully generated the formula: {template}")

                    print(f"I plugged the variables into the template to create an expression: {formula}")
                    calc_answer = str(self.__computation_state.get("answer", ""))

                    if "Error:" in calc_answer:
                        print("However, when I tried to execute the math, my environment threw an error. I likely lack the specific libraries needed to solve this type of equation.")
                    else:
                        print("I executed the generated formula and I have the answer ready.")
                else:
                    interrogative_start = filtered_query.split()[0]
                    identifier = filtered_query
                    special_start = ["definition", "explanation", "description", "comparison", "calculation",
                                     "translation",
                                     "meaning"] # special word in different form
                    for word in special_start:
                        identifier = identifier.replace(word, "")
                    # collapse any extra spaces
                    identifier = " ".join(identifier.split())
                    identifier = identifier.split()

                    if " ".join(identifier) == "":
                        print(f"The user starts their query with \"{interrogative_start.title()}\", but I couldn't pick out a clear topic or context.")
                    else:
                        print(f"The user starts their query with \"{interrogative_start.title()}\" and they are asking about \"{' '.join(identifier).title()}\".")
                    print("Let me think about this carefully...")

                    for s in special_start:
                        for u in filtered_query.split():
                            s_input = self.__nlp(s) # type: ignore
                            u_input = self.__nlp(u) # type: ignore
                            if (s_input.vector_norm != 0 and u_input.vector_norm != 0) and (
                                    s_input.similarity(u_input) > 0.60):
                                print(
                                    f"It seems like they want a {s} of \"{' '.join(identifier).title()}\".")

                    is_semantically_similar = self.__semantic_similarity(self.__special_stripped_query, best_match_question)
                    spacy_proceed = self.__nlp_similarity_value is not None
                    if (best_match_answer is None) or (highest_similarity < 0.65 and not is_semantically_similar): # type: ignore
                        print(
                            f"The closest match is only {int(highest_similarity * 100)}% similar when I used sequence matching.")
                        if spacy_proceed:
                            print(
                                f"Furthermore, an in-depth vector analysis revealed a similarity percentage of {int(self.__nlp_similarity_value * 100)}%.") # type: ignore
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
                                f"Additionally, doing a more in-depth vector NLP analysis resulted in {int(self.__nlp_similarity_value * 100)}% similarity. Although there is room for error, we will see.") # type: ignore
                        print("Let me recall that answer...")
            print("\n")

    def __generate_response(self, best_match_answer, best_match_question) -> str:
        """
        Generates a dynamic, human-like response based on the category of the matched answer.

        Args:
            best_match_answer (str): The raw answer retrieved from the database.
            best_match_question (str): The original question to provide context for definitions/deadlines.

        Returns:
            str: The formatted response string ready to be delivered to the user.
        """
        clean_answer = best_match_answer.strip().rstrip(".!?")

        identifier = get_category(self, best_match_question)

        if identifier is None or identifier == "":
            return "Sorry, I encountered an error on my end. Please try again later."

        if identifier == "generic":
            return clean_answer
        
        if identifier == "yesno":
            clean_answer = clean_answer[:1].lower() + clean_answer[1:]
            affirmative_templates = [
                "Yes, {}", "Absolutely, {}", "Certainly, {}", "Indeed, {}",
                "That's right, {}", "Correct, {}", "You got it, {}", "Sure thing, {}",
                "Of course, {}", "Definitely, {}", "Without a doubt, {}",
                "That's true, {}", "Right on, {}",
                "You're spot on, {}", "Exactly, {}", "Totally, {}",
                "No question about it, {}", "100%, {}", "I agree, {}"
            ]
            negative_templates = [
                "No, {}", "Not at all, {}", "Unfortunately, {}", "Of course not, {}",
                "That's not correct, {}", "Actually, no, {}", "I'm afraid not, {}",
                "Nope, {}", "Sorry, but no, {}", "That's not the case, {}", "Not quite, {}", 
                "That's incorrect, {}", "I'm sorry, {}", "Absolutely not, {}", "Nah, {}",
                "Doesn't seem so, {}", "I wouldn't say that, {}", "No way, {}",
                "That's a no, {}"
            ]

            ans = clean_answer.strip().lower()
            if ans.startswith(("no", "not", "don't", "do not", "never", "cannot")):
                template = random.choice(negative_templates)
                if ans.__contains__("no, "):
                    clean_answer = ans.replace("no, ", "", 1)
                else:
                    clean_answer = ans.replace("no ", "", 1)
            else:
                
                template = random.choice(affirmative_templates)
                if ans.__contains__("yes, "):
                    clean_answer = ans.replace("yes, ", "", 1)
                else:
                    clean_answer = ans.replace("yes ", "", 1)
            
            return template.format(clean_answer)

        elif identifier == "process":
            clean_answer = re.sub(r'(^|;\s*)([A-Z])', lambda m: f"{m.group(1)}{m.group(2).lower()}", clean_answer)
            starters = ["First, {}.", "To start, {}.", "First off, {}.", "As a first step, {}.", "To get started, {}.", "The first thing to do is {}."]
            continuations = ["Next, {}.", "Then, {}.", "After that, {}.", "Following that, {}.", "Once that's done, {}.", "From there, {}.", "Afterward, {}.", "The next step is to {}."]
            finishers = ["Finally, {}.", "Lastly, {}.", "To wrap up, {}.", "To finish, {}.", "As a final step, {}.", "To complete the process, {}.", "The last step is to {}."]

            steps = clean_answer.split("; ")

            complete_process = ""

            for i in range(len(steps)):
                current_step = steps[i].strip()
                if i == 0:
                    complete_process += random.choice(starters).format(current_step)
                elif i == (len(steps)) - 1:
                    complete_process += random.choice(finishers).format(current_step)
                else:
                    complete_process += random.choice(continuations).format(current_step)
                complete_process += " "

            return complete_process

        elif identifier == "definition":
            clean_answer = clean_answer[:1].lower() + clean_answer[1:]
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
            return random.choice(templates).format(term, clean_answer)

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
                "Don't forget, it must be done by {1}", "It has a due date of {1}",
                "Be sure to finish it before {1}", "Please submit it no later than {1}",
                "It needs to be turned in by {1}", "The final date to complete it is {1}",
                "Submission closes on {1}", "You have until {1} to complete it",
                "It is expected to be submitted by {1}", "It must be handed in by {1}",
                "The cutoff is {1}"
            ]
            return random.choice(templates).format(term, clean_answer)

        elif identifier == "location":
            templates = [
                "You can find it at {0}", "It's located at {0}",
                "Head over to {0} for more information", "Check it out at {0}",
                "Access it via {0}", "You'll find it here: {0}",
                "It's available at {0}", "Navigate to {0} to view it",
                "You can reach it at {0}", "Visit {0} to learn more",
                "Take a look at {0}", "More details can be found at {0}",
                "For further info, go to {0}", "To see it yourself, just go to {0}"
            ]
            return random.choice(templates).format(clean_answer)

        elif identifier == "eligibility":
            clean_answer = clean_answer[:1].lower() + clean_answer[1:]
            templates = [
                "To be eligible, {0}.",
                "In order to qualify, {0}.",
                "To meet the qualifications, {0}.",
                "As a prerequisite, {0}.",
                "The criteria state that {0}.",
                "For eligibility, {0}.",
                "To meet the requirements, {0}.",
                "Keep in mind that to qualify, {0}.",
                "As a general rule, {0}."
            ]
            return random.choice(templates).format(clean_answer)

        else:
            return "Cannot retrieve and generate response due to data in unfamiliar category. Please try again later."

    def __semantic_similarity(self, userInput, knowledgebaseData) -> bool:
        """
        Evaluates the semantic meaning between the user's query and a database entry.

        Args:
            userInput (str): The filtered user query.
            knowledgebaseData (str): The question from the database to compare against.

        Returns:
            bool: True if the SpaCy vector similarity exceeds 0.75, False otherwise.
        """
        if userInput is None or knowledgebaseData is None:
            return False
        UI_doc = self.__nlp(userInput) # type: ignore
        KB_doc = self.__nlp(knowledgebaseData) # type: ignore
        if UI_doc.vector_norm != 0 and KB_doc.vector_norm != 0:
            self.__nlp_similarity_value = UI_doc.similarity(KB_doc)
            return self.__nlp_similarity_value > 0.75
        else:
            return False
        
    def teach_memory(self, question, expected_answer, category) -> learn:  # type: ignore
        """
        Public API for training the bot with new question-answer-category triples.

        Make sure the expected_answer adheres to the training rules, as written in the DLM github repo README: https://github.com/VigneshT24/Dynamic_Learning_Model

        Category Options:
            - "yesno": make sure to start your answer responses with "yes" or "no" ONLY
            - "process": each answer must have three steps for your responses, separated by ";" (semicolon)
            - "definition": make sure to not mention the WORD/PHRASE to be defined & always start your response here with "the" only
            - "deadline": only include the deadline date, as an example, "March 31st 2025"
            - "location": mention the location only, nothing else. For example, "The xyz.com website"
            - "generic": format doesn't matter for this, give your answer in any comprehensive format
            - "eligibility": Make sure to ONLY start the response with a pronoun like "you", "they", "he", "she", etc
        
        More details in the Github repo README.
        """
        # calls the learn method from memory model file
        return learn(self, question, expected_answer, category)

    def teach_compute(self, generalized_query, var_num, corrected_template) -> dict:
        """
        Public API for correcting the computation model's formula.
        Passes the corrected template to the compute engine and recalculates.
        """
        return update_compute_database(generalized_query, var_num, corrected_template)

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
            - 'confirm_memory': The bot found a potential answer in 'learn' mode. The implementor 
                                should verify if the answer is expected.
            - 'confirm_compute': The bot calculated the answer to the mathematical query while in 'compute' mode. The implementor
                                should verify if the answer is correct.
            - 'needs_teaching': The bot could not find a valid answer or compute a result. The implementor 
                                should prompt the user for the correct answer and category, then pass 
                                those to the `teach()` method.
        """
        # initialize return schema
        response_data = {
            "status": "resolved",
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

        # filtering
        to_remove = ""
        if self.__model == "memory":
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

        # three modes, three different ways to handle (HYBRID ROUTING)
        if self.__mode == "apply":
            if highest_similarity >= 0.75:
                self.__model = "memory" # bypass the routing since it must be a memory trained query
            else:
                routing_msg = [
                    SystemMessage(content=(
                        "You are a strict binary routing script for an AI system.\n"
                        "Categorize the user's query into one of two buckets:\n"
                        "1. COMPUTE: Calculating numbers, math word problems, or unit conversions.\n"
                        "2. MEMORY: Factual information, definitions, yes/no questions, processes, or general knowledge.\n\n"
                        "EXAMPLES:\n"
                        "Q: 'What is the definition of ROS 2?' -> <ROUTE>MEMORY</ROUTE>\n"
                        "Q: 'Add -45.5 and 10' -> <ROUTE>COMPUTE</ROUTE>\n"
                        "Q: 'Can HuggingFace transformers be loaded lazily?' -> <ROUTE>MEMORY</ROUTE>\n"
                        "Q: 'Multiply 0.85 by 12' -> <ROUTE>COMPUTE</ROUTE>\n"
                        "Q: 'What is the process for analyzing a quantum circuit?' -> <ROUTE>MEMORY</ROUTE>\n\n"
                        "Output ONLY the exact XML tag <ROUTE>COMPUTE</ROUTE> or <ROUTE>MEMORY</ROUTE>. Do not output any other text."
                    )),
                    HumanMessage(content=self.__query)
                ]

                route_response = self.__router_llm.invoke(routing_msg).content.strip().upper()

                if "<ROUTE>COMPUTE</ROUTE>" in route_response:
                    self.__model = "compute"
                else:
                    self.__model = "memory"

        elif self.__mode == "train_memory":
            self.__model = "memory"
        elif self.__mode == "train_compute":
            self.__model = "compute"

        response_data["context"] = {
            "special_stripped_query": self.__special_stripped_query,
            "best_match_answer": best_match_answer
        }

        # primary model attempt
        with contextlib.redirect_stdout(cot_buffer):
            self.__generate_thought(self.__query, filtered_query, best_match_question, best_match_answer, highest_similarity, display_thought)

        is_valid_match = (not self.__unsure_while_thinking) and ((highest_similarity >= 0.65) or (best_match_answer and self.__semantic_similarity(self.__special_stripped_query, best_match_question)))

        # resolution & final answer capture
        with contextlib.redirect_stdout(answer_buffer):
            if self.__model == "memory":
                if is_valid_match:
                    self.__unsure_while_thinking = False
                    print(self.__generate_response(best_match_answer, best_match_question))
                    response_data["status"] = "confirm_memory" if self.__mode == "train_memory" else "resolved"
                else:
                    if self.__mode == "apply":
                        print(random.choice(self.__fallback_responses))
                    response_data["status"] = "needs_teaching" if self.__mode == "train_memory" else "resolved"

            elif self.__model == "compute":
                if self.__computation_state:
                    response_data["status"] = "confirm_compute" if self.__mode == "train_compute" else "resolved"

                    calc_answer = str(self.__computation_state.get('answer', ''))
                    if "Error:" in calc_answer:
                        print(random.choice(self.__fallback_responses))
                    else:
                        print(f"The calculated result: {calc_answer}")
                    
                    # pack the computation context for the implementor
                    response_data["context"] = {
                        "generalized_query": self.__computation_state.get("generalized_query"),
                        "var_num": self.__computation_state.get("var_num"),
                        "formula": self.__computation_state.get("formula"),
                        "answer": self.__computation_state.get("answer")
                    }
                else:
                    if self.__computation_feedback != "":
                        print(self.__computation_feedback)
                        self.__computation_feedback = ""
                    else:
                        print(random.choice(self.__fallback_responses))
                        
                    response_data["status"] = "needs_teaching" if self.__mode == "train_compute" else "resolved"

        # extract and clean data
        full_thought = cot_buffer.getvalue()
        final_answer = answer_buffer.getvalue()

        response_data["thought"] = full_thought.strip()
        response_data["answer"] = final_answer.strip()

        cot_buffer.close()
        answer_buffer.close()

        return response_data