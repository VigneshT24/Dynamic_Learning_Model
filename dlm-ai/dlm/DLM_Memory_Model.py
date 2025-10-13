import sqlite3

def get_category(self, exact_question):  # returns category as a string or None
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
    conn = sqlite3.connect(self._DLM__filename)
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


def get_specific_question(self, exact_answer):  # returns question as a string or None
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
    conn = sqlite3.connect(self._DLM__filename)
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

def learn(self, expectation, category):  # no return, void
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
    conn = sqlite3.connect(self._DLM__filename)
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO knowledge_base (question, answer, category) VALUES (?, ?, ?)",
        (self._DLM__special_stripped_query, expectation, category)
    )
    conn.commit()
    conn.close()
