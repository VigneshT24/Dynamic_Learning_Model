def get_category(self, exact_question) -> str | None:  # returns category as a string or None
    """
    Retrieves the category tag for a specific question from the knowledge base.

    Args:
        exact_question (str): The exact question string to look up.

    Returns:
        str or None: The category (e.g., 'yesno', 'definition') if found, otherwise None.
    """
    if not hasattr(self, '_DLM__cursor') or not self._DLM__cursor:
        return None
    try:
        self._DLM__cursor.execute(
            "SELECT category FROM knowledge_base WHERE question = ?",
            (exact_question,)
        )
        row = self._DLM__cursor.fetchone()

        if row:
            return row[0]  # this is the category/question_type
        else:
            return None  # question not found

    except Exception as e:
        print(f"[SYSTEM]: Database Read Error in get_category: {e}")
        return None

def get_specific_question(self, exact_answer) -> str | None:  # returns question as a string or None
    """
    Retrieves the original question associated with an exact answer from the knowledge base.

    Args:
        exact_answer (str): The exact answer string to look up.

    Returns:
        str or None: The corresponding question if found, otherwise None.
    """
    if not hasattr(self, '_DLM__cursor') or not self._DLM__cursor:
        return None

    try:
        self._DLM__cursor.execute(
            "SELECT question FROM knowledge_base WHERE answer = ?",
            (exact_answer,)
        )
        row = self._DLM__cursor.fetchone()

        if row:
            return row[0]
        else:
            return None

    except Exception as e:
        print(f"[SYSTEM]: Database Read Error in get_specific_question: {e}")
        return None

def learn(self, question, expectation, category) -> bool:  
    """
    Saves or updates a question-answer pair and its category in the knowledge base.

    If the question already exists, its answer and category are overwritten (Upsert).

    Args:
        question (str): The stripped user query to store.
        expectation (str): The expected correct answer.
        category (str): The type of question (e.g., 'yesno', 'process', 'definition').

    Returns:
        bool: True if the database write was successful, False otherwise.
    """
    # we need to both run cursor and connection
    if not hasattr(self, '_DLM__cursor') or not self._DLM__conn:
        print("[SYSTEM]: Error - Cannot learn, database connection lost.")
        return False

    try:
        self._DLM__cursor.execute(
            """
            INSERT INTO knowledge_base (question, answer, category) 
            VALUES (?, ?, ?)
            ON CONFLICT(question) DO UPDATE SET 
                answer = excluded.answer,
                category = excluded.category
            """,
            (question, expectation, category)
        )

        self._DLM__conn.commit()
        return True
    
    except Exception as e:
        print(f"[SYSTEM]: Database Write Error in learn: {e}")
        return False