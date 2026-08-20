import os
import regex as re
import sqlite3
import json
import math
import sympy as sp
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings

# defining allowed mathematical environments for eval
allowed_env = {
        "sp": sp,
        "x": sp.Symbol('x'),
        "y": sp.Symbol('y'),
        "z": sp.Symbol('z'),
        "t": sp.Symbol('t')}

def get_db_path():
    """Creates and returns a newly created DB path at the user's home directory."""
    home_dir = os.path.expanduser("~")
    dlm_dir = os.path.join(home_dir, ".dlm")
    os.makedirs(dlm_dir, exist_ok=True)
    return os.path.join(dlm_dir, "dlm_compute_model.db")

COMPUTE_DB_PATH = get_db_path()

def setup_db():
    """After creating the DB, this method sets up the DB with column names."""
    conn = sqlite3.connect(COMPUTE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skills (
            id INTEGER PRIMARY KEY,
            query_mold TEXT,
            embedding TEXT,
            formula_template TEXT
        )
    ''')
    conn.commit()
    conn.close()

def cosine_similarity(vec_1, vec_2):
    """Calculating the Cosine Similarity."""
    dot_product = sum(a * b for a, b in zip(vec_1, vec_2))
    mag1 = math.sqrt(sum(a * a for a in vec_1))
    mag2 = math.sqrt(sum(b * b for b in vec_2))
    if mag1 == 0 or mag2 == 0: return 0.0
    return dot_product / (mag1 * mag2)

# each state is a node consisting of the following
class State(TypedDict):
    """Creating a LangGraph State Node"""
    query: str # e.g., convert 5 km to miles
    generalized_query: str # e.g., convert [x] km to [y]
    var_num: list
    formula: str
    formula_template: str
    answer: str
    route: str # e.g., "apply" or "learn" or "error"

def is_safe_formula(formula: str) -> bool:
    """
    Acts as a security gatekeeper to prevent malicious Python code execution.
    Blocks dunder methods and dangerous keywords from reaching eval().
    """
    dangerous_keywords = [
        "__", "import", "eval", "exec", "globals", "locals",
        "open", "getattr", "setattr", "delattr", "system", "os", "subprocess"
    ]
    
    # if any dangerous keyword is found in the string reject it
    for keyword in dangerous_keywords:
        if keyword in formula:
            return False
            
    return True

def normalize_and_extract(state: State) -> dict:
    """Extracts all numbers from query, generates the general query for model to study, and then returns the generalized query and the list of numbers."""
    user_query = state['query'].replace(',', '')
    pattern = r'-?(?:\d*\.\d+|\d+)'
    var_num = re.findall(pattern, user_query)
    generalized_query = re.sub(pattern, "[x]", user_query)

    return {'generalized_query': generalized_query, 'var_num': var_num}

def compute_answer(state: State) -> dict:
    """Uses the LangGraph State Node to get the formula, solve it, and then output the answer using a dictionary."""
    formula = state["formula"]

    # security check
    if not is_safe_formula(formula):
        return {'answer': "Error: Formula rejected due to potentially malicious or unauthorized syntax."}

    try:
        result = eval(formula, {"__builtins__": {}}, allowed_env)
        answer = str(result)
    except Exception as e:
        answer = f"Error: {str(e)}"

    return {'answer': answer}

def update_compute_database(generalized_query: str, var_num: list, corrected_template: str) -> dict:
    """An exposed method for inversion-of-control to update a math formula and recalculate if initial output is inaccurate."""

    # security check
    if not is_safe_formula(corrected_template):
        return {'formula': corrected_template, 'answer': "Error: Formula rejected due to potentially malicious or unauthorized syntax."}

    # update the database
    conn = sqlite3.connect(COMPUTE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE skills SET formula_template = ? WHERE query_mold = ?", 
        (corrected_template, generalized_query)
    )
    conn.commit()
    conn.close()

    # recalculate with the new formula
    formula = corrected_template
    for i, num in enumerate(var_num):
        formula = formula.replace(f"[x{i}]", num)

    try:
        answer = str(eval(formula, {"__builtins__": {}}, allowed_env))
    except Exception as e:
        answer = f"Error: {str(e)}"

    return {"formula": formula, "answer": answer}

def check_database(state: State) -> dict:
    """
    Method to check if compute database contains a similar query/formula when compared to the new query asked.
    Uses 'generate-and-verify' methodology to prevent hallucination by using a veto judge.
    """

    # initialize the embedder and llm (lazy load)
    llm = ChatOllama(model='llama3.2', base_url='http://localhost:11434')
    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    generalized_query = state["generalized_query"]
    var_num = state["var_num"]

    query_vector = embedder.embed_query(generalized_query)

    # creating a specific SQLite database for storing computation model intel, seperate from recall model
    conn = sqlite3.connect(COMPUTE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT query_mold, embedding, formula_template FROM skills")
    rows = cursor.fetchall()
    conn.close()

    best_score = 0.0
    best_formula = None
    best_mold = None

    for row in rows:
        db_vector = json.loads(row[1])
        score = cosine_similarity(query_vector, db_vector)

        if score > best_score:
            best_score = score
            best_formula = row[2] # this is the formula_template from each row
            best_mold = row[0] # the general query template stored in db

    if best_score > 0.85:
        veto_msg = [SystemMessage(content=(
            "You are an expert Semantic Routing Judge for a math computation system.\n"
            "Your objective is to determine if the 'User Query' and the 'Database Match' have the EXACT same core mathematical intent.\n\n"
            
            "RULES:\n"
            "1. Focus ONLY on mathematical verbs (add, subtract, convert) and units/direction (e.g., C to F).\n"
            "2. The '[x]' tokens represent generic number placeholders.\n"
            "3. Conversational wrappers ('hey', 'can you', 'please calculate') do NOT change the core mathematical intent.\n"
            "4. Ignore any differences in capitilization, punctuation, or spelling errors.\n"
            "5. CRITICAL: The User Query and Database Match MUST have the exact same number of [x] variables. If one has [x][x] and the other has [x], output <verdict>NO</verdict>.\n\n"
            
            "OUTPUT FORMAT:\n"
            "You must output your response strictly using these XML tags:\n"
            "<analysis> (Write a 1-2 sentence comparison of the core math intent) </analysis>\n"
            "<verdict> (Output exactly YES or NO) </verdict>\n\n"
            
            "EXAMPLES:\n"
            "User Query: hey dlm, can you add [x] and [x]\n"
            "Database Match: add [x] and [x]\n"
            "Output:\n"
            "<analysis>Both queries intend to perform an addition operation on two numbers. The conversational filler in the User Query does not alter the math.</analysis>\n"
            "<verdict>YES</verdict>\n\n"
            
            "User Query: convert [x] miles to km\n"
            "Database Match: convert [x] km to miles\n"
            "Output:\n"
            "<analysis>The User Query converts miles to kilometers, while the Database Match converts kilometers to miles. The direction is opposite.</analysis>\n"
            "<verdict>NO</verdict>"
        )), HumanMessage(content=f"User Query: {generalized_query}\nDatabase Match: {best_mold}")]

        veto_response = llm.invoke(veto_msg).content.strip()

        # COMMENT THIS OUT BEFORE PRODUCTION
        # print(f"\n[JUDGE LOG]:\n{veto_response}\n")

        if "<VERDICT>YES</VERDICT>" in veto_response.upper():
            formula = best_formula
            for i, num in enumerate(var_num):
                formula = formula.replace(f"[x{i}]", num) # type: ignore

            # if score is above 0.85 and NO mismatch
            return {'formula': formula, 'formula_template': best_formula, 'route': 'apply'}
        else:
            # print("\n[ROUTER LOG]: Database match vetoed due to directionality mismatch.")

            # if score is above 0.85 but is a mismatch
            return {'route': 'learn'}

    # if score is anyway below 0.85
    return {'route': 'learn'}

def route_query(state: State) -> str:
    """Method to help route the query using 'traffic-cop' method."""
    route = state["route"]

    if route == "apply":
        return "compute_answer"
    else:
        return "llm_reasoning"

def llm_reasoning(state: State) -> dict:
    """
    Uses Ollama LLM to analyze the query and generate a one-line expression to solve it.

    If query is commpletely new and requires a new formula, a request will be sent to the 
    compute database to generate and save the new details for next time.
    """

    # initialize the embedder and llm (lazy load)
    llm = ChatOllama(model='llama3.2', base_url='http://localhost:11434')
    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    query = state["query"]
    generalized_query = state["generalized_query"]
    var_num = state["var_num"]

    # ask LLM to solve the query
    msg = [
        SystemMessage(content=(
            "You are a Python math translator. Your ONLY job is to write a 1-line Python expression for the user's query.\n"
            "RULES:\n"
            "1. DO NOT calculate the final answer. Write the equation (e.g., write 5 + 5, NOT 10).\n"
            "2. For basic arithmetic, ONLY output valid Python operators (+, -, *, /, **).\n"
            "3. For symbolic math (calculus, algebra), use the `sp` prefix for SymPy functions (e.g., sp.diff, sp.integrate, sp.solve).\n"
            "4. The variables x, y, z, and t are already defined as SymPy symbols. Use them directly. To solve an equation like '2x = 10', format it as sp.solve(2*x - 10, x).\n"
            "5. Use 3.14159 or sp.pi for Pi.\n"
            "6. If the operation is division by zero, write out the division directly (e.g., 15 / 0).\n"
            "7. If calculating 2D straight-line distance, use the Pythagorean theorem: (a**2 + b**2)**0.5\n"
            "8. Do NOT perform unnecessary unit conversions if the input units already match (e.g., mAh divided by mA is just division).\n"
            "9. Ignore non-math text commands like 'print' or conversational filler.\n"
            "10. Do NOT wrap the formula in markdown code blocks (```).\n\n"
            "EXAMPLES:\n"
            "User: convert 100 celsius to fahrenheit\n"
            "REASONING: To convert Celsius to Fahrenheit, multiply by 9/5 and add 32.\n"
            "FORMULA: (100 * 9/5) + 32\n\n"
            "User: convert 100 fahrenheit to celsius\n"
            "REASONING: To convert Fahrenheit to Celsius, subtract 32 and multiply by 5/9.\n"
            "FORMULA: (100 - 32) * 5/9\n\n"
            "User: convert 99 km to miles\n"
            "REASONING: To convert kilometers to miles, multiply by 0.621371.\n"
            "FORMULA: 99 * 0.621371\n\n"
            "User: add -5 and 10\n"
            "REASONING: The user wants to add negative 5 and positive 10.\n"
            "FORMULA: -5 + 10\n\n"
            "User: add .5 and 3.1\n"
            "REASONING: The user wants to add 0.5 and positive 3.1\n"
            "FORMULA: 0.5 + 3.1\n\n"
            "User: subtract 10 from 50\n"
            "REASONING: The user wants 50 minus 10.\n"
            "FORMULA: 50 - 10\n\n"
            "User: what is 15 percent of 80\n"
            "REASONING: To find a percentage, divide the percentage by 100 and multiply by the target number.\n"
            "FORMULA: (15 / 100) * 80\n\n"
            "User: find the square root of 144\n"
            "REASONING: The square root is equivalent to raising a number to the power of 0.5.\n"
            "FORMULA: 144 ** 0.5\n\n"
            "User: area of a circle with radius 5\n"
            "REASONING: Area is Pi times the radius squared. Pi is 3.14159.\n"
            "FORMULA: 3.14159 * 5 ** 2\n\n"
            "User: distance of 15 on x and 20 on y\n"
            "REASONING: Use the Pythagorean theorem.\n"
            "FORMULA: (15 ** 2 + 20 ** 2) ** 0.5\n\n"
            "User: 5000 mAh battery, motors draw 250 mA, runtime in hours?\n"
            "REASONING: Divide capacity by draw to get runtime. Units match.\n"
            "FORMULA: 5000 / 250\n\n"
            "User: what is the derivative of 15x\n"
            "REASONING: The user wants the derivative of 15*x with respect to x.\n"
            "FORMULA: sp.diff(15*x, x)\n\n"
            "User: integrate x^2\n"
            "REASONING: The user wants the indefinite integral of x**2 with respect to x.\n"
            "FORMULA: sp.integrate(x**2, x)\n\n"
            "User: solve 2x + 5 = 15 for x\n"
            "REASONING: Set the equation to zero (2*x + 5 - 15) and use sp.solve.\n"
            "FORMULA: sp.solve(2*x + 5 - 15, x)"
        )), HumanMessage(content=query)
    ]
    response = llm.invoke(msg)

    raw = response.content
    formula = raw.split("FORMULA:")[-1].strip() if "FORMULA:" in raw else raw.strip()

    formula = formula.replace("```python", "").replace("```", "").strip()

    if "\n" in formula:
        formula = [line for line in formula.split("\n") if line.strip()][-1].strip()

    formula_temp = formula
    for i, num in enumerate(var_num):
        formula_temp = formula_temp.replace(num, f"[x{i}]", 1)

    query_vector = embedder.embed_query(generalized_query)

    conn = sqlite3.connect(COMPUTE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO skills (query_mold, embedding, formula_template) VALUES (?, ?, ?)",
                   (generalized_query, json.dumps(query_vector), formula_temp))

    conn.commit()
    conn.close()

    return {"formula": formula, 'formula_template': formula_temp}

workflow = StateGraph(State)

# establishing nodes
workflow.add_node("normalize", normalize_and_extract)
workflow.add_node("check_db", check_database)
workflow.add_node("llm_reasoning", llm_reasoning)
workflow.add_node("compute_answer", compute_answer)

# draw the edges
# by setting entry point, the starting node is normalized
workflow.set_entry_point("normalize")

# from normalize, it needs to go to check_db to check if the query already exists
workflow.add_edge("normalize", "check_db")

# traffic cop checking
workflow.add_conditional_edges("check_db", route_query)

# connect the rest of the path until the final execution
workflow.add_edge("llm_reasoning", "compute_answer")
workflow.add_edge("compute_answer", END)

setup_db()

dlm_compute_engine = workflow.compile()
