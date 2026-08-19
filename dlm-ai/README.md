<table>
  <tr>
    <td><img src="https://ik.imagekit.io/cqhzoyggfm/DLM%20Logo.png?updatedAt=1759635222204" width="90"></td>
    <td><h1>Dynamic Learning Model</h1></td>
  </tr>
</table>

[![PyPI version](https://img.shields.io/pypi/v/dynamic-learning-model.svg)](https://pypi.org/project/dynamic-learning-model/)
[![Python Version](https://img.shields.io/badge/python-3.12.0%2B-blue)](https://pypi.org/project/dynamic-learning-model/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Built With/Tech-Stack

**Python** · **Ollama Llama 3.2** · **SQLite** · **SymPy** · **SpaCy** · **LangGraph** · **LangChain**

## Overview

Dynamic Learning Model (DLM) is a locally hosted Python AI framework for building domain-specific assistants with persistent memory, symbolic mathematical reasoning, and human-in-the-loop learning.

**Important Architecture Note**: DLM acts as a backend engine, not a standalone chatbot, therefore, it doesn't generate a graphical chat interface. It processes queries and returns the answer, the thought process (if requested), and other structured information in a Python dictionary. It is the responsibility of the **implementor** to build the application loop, handle these states, interact with the user, and pass training data back to the bot via the `teach_memory()` and `teach_compute()` methods.

Additionally, DLM does not retrain the underlying LLM. Training operations modify DLM's local knowledge and computation databases, allowing the system to improve its responses without modifying the neural model weights.

**Key capabilities include:**

- **FAQ Handling** - Learns and responds to frequently asked questions based on the knowledge it has been trained on.
- **Symbolic Mathematical Reasoning** - Performs clear, step-by-step logic to solve numerical arithmetic, unit conversions, and advanced symbolic math (algebra, calculus, integrals) using SymPy and LangGraph.
- **Custom Knowledge Integration** - DLM is fully extensible. You can initialize it with an empty SQL database and train it with your domain-specific knowledge.
- **Local Privacy** - DLM runs 100% locally utilizing the Ollama inference engine, keeping all your data secure.

## Table of Contents

- [Why DLM?](#why-dlm)
- [High Level Architecture Diagram](#high-level-architecture-diagram)
- [Features](#features)
- [How Learning Works](#how-learning-works)
- [Prerequisites & Installation](#prerequisites--installation)
- [Initialization & Parameters](#initialization--parameters)
- [Response Architecture](#response-architecture)
- [Implementation Examples](#implementation-examples)
- [Training Guidelines](#training-guidelines)
- [Important Notices](#important-notices)
- [License](#license)
- [Disclaimer](#disclaimer)

## Why DLM?

Large language models are powerful general-purpose systems, but they aren't always ideal for domain-specific applications. They can hallucinate facts, perform unreliable arithmetic, and require external APIs when deployed through cloud-based services.

DLM was designed around a different approach: use an LLM for language understanding, persistent local memory for domain knowledge, and deterministic computational tools such as SymPy for mathematics.

The goal is to build a structured system that can be trained, corrected, inspected, and specialized for a particular domain.

## High Level Architecture Diagram

```text
User Query
    ↓
DLM Query Router (LLaMA)
    ↓
[Is it Math or Facts?]
   /              \
  ↓                ↓
COMPUTE ENGINE    MEMORY ENGINE
(LangGraph)       (SpaCy)
(SymPy)           (SQLite)
  ↓                ↓
   \              /
    ↓            ↓
Structured Result (Python Dictionary)
    ↓
Application Layer (Your Code)
```

## Features

* **Persistent Memory**: Stores domain-specific knowledge in SQLite.
* **Human-in-the-Loop Learning**: Users can correct answers and teach new information.
* **Symbolic Mathematics**: Uses SymPy for algebra, calculus, integration, and equation solving.
* **Hybrid Query Routing**: Routes queries between memory and computation systems.
* **Local LLM Inference:** Uses Ollama for locally hosted language models.
* **Local Data Storage**: Knowledge and training databases remain on the host machine.
* **Python API**: Designed as a backend engine that can be integrated into custom applications.
* **Domain Specialization**: Can be trained for organization-specific knowledge.

## How Learning Works

DLM uses a human-in-the-loop learning system. Instead of retraining the underlying LLM, DLM stores new knowledge and corrections in local SQLite databases.

### Memory Learning

When DLM doesn't know an answer, it can ask the implementor to provide one. The answer is then stored using `teach_memory()` and can be used for future questions.

```text
Question
   ↓
DLM doesn't know
   ↓
Human provides answer
   ↓
teach_memory()
   ↓
Saved to database
   ↓
Used in future questions
```

### Compute Learning

For mathematical questions, DLM uses an LLM to generate a formula. If the formula is incorrect, the implementor can provide a correction using `teach_compute()`.

```text
Math Question
   ↓
LLM generates formula
   ↓
SymPy calculates result (if applicable)
   ↓
Human verifies
   ↓
Correction saved if needed
```
> Note: DLM does not retrain the underlying LLM. Its learning comes from updating its local knowledge and computation databases.

## Prerequisites & Installation

**Critical Prerequisite:** This package utilizes local LLM inference to ensure complete data privacy and requires an external engine to run. 
Before installing DLM, you **must** install the Ollama engine for your operating system from [ollama.com](https://ollama.com). 

DLM will automatically handle booting the background server and downloading the required neural network models (`llama3.2` and `nomic-embed-text`) upon its first run.

Once Ollama is installed on your machine, install DLM via pip:
```bash
pip install dynamic-learning-model
```

> **Requirements:** Python 3.12.0 or higher is required. SpaCy's `en_core_web_lg` vector model will automatically download itself on first launch if not found.

## Initialization & Parameters

The constructor requires passing in up to two parameters:

1. **Bot Mode**
   - `"train_memory"` - Enables teaching capabilities for factual questions. The engine will request training when it encounters unknown queries.
   - `"train_compute"` - Enables teaching capabilities for the math engine. It allows you to correct the symbolic Python/SymPy formulas generated by the LLM.
   - `"apply"` - Deployment mode. The bot seamlessly hybrid-routes between its compute and memory models using auto-routing but will not prompt for database updates.
2. **Database Path (Optional)**
   - Absolute path to your SQLite database. This is optional; DLM automatically creates and uses `~/.dlm/dlm_database.db` and `~/.dlm/dlm_compute_model.db` in the user's home directory if not specified.

**`ask()` method parameters**:
   - `query` - The question you want DLM to answer (passed as a string).
   - `display_thought` - Whether or not you want DLM to return its internal Chain-of-Thought (passed as a boolean).

## Response Architecture
Calling `bot.ask(query, display_thought=True)` does not print directly to the console. It returns a structured Python dictionary that the implementor must handle.

**Expected Dictionary Keys:**
- `status` (str): The output state of the interaction (`resolved`, `needs_teaching`, `confirm_memory`, `confirm_compute`, or `refused`).
- `thought` (str): The step-by-step thought process of the DLM bot. Empty if `display_thought` is False.
- `answer`: (str): The final formulated answer, computation result, or fallback prompt.
- `context`: (dict): Metadata needed for database insertion (e.g., `special_stripped_query`, `generalized_query`, `var_num`).

## Implementation Examples
Below are the standard application loops an implementor should use to handle the states returned by DLM.

### 1. Training Factual Memory (`"train_memory"` mode)

```python
from dlm import DLM

# Initialize in memory training mode
bot = DLM("train_memory") 

while True:
    query = input("\nAsk a factual question: ")
    response = bot.ask(query, display_thought=True)

    if response["thought"]:
        print(response["thought"])

    # State Routing
    if response["status"] == "needs_teaching":
        answer = input(f"\nI don't know the answer. Please teach me: ")
        category = input("What category does that answer belong to? ")
        
        # Pass the extracted context to the memory training API
        bot.teach_memory(response["context"]["special_stripped_query"], answer, category)
        print("Knowledge base updated!")

    elif response["status"] == "confirm_memory":
        print(f"\n{response['answer']}")
        verify = input("Is my answer correct? Press enter to accept, or type the correct answer: ")
        
        if verify != "":
            category = input("What category does that new answer belong to? ")
            bot.teach_memory(response["context"]["special_stripped_query"], verify, category)
            print("Knowledge base updated with correction!")

    elif response["status"] in ["resolved", "refused"]:
        print(f"\n{response['answer']}")
```

### 2. Training the Math Engine (`"train_compute"` mode)

```python
from dlm import DLM

# Initialize in compute training mode
bot = DLM("train_compute") 

while True:
    query = input("\nEnter a math or calculus problem: ")
    response = bot.ask(query, display_thought=True)

    if response["thought"]:
        print(response["thought"])

    if response["status"] in ["confirm_compute", "needs_teaching"]:
        verify = input("\nIs this calculation correct? (Y/N): ")
        
        if verify.lower() == 'n':
            print(f"Extracted Variables: {response['context']['var_num']}")
            corrected = input("Enter the correct Python/SymPy formula using [x] variables (e.g., sp.diff([x0]*x, x)): ")
            
            # Pass the extracted context and new formula to the compute training API
            bot.teach_compute(response["context"]["generalized_query"], response["context"]["var_num"], corrected)
            print("Compute database permanently updated!")
            
    elif response["status"] in ["resolved", "refused"]:
        print(f"\n{response['answer']}")
```

### 3. Deployment (`"apply"` mode)

```python
from dlm import DLM

# Initialize in deployment mode
bot = DLM("apply")

while True:
    query = input("\nAsk anything (facts or math): ")
    response = bot.ask(query, display_thought=True)

    if response["thought"]:
        print(response["thought"])

    # In apply mode, the system auto-routes and does NOT ask the user for training data
    print(f"\n{response['answer']}")
```

###  4. Complete Implementation Example

```python
from dlm import DLM

def test_dlm_architecture():
    """Example Implementation."""

    print("\n\n=========================================")
    print("TRAIN_MEMORY TEST")
    print("=========================================\n\n")

    dlm_train_mem = DLM("train_memory")

    query = input("MEMORY TRAINING: ")
    print(f"\n[QUERY]: {query}")
    result = dlm_train_mem.ask(query, display_thought=True)
    print("\n\nStatus: ", result['status'], "\n\nAnswer: ", result['answer'], "\n\nThought: ", result['thought'], "\n\nContext: ", result['context'], "\n\n")

    print("\n\n=========================================")
    print("TRAIN_COMPUTE TEST")
    print("=========================================\n\n")

    dlm_train_comp = DLM("train_compute")

    query = input("COMPUTE TRAINING: ")
    print(f"[QUERY]: {query}")
    result = dlm_train_comp.ask(query, display_thought=True)
    print("\n\nStatus: ", result['status'], "\n\nAnswer: ", result['answer'], "\n\nThought: ", result['thought'], "\n\nContext: ", result['context'], "\n\n")

    print("\n\n=========================================")
    print("APPLY (PRODUCTION) TEST")
    print("=========================================\n\n")

    dlm_apply = DLM("apply")

    query = input("APPLY MODE: ")
    print(f"[QUERY]: {query}")
    result = dlm_apply.ask(query, display_thought=True)
    print("\n\nStatus: ", result['status'], "\n\nAnswer: ", result['answer'], "\n\nThought: ", result['thought'], "\n\nContext: ", result['context'], "\n\n")

def implementor_testing():
    print("========================================= IMPLEMENTOR INTERFACE =========================================")
    mode = input("\n\nMode (A = Apply, M = Train_Memory, C = Train_Compute): ").strip().lower()

    if mode == "a":
        mode = "apply"
    elif mode == "m":
        mode = "train_memory"
    elif mode == "c":
        mode = "train_compute"
    else:
        print("Proper mode not provided. Exiting the program.")
        return

    while True:
        q = input("ASK: ")

        if q.lower() == "switch to apply":
            mode = "apply"
            print("\nSuccessfully switched to 'APPLY' mode.")
            continue
        elif q.lower() == "switch to train memory":
            mode = "train_memory"
            print("\nSuccessfully switched to 'TRAIN_MEMORY' mode.")
            continue
        elif q.lower() == "switch to train compute":
            mode = "train_compute"
            print("\nSuccessfully switched to 'TRAIN_COMPUTE' mode")
            continue
        elif q.lower() == "quit":
            print("\nEnding the program.")
            break

        dlm_bot = DLM(mode)
        response = dlm_bot.ask(q, True)

        match response["status"].lower():
            case "resolved":
                print("\n\nTHOUGHT: ", response['thought'], "\n\nANSWER: ", response['answer'])
            case "refused":
                print("\n\nANSWER: ", response['answer'])
            case "confirm_memory":
                print("\n\nTHOUGHT: ", response['thought'])
                print("\n\nPROPOSED ANSWER: ", response['answer'])
                
                feedback = input("\nIs this the right answer? (Y/N): ").strip().upper()
                while feedback != "Y" and feedback != "N":
                    feedback = input("\nPlease only enter either 'Y' for yes or 'N' for no: ")
                if feedback == 'N':
                    print("\n[MEMORY CORRECTION MODE]")
                    correct_ans = input("\nEnter the correct answer: ").strip()
                    if correct_ans == "":
                        print("No correct answer provided. Nothing saved.")
                    else:
                        category = input("\nEnter the category (e.g., generic, yesno, definition): ").strip()
                        while category == "" or category not in ['generic', 'yesno', 'process', 'definition', 'deadline', 'location', 'eligibility']:
                            category = input("\nPlease enter a valid category: ")
                        
                        # grab the cleanly stripped query we saved in the context dict
                        query_to_teach = response['context'].get('special_stripped_query')
                        
                        success = dlm_bot.teach_memory(query_to_teach, correct_ans, category)
                        if success:
                            print("\n[SYSTEM LOG]: Memory successfully updated.")
                        else:
                            print("\n[SYSTEM LOG]: Failed to update memory.")
                        
            case "confirm_compute":
                print("\n\nTHOUGHT: ", response['thought'])
                print(f"\n\nFORMULA USED: {response['context'].get('formula')}")
                print(f"\nCALCULATED ANSWER: {response['context'].get('answer')}")
                
                feedback = input("\nIs this calculation correct? (Y/N): ").strip().upper()
                while feedback != "Y" and feedback != "N":
                    feedback = input("\nPlease only enter either 'Y' for yes or 'N' for no: ")
                if feedback == 'N':
                    print("\n[COMPUTE CORRECTION MODE]")
                    
                    # pull variables from context to show the user what [x] maps to what number
                    var_num = response['context'].get('var_num', [])
                    mapping = ", ".join(f"[x{i}] = {v}" for i, v in enumerate(var_num))
                    print(f"Extracted Variables: {mapping}")
                    
                    corrected_template = input("Enter the correct Python formula using [x] variables (e.g., [x0] * 9/5 + 32): ").strip()
                    if corrected_template == "":
                        print("No correction provided. Nothing saved.")
                    else:
                        generalized_query = response['context'].get('generalized_query')
                        
                        # send it back to the compute engine to overwrite the database and recalculate
                        new_state = dlm_bot.teach_compute(generalized_query, var_num, corrected_template)
                        
                        print(f"\nCorrected Final Answer: {new_state.get('answer')}")
                        print("[SYSTEM LOG]: Compute database permanently updated with your correction.")
                    
            case "needs_teaching":
                print("\n\nTHOUGHT: ", response['thought'])
                print("\n[SYSTEM LOG]: The bot does not know the answer to this query.")
                
                # defaulting to memory teaching when stumped
                correct_ans = input("Enter the expected answer: ").strip()
                if correct_ans == "":
                    print("No correct answer provided. Nothing saved.")
                else:
                    category = input("Enter the category (generic, location, deadline, process, yesno, eligibility, definition): ").strip()

                    while category == "" or category not in ['generic', 'yesno', 'process', 'definition', 'deadline', 'location', 'eligibility']:
                        category = input("\nPlease enter a valid category: ")
                    
                    query_to_teach = response['context'].get('special_stripped_query')
                    
                    success = dlm_bot.teach_memory(query_to_teach, correct_ans, category)
                    if success:
                        print("\n[SYSTEM LOG]: New knowledge successfully added to memory.")
                    else:
                        print("\n[SYSTEM LOG]: Failed to update memory.")

if __name__ == "__main__":
    implementor_testing()
```

## Training Guidelines

DLM's natural language generation relies on categorizing knowledge. When teaching the bot via `bot.teach_memory()`, the implementor must provide clean, raw facts and assign them to a specific category. 

DLM wraps these raw facts in dynamic templates. If you include conversational filler in your training data (e.g., training it with *"The deadline is December 15th"* instead of just *"December 15th"*), the bot will output grammatically awkward sentences.

**Expected Formats by Category:**

| Category | What to Train (Expected Format) | Example Training Input | Example Bot Output |
| :--- | :--- | :--- | :--- |
| **generic** | Answer can be in any format, no rules | Hi, I am DLM, hope you are doing well. | *"Hi, I am DLM, hope you are doing well."* |
| **yesno** | Start directly with "Yes" or "No", followed by the reason. | Yes, because of Rayleigh scattering. | *"Absolutely, yes, because of Rayleigh scattering."* |
| **process** | A list of steps separated strictly by **semicolons**. | Get bread; add peanut butter; eat it. | *"First, get bread. Next, add peanut butter. Lastly, eat it."* |
| **definition** | The raw, objective definition of the subject. | The process plants use to make food. | *"By definition, it is the process plants use to make food."* |
| **deadline** | The specific date, time, or timeframe. | December 15th. | *"The deadline is December 15th."* |
| **location** | A place, building, or directional instruction. | At the center of campus. | *"You can find it at the center of campus."* |
| **eligibility**| The specific conditions or prerequisites required. | you have a GPA over 3.5. | *"You qualify only if you have a GPA over 3.5."* |

## DB Browser for SQLite

It is highly recommended to download the DB Broswer (SQLite) application to view your database live to see how the queries are stored and potentially debug the database if it is corrupted. Additionally, you can directly write/overwrite in the application itself if you prefer that over using the terminal to train your DLM. Please follow this link to download DB Browser for SQLite: https://sqlitebrowser.org/dl/

## Troubleshooting

| Error / Issue | Solution |
| :--- | :--- |
| `FileNotFoundError: [WinError 2] The system cannot find the file specified` | **Ollama is not installed or not in your system PATH.** Download it from [ollama.com](https://ollama.com), install it, and restart your terminal. |
| `OSError: [E050] Can't find model 'en_core_web_lg'` | The automatic SpaCy downloader was blocked by your firewall or lacked permissions. Manually run: `python -m spacy download en_core_web_lg` |
| `NameError: name 'x' is not defined` | In `train_compute` mode, you entered a correction without using the `sp.` SymPy prefix. Ensure formulas look like `sp.solve(...)`. |
| The bot takes 10+ seconds to answer the first question | **This is normal.** The system is lazily loading the massive models into your RAM. Subsequent questions will be answered quicker. |

## Important Notices

1. **Training data quality matters.** DLM's accuracy in learning modes depends entirely on the consistency and clarity of the question/answer pairs it's trained with. Inconsistent category labeling can produce corrupted responses later.
2. **Database files are local and untracked.** DLM stores all trained knowledge in local SQLite files (`dlm_database.db` and `dlm_compute_model.db`). Back up these files regularly - there is no built-in cloud sync or recovery mechanism.
3. **Model loading behavior.** Underlying NLP and vector models (`en_core_web_lg`, `llama3.2`) are lazy-loaded and shared across instances. The first call in a session may take longer due to model loading into RAM; subsequent calls may be significantly faster (depending on your computer's specs).
4. **SymPy Compute Integration.** The compute engine utilizes the `sympy` library within a localized `eval()` environment to perform calculus, integration, and algebraic solving. Ensure corrected formulas in `train_compute` mode utilize standard `sp.` prefixes (e.g., `sp.solve()`, `sp.diff()`).

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Dynamic Learning Model (DLM) is provided **"as-is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the author be liable for any claim, damages, or other liability arising from the use of this software.

DLM **may produce inaccurate, incomplete, or unexpected responses**, particularly for ambiguous queries or insufficiently trained knowledge bases. **Do not rely on DLM's output for decisions involving safety, legal, medical, or financial consequences without independent verification.**

All data provided to DLM (training queries, database contents) is processed and stored **locally** on the host machine. DLM does not transmit user data externally, except for any underlying third-party model downloads required on the first run, which are subject to those providers' own terms.