<table>
  <tr>
    <td><img src="https://ik.imagekit.io/cqhzoyggfm/DLM%20Logo.png?updatedAt=1759635222204" width="90"></td>
    <td><h1>Dynamic Learning Model</h1></td>
  </tr>
</table>

[![PyPI version](https://img.shields.io/pypi/v/dynamic-learning-model.svg)](https://pypi.org/project/dynamic-learning-model/)
[![Python Version](https://img.shields.io/badge/python-3.12.0%2B-blue)](https://pypi.org/project/dynamic-learning-model/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

The Dynamic Learning Model (DLM) is a hybrid, domain-specific AI system designed to learn, adapt, and respond intelligently to user queries. It combines natural language understanding with structured reasoning, continually improving as it is trained.

**Important Architecture Note**: DLM acts as a backend engine, not a standalone chatbot. It processes queries and returns the answer, the thought process (if requested), and other structured information in a Python dictionary. It is the responsibility of the **implementor** to build the application loop, handle these states, interact with the user, and pass training data back to the bot via the `teach()` method.

**Key capabilities include:**

- **FAQ Handling** - Learns and responds to frequently asked questions based on the knowledge it has been trained on.
- **Chain-of-Thought (CoT) Reasoning** - Performs clear, step-by-step logic to solve non-ambiguous arithmetic, geometric, and unit conversion problems.
- **Custom Knowledge Integration** - DLM is fully extensible. You can initialize it with an empty SQL database and train it with your domain-specific knowledge.

Whether you're building a student support bot, a domain-specific assistant, or a computation system, DLM offers a flexible foundation to power your intelligent applications.

## Table of Contents

- [Installation](#installation)
- [Initialization & Parameters](#initialization_&_parameters)
- [Response Architecture](#response_architecture)
- [Implementation Example](#implementation_example)
- [Training Guidelines](#training_guidelines)
- [Important Notices](#important_notices)
- [License](#license)

## Installation

```bash
pip install dynamic-learning-model
```

> **Requirements:** Python 3.12.0 or higher is required to use this bot in your program. All required dependencies are installed automatically with the package.

## Initialization & Parameters

The constructor requires passing in two parameters:

1. **Bot Mode**
   - `"learn"` - Enables the memory model's teaching capabilities. The engine will request training when it encounters unknown queries.
   - `"apply"` - Deployment mode. The bot switches between its compute and memory models but will not prompt for database updates.
2. Database Path (Optional)
   - Absolute path to your SQLite database. This is optional; DLM automatically creates and uses `~/.dlm/dlm_database.db` in the user's home directory ONLY IF the user didn't specify their own database file.
3. `ask()` method parameters
   - `query` - The question that you want DLM to answer; must be passed as a string
   - `display_thought` - Whether or not you want DLM to return its thought process (chain of thought); must be passed as a boolean

## Response Architecture
Calling `bot.ask(query, display_thought=True)` does not print to the console. Instead, it returns a structured Python dictionary that the implementor must handle.

Expected Dictionary Keys:
- `**status**` (str): The output state of the interaction (`resolved`, `needs_teaching`, `confirm_teaching`, or `refused`)
- `**thought**` (str): The step-by-step thought process of the DLM bot (a.k.a. its chain of thought (CoT)). This is empty if `display_thought` is False.
- `**answer**`: (str): The final formulated answer or fallback prompt
- `**context**`: (dict): Metadata, including the crucial `special_stripped_query` needed for database insertion

An example output of printing the entire dictionary in the terminal:
```bash
{
    "status": "resolved",
    "thought": "I am looking through the database for matching keywords...\nI found a high-confidence match for the category 'process'.\nLet me format this using the proper template.",
    "answer": "First, click 'Forgot Password'. Next, check your email for the link. Lastly, create a new password.",
    "context": {
        "special_stripped_query": "how reset password",
        "best_match_answer": "click 'Forgot Password'; check your email for the link; create a new password"
    }
}
```

## Implementation Example
Below is the standard application loop an implementor should use to handle the states returned by DLM.

**`"learn"` mode** (training queries):

```python
from dlm import DLM

# Initialize in learn mode
bot = DLM("learn", "knowledge.db") # initializing with a custom DB is optional

while True:
    query = input("\nAsk a question or enter a math problem: ")
    
    # 1. Ask the engine
    response = bot.ask(query, display_thought=True)

    if response["thought"]:
        print(response["thought"])

    # 2. State Routing
    if response["status"] == "needs_teaching":
        # The bot doesn't know the answer. The implementor must ask the user for it.
        answer = input(f"\nI don't know the answer to \"{query}\". Please teach me: ")
        category = input("What category does that answer belong to? ")
        
        # IMPORTANT: Always use special_stripped_query to save to the database!
        bot.teach(response["context"]["special_stripped_query"], answer, category)
        print("Knowledge base updated!")

    elif response["status"] == "confirm_teaching":
        # The bot guessed the answer but wants verification.
        print(f"\n{response['answer']}")
        verify = input("Is my answer correct? Press enter to accept, or type the correct answer: ")
        
        if verify != "":
            category = input("What category does that new answer belong to? ")
            bot.teach(response["context"]["special_stripped_query"], verify, category)
            print("Knowledge base updated with correction!")

    elif response["status"] == "resolved":
        # The bot successfully answered from memory or computed the math.
        print(f"\n{response['answer']}")
        
    elif response["status"] == "refused":
        # The query was inappropriate, empty, or violated guardrails.
        print(f"\n{response['answer']}")
```

**`"apply"` mode** (deployment/production use after training):

```python
from DLM import DLM

# Initialize in apply mode (deployment/production)
bot = DLM("apply", "knowledge.db") # initializing with a custom DB is optional

while True:
    query = input("\nAsk a question or enter a math problem: ")
    
    # 1. Ask the engine
    response = bot.ask(query, display_thought=True)

    # Display the bot's Chain-of-Thought reasoning (if any)
    if response["thought"]:
        print(response["thought"])

    # 2. State Routing for Apply Mode
    if response["status"] == "resolved":
        # The bot successfully computed the math or retrieved an answer from memory.
        print(f"\n{response['answer']}")
        
    elif response["status"] in ["needs_teaching", "confirm_teaching"]:
        # In 'apply' mode, we do NOT prompt the user to teach the bot.
        # We simply output the graceful fallback message the bot generated.
        print(f"\n{response['answer']}")
        
    elif response["status"] == "refused":
        # The query was inappropriate, empty, or violated guardrails.
        print(f"\n{response['answer']}")
```

## Training Guidelines

DLM's natural language generation relies on categorizing knowledge. When teaching the bot (via `bot.teach()`), the implementor must provide clean, raw facts and assign them to a specific category. 

DLM wraps these raw facts in dynamic templates. If you include conversational filler in your training data (e.g., training it with *"The deadline is December 15th"* instead of just *"December 15th"*), the bot will output grammatically awkward sentences like *"The deadline is the deadline is December 15th"*.

**Expected Formats by Category:**

| Category | What to Train (Expected Format) | Example Training Input | Example Bot Output |
| :--- | :--- | :--- | :--- |
| **yesno** | Start directly with "Yes" or "No", followed by the reason. | Yes, because of Rayleigh scattering. | *"Absolutely, because of Rayleigh scattering."* |
| **process** | A list of steps separated strictly by **semicolons**. | Get bread; add peanut butter; eat it. | *"First, get bread. Next, add peanut butter. Lastly, eat it."* |
| **definition** | The raw, objective definition of the subject. | The process plants use to make food. | *"By definition, it is the process plants use to make food."* |
| **deadline** | The specific date, time, or timeframe. | December 15th. | *"The deadline is December 15th."* |
| **location** | A place, building, or directional instruction. | In the center of campus. | *"You can find it at the center of campus."* |
| **eligibility**| The specific conditions or prerequisites required. | you have a GPA over 3.5. | *"You qualify only if you have a GPA over 3.5."* |

## Important Notices

1. **Training data quality matters.** DLM's accuracy in `"learn"` mode depends entirely on the consistency and clarity of the question/answer pairs it's trained with. Inconsistent category labeling or vague phrasing during training can produce inaccurate or corrupted responses later.
2. **Database files are local and untracked.** DLM stores all trained knowledge in the SQLite file you provide. Back up this file regularly - there is no built-in cloud sync, versioning, or recovery mechanism.
3. **Model loading behavior.** Underlying NLP and transformer models (spaCy, HuggingFace) are lazy-loaded and shared across instances. The first call in a session may take longer due to model loading; subsequent calls are significantly faster.
4. **Compute mode limitations.** Chain-of-Thought computation is designed for clear, non-ambiguous arithmetic, geometric, and unit-conversion problems. Ambiguous or multi-interpretation queries may produce incorrect results - always verify outputs for critical use cases.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Disclaimer

Dynamic Learning Model (DLM) is provided **"as-is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the author be liable for any claim, damages, or other liability arising from the use of this software.

DLM **may produce inaccurate, incomplete, or unexpected responses**, particularly for ambiguous queries or insufficiently trained knowledge bases. **Do not rely on DLM's output for decisions involving safety, legal, medical, or financial consequences without independent verification.**

All data provided to DLM (training queries, database contents) is processed and stored **locally** on the host machine. DLM does not transmit user data externally, except for any underlying third-party model downloads (e.g. HuggingFace, spaCy) required on first run, which are subject to those providers' own terms.
