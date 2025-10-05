![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=SQLite&logoColor=white)
![HuggingFace Transformers](https://img.shields.io/badge/-HuggingFace-3B4252?style=flat&logo=huggingface&logoColor=)

![DLM Logo](https://ik.imagekit.io/cqhzoyggfm/DLM%20Logo.png?updatedAt=1759635222204)

# Dynamic Learning Model
**ABOUT**:

The Dynamic Learning Model (DLM) is a hybrid AI system designed to learn, adapt, and intelligently respond to user queries. It combines natural language understanding with structured reasoning, continually improving as it is trained.

Key capabilities include:

* FAQ Handling: Learns and responds to frequently asked questions based on the knowledge it has been trained on.

* Chain-of-Thought (CoT) Reasoning: Performs clear, step-by-step logic to solve non-ambiguous arithmetic, geometric, and unit conversion problems.

* Custom Knowledge Integration: DLM is fully extensible. You can initialize it with an empty SQL database and train it with your domain-specific knowledge.

Whether you're building a student support bot, a domain-specific assistant, or a computation system, DLM offers a flexible foundation to power your intelligent applications

**REQUIRED PARAMETERS**:
* The constructor requires passing in two parameters:
  - Bot Mode:
      - 'learn' = Enables training using the memory model. The bot can be updated with new information,
      - 'apply' = The bot automatically switches between its "compute" and "memory" model depending on the query asked
  - Empty SQL Database for training the bot with queries and for the memory model
* The ask() method also requires passing in two parameters:
  - Query: "What is the definition of FAFSA" (as an example)
  - Display Thought: "True" to allow the bot's Chain of Thought to be displayed, or else "False"

**GET STARTED**:
* To install, run: 
```bash
pip install dynamic-learning-model
```
* ***Python 3.12 or higher is required to use this bot in your program***

('learn' mode [training queries])
* You can find the training password in the ```__trainingPwd``` variable defined within the DLM.py file
```python
from dlm import DLM

training_bot = DLM("learn", "college_knowledge.db")

training_bot.ask("What is FAFSA in college?", True)
```

('apply' mode [deployment/production use after training])
```python
from dlm import DLM

commercial_bot = DLM("apply", "college_knowledge.db")

commercial_bot.ask("What is the difference between FAFSA and CADAA in California?", False)

# or

commercial_bot.ask("Tell me the result for the following: 5 * 5 * 5 + 5 / 5", True)
```

**HIGH-LEVEL PIPELINE VISUAL**:

![image](https://github.com/user-attachments/assets/e61d3f5d-87ca-4c81-bcb4-c28a0df65300)

