![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=SQLite&logoColor=white)

# Dynamic Learning Model
The Dynamic Learning Model (DLM) is a hybrid AI system designed to learn, adapt, and intelligently respond to user queries. It combines natural language understanding with structured reasoning, improving over time as it interacts with users.

Key capabilities include:

* FAQ Handling: Learns and responds to frequently asked questions based on the knowledge it has been trained on.

* Chain-of-Thought (CoT) Reasoning: Performs clear, step-by-step logic to solve non-ambiguous arithmetic and unit conversion problems.

* Custom Knowledge Integration: DLM is fully extensible. You can initialize it with an empty SQL database and train it with your own domain-specific knowledge.

Whether you're building a student support bot, a domain-specific assistant, or an adaptive Q&A system, DLM offers a flexible foundation to power your intelligent applications

* This model uses SpaCy, SQLite, & NLTK for many of its functions

NOTICE: 
* This is a public package. To install it, run: 
```bash
pip install dynamic-learning-model
```
* The training password is: 371507
* ***You must have Python 3.12 or higher to run this bot***

Getting Started:

(Experimental 'e' mode [computation queries])
```python
from dlm import DLM

computation_bot = DLM("e", "college_knowledge.db")

computation_bot.ask("Compute the following: 5 * 5 * 5 + 5 / 5", False)
```

(Training 't' mode [training queries])
```python
from dlm import DLM

computation_bot = DLM("t", "college_knowledge.db")

computation_bot.ask("What is FAFSA in college?", False)
```

(Commercial 'c' mode [deployment/production use after training])
```python
from dlm import DLM

computation_bot = DLM("c", "college_knowledge.db")

computation_bot.ask("What is the difference between FAFSA and CADAA in California?", False)
```

![image](https://github.com/user-attachments/assets/340dc69a-8374-45df-ac1e-82431c5111f2)


![image](https://github.com/user-attachments/assets/422f1045-07bc-4ddf-ae28-9f5731324b93)
