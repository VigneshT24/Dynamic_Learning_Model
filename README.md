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

**Key capabilities include:**

- **FAQ Handling** - Learns and responds to frequently asked questions based on the knowledge it has been trained on.
- **Chain-of-Thought (CoT) Reasoning** - Performs clear, step-by-step logic to solve non-ambiguous arithmetic, geometric, and unit conversion problems.
- **Custom Knowledge Integration** - DLM is fully extensible. You can initialize it with an empty SQL database and train it with your domain-specific knowledge.

Whether you're building a student support bot, a domain-specific assistant, or a computation system, DLM offers a flexible foundation to power your intelligent applications.

## Table of Contents

- [Installation](#installation)
- [Required Parameters](#required-parameters)
- [Usage](#usage)
- [Important Notices](#important-notices)
- [License](#license)
- [Disclaimer](#disclaimer)

## Installation

```bash
pip install dynamic-learning-model
```

> **Requirements:** Python 3.12.0 or higher is required to use this bot in your program. All required dependencies are installed automatically with the package.

## Required Parameters

The constructor requires passing in two parameters:

1. **Bot Mode**
   - `"learn"` - Enables training using the memory model. The bot can be updated with new information.
   - `"apply"` - The bot automatically switches between its "compute" and "memory" models depending on the query asked.
2. **Database Path** - An empty (or existing) SQLite database file used for training and as the bot's memory model.

The `ask()` method also requires two parameters:

1. **Query** - e.g., `"What is the definition of FAFSA?"`
2. **Display Thought** - `True` to display the bot's Chain-of-Thought reasoning, or `False` to suppress it.

## Usage

**`"learn"` mode** (training queries):

```python
from dlm import DLM

training_bot = DLM("learn", "college_knowledge.db")
training_bot.ask("What is FAFSA in college?", True)
```

**`"apply"` mode** (deployment / production use after training):

```python
from dlm import DLM

commercial_bot = DLM("apply", "college_knowledge.db")
commercial_bot.ask("What is the difference between FAFSA and CADAA in California?", False)

# or

commercial_bot.ask("Tell me the result for the following: 5 * 5 * 5 + 5 / 5", True)
```

### High-Level Pipeline

![DLM Pipeline](https://github.com/user-attachments/assets/e61d3f5d-87ca-4c81-bcb4-c28a0df65300)

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
