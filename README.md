# CMPSC_442

**CMPSC 442** is an Artificial Intelligence course offered at Pennsylvania State University. The course explores the fundamentals of AI, its historical development, core algorithms, and practical applications.

Throughout the semester, we studied key concepts such as:
- The history and evolution of Artificial Intelligence
- Types of Machine Learning models: Supervised and Unsupervised Learning
- The current trends in AI, including its integration with computer vision

## Projects

We applied core AI algorithms to classic problem-solving games.

### Project 1: Search Algorithms
Implemented basic search strategies to solve classic AI puzzles:

- **Breadth-First Search (BFS)**
- **Depth-First Search (DFS)**

Puzzles solved include:

- **N-Queens** — Place N queens on an N×N chessboard such that no two queens threaten each other. The algorithm must explore valid configurations by ensuring no row, column, or diagonal conflict.
  
- **Lights Out** — A grid-based puzzle where toggling a light also toggles its neighbors. The goal is to turn off all lights, requiring careful planning or brute-force search.

- **Linear Disk Movement** — A sequence of disks must be moved from the left side of a row to the right side under movement constraints. Disks can only move to adjacent empty spots or jump over one disk to an empty spot.

To run the **Lights Out** game with a GUI, use the following command:

```bash
python lights_out_gui.py rows cols
```


### Project 2: Heuristic Search
This project focuses on informed search algorithms, particularly:
- **Iterative Deepening Depth-First Search (IDDFS)**
- **A\* Search** with heuristic functions

We applied these algorithms to solve the following puzzles:

- **Tile Puzzle** — a classic sliding puzzle (e.g., 8-puzzle)
- **Grid Navigation** — pathfinding in a 2D grid with obstacles
- **Linear Disk Movement (Revisited)** — an optimization of the earlier version with heuristic improvements
- **Dominoes Game** — placing dominoes on a grid to satisfy constraints

To run the **Dominoes Game** with a GUI, use the following command

```bash
python homework2_dominoes_game_gui.py rows cols
```


### Project 3: Constraint Satisfaction — Sudoku Solver

In this project, we developed an AI-based solver for the classic **Sudoku** puzzle using **Constraint Satisfaction Problem (CSP)** techniques.

The solver includes:

- **AC-3 Inference Algorithm** — Enforces arc consistency by eliminating inconsistent values between variables (cells) connected by constraints.
- **Improved Inference** — Builds on AC-3 by applying rule-based logic to identify cells where a value must go due to uniqueness in a row, column, or 3×3 box.
- **Backtracking with Guessing** — If inference alone cannot solve the puzzle, the algorithm guesses possible values for ambiguous cells and recursively explores solutions.

The Sudoku board is parsed from an input file where:
- Digits represent known values
- Asterisks (`*`) represent unknown cells

The solver progressively narrows down possibilities using domain reduction and, if needed, explores branches with trial assignments.


### Project 4: Naive Bayes Spam Filter

This project implements a **Spam Filter** using the **Naive Bayes Classification** algorithm. It classifies emails as spam or ham (not spam) based on their word distributions.

Key components:

- **Token Extraction**: Each email is parsed and tokenized using Python's `email` module. Tokens are extracted from the email body, ignoring headers and formatting.
  
- **Training with Smoothing**: 
  - Emails are divided into spam and ham categories.
  - Word frequencies are collected from each class.
  - **Laplace smoothing** is applied to avoid zero probabilities for unseen words.
  - Log probabilities are computed for all tokens including an `<UNK>` token for out-of-vocabulary words.

- **Classification**: 
  - For a new email, the log probabilities for spam and ham are calculated using token counts and class priors.
  - The class with the higher log-probability is selected.

- **Interpretability**: 
  - The filter also provides the `most_indicative_spam(n)` and `most_indicative_ham(n)` functions to return the top `n` most informative words that strongly indicate either spam or ham. This is based on the relative likelihoods of words occurring in spam versus ham emails.

This project demonstrates the effectiveness of the Naive Bayes algorithm in real-world text classification tasks like spam detection, emphasizing both performance and interpretability.


### Project 5: Hidden Markov Models for POS Tagging

This project implements a **Hidden Markov Model (HMM)** to perform **Part-of-Speech (POS) tagging** on sentences using labeled training data.

The goal of this assignment was to gain practical experience working with probabilistic models for sequence labeling, particularly HMMs.

Key features include:

- **Training on Corpus Data**:  
  The model is trained on a labeled corpus of token=tag pairs to estimate:
  - **Initial probabilities** of each tag
  - **Transition probabilities** between tags
  - **Emission probabilities** of words given tags  
  All estimates use **Laplace smoothing** to handle unseen tokens and transitions.

- **Most Probable Tags (Naive Decoding)**:  
  Assigns the most likely tag to each token independently, based solely on emission probabilities. This approach is fast but ignores tag dependencies.

- **Viterbi Algorithm**:  
  Uses dynamic programming to compute the most likely tag sequence for a sentence by considering the full tag-transition structure of the HMM. It achieves high accuracy by modeling dependencies between tags.

- **Handling Unknown Words**:  
  Unseen tokens are mapped to a `<UNK>` token and handled with smoothed probabilities.

This implementation demonstrates how probabilistic models and dynamic programming can be combined to solve sequence prediction tasks in natural language processing.


### Project 6: Fine-Tuning with GRPO on SmolLM

This project involves fine-tuning a **causal language model** using **GRPO (Group Relative Policy Optimization)** on the `mlabonne/smoltldr` dataset, with a focus on **reinforcement learning from reward functions**.

We use a pre-trained model (`SmolLM-135M-Instruct`) and fine-tune it on CPU for efficient training on constrained devices.

#### Key Components:

- **Dataset**: `smoltldr` — a collection of prompts and concise TL;DR-style summaries.
- **LoRA (Low-Rank Adaptation)**: Lightweight fine-tuning method applied to selected attention modules (`q_proj`, `v_proj`).
- **Reward Function**: A custom reward function encourages summaries close to a target length (50 words) to simulate desirable generation behavior.
- **Safe Generation**: Patched `generate` method ensures no `NaN` values are produced using `InfNanRemoveLogitsProcessor`.
- **Training**:
  - Optimizer: `AdamW`
  - Prompt length: 256 tokens
  - Completion length: 64 tokens
  - Batch size: 8 (no FP16 for CPU)
  - Epochs: 1 (50 samples used for demonstration)
- **Evaluation**:
  - Model is tested on 50 held-out examples
  - Results are saved in `evaluation_results.json`

#### Logging:
- Logs include device info, training details, prompt examples, and evaluation results.
- Optionally integrates with **Weights & Biases (wandb)** for experiment tracking.

#### Output:
- Final model and tokenizer are saved to `./GRPO_mac/final_model`.

This project demonstrates practical fine-tuning with GRPO in a resource-constrained environment, highlighting reward-driven generation and efficient adaptation using LoRA.


### Reflection
I learned a variety of algorithms and its application to puzzle for the first half of the projects. It was fun because I could see the results instantly and it was quiet satisfying. 
However, the second half focused more on the real-world improvement and insights of AIs, which are much harder concepts. I learnerd how we can apply those algorithms from puzzle solutions to real Artificial Intelligence improvement. 
Overall, the class was meaningful and impressive.
