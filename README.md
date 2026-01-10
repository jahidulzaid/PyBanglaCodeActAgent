# PyBanglaCodeAct

<p align="center">
  <img src="https://github.com/user-attachments/assets/711b6064-0844-490e-879d-697b12b0c488" alt="Project logo" width="200" height="200">
</p>

<p align="center">
  <a href="https://github.com/jahidulzaid/PyBanglaCodeActAgent/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python Version">
  </a>
  <a href="https://github.com/jahidulzaid/PyBanglaCodeActAgent/stargazers">
    <img src="https://img.shields.io/github/stars/jahidulzaid/PyBanglaCodeActAgent.svg" alt="Stars">
  </a>
  <a href="https://github.com/jahidulzaid/PyBanglaCodeActAgent/issues">
    <img src="https://img.shields.io/github/issues/jahidulzaid/PyBanglaCodeActAgent.svg" alt="Issues">
  </a>
</p>

## Table of contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Data formats](#data-formats)
- [Configuration and CLI](#configuration-and-cli)
- [Evaluation](#evaluation)
- [Repository layout](#repository-layout)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

PyBanglaCodeAct is a CodeAct/REACT-style agent for Bangla (Bengali) programming tasks. It accepts Bangla instructions, plans, generates Python code with multilingual LLMs (for example, Qwen3-8B), executes that code in a sandboxed REPL, and iteratively self-corrects through a Thought -> Code -> Observation loop. The agent reaches 94.0% pass@1 on the mHumanEval Bangla development set and demonstrates the effectiveness of execution-aware generation for low-resource languages.

## Features

- Multilingual support tuned for Bangla instructions
- Iterative self-correction with execution feedback
- vLLM-backed inference for fast generation
- Sandboxed Python REPL with timeouts
- Built-in scoring scripts for pass@1 evaluation
- Colorized logging with syntax highlighting
- Extensible architecture for custom prompts and models

## Prerequisites

- Python 3.9 or newer
- CUDA-capable GPU recommended (tested with Qwen3-8B; 16 GB VRAM or more works best)

## Installation

1) Clone the repository:
```bash
git clone https://github.com/jahidulzaid/PyBanglaCodeActAgent.git
cd PyBanglaCodeActAgent
```

2) Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

3) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4) (Optional) Install as an editable package to use the console script:
```bash
pip install -e .
```
This provides the `pybanglacodeact` entry point, equivalent to running `python PyBanglaCodeAct.py`.

## Quick start

- Run the agent on the provided development split:
```bash
python PyBanglaCodeAct.py --input dev.csv --output submission.json
```
This writes `submission.json` and a zipped copy `submission.zip`.

- Customize the run (model, retries, seed):
```bash
python PyBanglaCodeAct.py --input dev.csv --output submission.json --model "Qwen/Qwen3-8B" --retries 15 --seed 42
```

- Minimal example run:
```bash
python zero-result_baseline/example.py
```

For a step-by-step walkthrough, see `docs/QUICKSTART.md`.

## Data formats

- **Input CSV** must include:
  - `id`: unique integer identifier
  - `instruction`: Bangla task description
  - `test_list`: stringified Python list of assert statements

  Example:
  ```csv
  id,instruction,test_list
  1,"Write a function add(a, b) that returns their sum","['assert add(2, 3) == 5', 'assert add(-1, 1) == 0']"
  ```

- **Output JSON** produced by the agent:
  ```json
  [
    {
      "id": 1,
      "response": "def add(a, b):\n    return a + b"
    }
  ]
  ```

## Configuration and CLI

Key flags (identical for `python PyBanglaCodeAct.py` and `pybanglacodeact`):

| Option      | Default           | Description                                                  |
|-------------|-------------------|--------------------------------------------------------------|
| `--input`   | `dev.csv`         | Input CSV with `id`, `instruction`, `test_list`             |
| `--output`  | `submission.json` | Output JSON file name (a zip file with the same stem is created) |
| `--model`   | `Qwen/Qwen3-8B`   | Model name to load with vLLM                                 |
| `--retries` | `15`              | Maximum retries per task during decoding                    |
| `--seed`    | `42`              | Random seed for reproducibility                             |

Common tuning knobs such as max iterations, temperature, and timeouts are documented in `config.py` if you want to adjust the defaults in code.

## Evaluation

- Evaluate pass@1 against `dev.csv` from the project root:
```bash
python scoring/scoring.py
```
The script expects `dev.csv` and `submission.json` in the current directory. Adjust `reference_dir` and `prediction_dir` inside `scoring/scoring.py` if your files live elsewhere.

- An alternate evaluator is available at `test_phase/scoring_v2.py` (set `CSV_PATH` and `SUB_PATH` as needed).

## Repository layout

- `PyBanglaCodeAct.py` - main CLI for inference and submission generation
- `config.py` - reference defaults for model, sampling, and agent behavior
- `dev.csv`, `trial.csv` - provided datasets
- `docs/` - quick start and contributing guides plus architecture diagram
- `scoring/`, `test_phase/` - pass@1 evaluation scripts and test splits
- `zero-result_baseline/` - simple example script to sanity-check the agent
- `dev_phase/` - experimental runs and checkpoints used during development

## Results

### mHumanEval Bangla benchmark

| Model | Method | Pass@1 (Dev) | Pass@1 (Test) |
|-------|--------|--------------|---------------|
| **Qwen3-8B** | **BanglaCodeAct** | **94.0%** | **71.6%** |
| Qwen3-8B | Self-consistency | 90.0% | - |
| Qwen2.5-14B | BanglaCodeAct | 85.0% | - |
| DeepSeek-Coder-V2 | BanglaCodeAct | 71.4% | - |
| Llama-3.1-8B | Zero-shot | 45.0% | - |

Full comparisons are available in the accompanying paper.

## Contributing

Contributions are welcome. See `docs/CONTRIBUTING.md` for guidelines on reporting issues, adding features, and testing changes.

## Citation

If you use PyBanglaCodeAct in your research, please cite:

```bibtex
@article{islam2025pybangla,
  title={PyBangla at BLP-2025 Task 2: Enhancing Bangla-to-Python Code Generation with Iterative Self-Correction and Multilingual Agents},
  author={Islam, Jahidul and Ataullha, Md and Azad, Saiful},
  journal={arXiv preprint arXiv:2512.23713},
  year={2025}
}
```
