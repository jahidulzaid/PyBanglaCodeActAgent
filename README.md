# PyBanglaCodeAct

<p align="center">
  <img src="https://github.com/user-attachments/assets/711b6064-0844-490e-879d-697b12b0c488" alt="Profile image" width="200" height="200">
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

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#results">Results</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

**PyBanglaCodeActAgent** is a state-of-the-art CodeAct/REACT-style agent designed for **Bangla (Bengali) programming tasks**. It leverages multilingual Large Language Models (LLMs) to:

- Accept programming problems written in **Bangla**
- Generate structured plans and Python code using LLMs (e.g., Qwen3-8B)
- Execute code in a sandboxed Python REPL with real-time feedback
- Iteratively self-correct through a **Thought → Code → Observation** loop
- Achieve **94.0% pass@1** on the mHumanEval Bangla dataset (dev set)

This project demonstrates the effectiveness of agent-driven code generation with execution feedback for low-resource languages like Bangla.

## Features

- **Multilingual Support**: Optimized for Bangla programming instructions
- **Iterative Self-Correction**: Agent learns from execution errors and retries
- **High Performance**: Achieves state-of-the-art results on Bangla code generation
- **Safe Execution**: Sandboxed Python REPL with configurable timeouts
- **Comprehensive Evaluation**: Built-in scoring and test harness
- **Rich Logging**: Color-coded console output with syntax highlighting
- **Flexible Architecture**: Easy to extend and customize

## Installation

### Prerequisites

- **Python**: 3.9 or higher
- **GPU**: CUDA-capable GPU recommended (for LLM inference)
- **VRAM**: At least 16GB for Qwen3-8B model

### Method 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/jahidulzaid/PyBanglaCodeActAgent.git
cd PyBanglaCodeActAgent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Install as Package

```bash
pip install -e .
```

This will install the package in editable mode and make the `pybanglacodeact` command available globally.

## Quick Start

### Basic Usage

Run the agent on the provided development dataset:

```bash
python PyBanglaCodeAct.py --input dev.csv --output submission.json
```

📚 **For detailed setup instructions, see the [Quick Start Guide](QUICKSTART.md)**

### With Custom Parameters

```bash
python PyBanglaCodeAct.py \
  --input dev.csv \
  --output submission.json \
  --model "Qwen/Qwen3-8B" \
  --retries 15 \
  --seed 42
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `dev.csv` | Input CSV file with columns: id, instruction, test_list |
| `--output` | `submission.json` | Output JSON file for submission |
| `--model` | `Qwen/Qwen3-8B` | Model name to use for code generation |
| `--retries` | `15` | Maximum number of retries for each task |
| `--seed` | `42` | Random seed for reproducibility |

## Usage

### Data Format

The input CSV file should have the following columns:

- `id`: Unique identifier for each task
- `instruction`: Programming task description in Bangla
- `test_list`: Python test assertions to validate the solution

Example:

```csv
id,instruction,test_list
1,"একটি ফাংশন লিখুন যা দুটি সংখ্যার যোগফল রিটার্ন করবে।","assert add(2, 3) == 5\nassert add(-1, 1) == 0"
```

### Output Format

The output JSON file contains:

```json
[
  {
    "id": 1,
    "response": "def add(a, b):\n    return a + b"
  }
]
```

### Interactive Notebooks

Explore the example notebooks for interactive usage:

- **`Sample_Code_Prompting_v2.ipynb`**: Prompting strategies and agent workflow
- **`Sample_Code_Finetuning_v2.ipynb`**: Fine-tuning experiments and model training

### Scoring Your Results

Evaluate generated code against test cases:

```bash
python scoring.py --gold dev.csv --pred submission.json --metric pass@1
```

## Architecture

### Agent Workflow

```

```

### Key Components

- **`BanglaCodeAct`**: Main agent orchestrating the Thought-Code-Observation loop
- **`PythonREPL`**: Sandboxed Python execution environment with timeout protection
- **`llm_engine`**: LLM inference wrapper using vLLM for efficient generation
- **Custom Logger**: Color-coded output for better debugging

## Results

### mHumanEval Bangla Benchmark

| Model | Method | Pass@1 (Dev) | Pass@1 (Test) |
|-------|--------|--------------|---------------|
| **Qwen3-8B** | **BanglaCodeAct** | **94.0%** | **71.6%** |
| Qwen3-8B | Self-Consistency | 90.0% | - |
| Qwen2.5-14B | BanglaCodeAct | 85.0% | - |
| DeepSeek-Coder-V2 | BanglaCodeAct | 71.4% | - |
| Llama-3.1-8B | Zero-Shot | 45.0% | - |

*Full results and comparisons available in the paper.*


## Citation

If you use PyBanglaCodeAct in your research, please cite:

```bibtex

```