# Quick Start Guide

This guide will help you get started with PyBanglaCodeAct in just a few minutes.

## Prerequisites

Before you begin, ensure you have:
- Python 3.9 or higher installed
- A CUDA-capable GPU (recommended, at least 16GB VRAM)
- Git installed

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/jahidulzaid/PyBanglaCodeActAgent.git
cd PyBanglaCodeActAgent
```

### Step 2: Create Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Your First Task

### Option 1: Use the Example Script

```bash
python example.py
```

This will run a simple example demonstrating how the agent works.

### Option 2: Process the Development Dataset

```bash
python PyBanglaCodeAct.py --input dev.csv --output submission.json
```

This will process all tasks in `dev.csv` and generate `submission.json`.

### Option 3: Custom Parameters

```bash
python PyBanglaCodeAct.py \
  --input dev.csv \
  --output my_submission.json \
  --model "Qwen/Qwen3-8B" \
  --retries 20 \
  --seed 42
```

## Understanding the Output

The agent will display:
- 🟢 **Green text**: Final answers and successful outputs
- 🟡 **Yellow text**: Warnings and agent thoughts
- 🔴 **Red text**: Errors
- 🟣 **Purple text**: Agent reasoning sections
- **Syntax-highlighted code**: Python code being executed

## Next Steps

1. **Explore Notebooks**: Check out the Jupyter notebooks in the repository:
   - `Sample_Code_Prompting_v2.ipynb` - Interactive prompting examples
   - `Sample_Code_Finetuning_v2.ipynb` - Fine-tuning experiments

2. **Evaluate Results**: Score your submissions:
   ```bash
   python scoring.py --gold dev.csv --pred submission.json --metric pass@1
   ```

3. **Customize Configuration**: Edit `config.py` to adjust:
   - Model parameters
   - Sampling settings
   - Timeout values
   - Retry logic

4. **Read the Documentation**: Check out:
   - [README.md](README.md) - Full documentation
   - [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
   - [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community guidelines

## Common Issues

### Out of Memory
- Reduce `max_model_len` in the code
- Use a smaller model
- Enable quantization (uncomment in config)

### Slow Inference
- Ensure CUDA is properly installed
- Use tensor parallelism for multi-GPU
- Enable prefix caching (default: enabled)

### Import Errors
- Make sure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```

## Getting Help

- **Issues**: [Open an issue](https://github.com/jahidulzaid/PyBanglaCodeActAgent/issues)
- **Discussions**: Join the community discussions
- **Documentation**: Read the [full README](README.md)

## What's Next?

- Try different models
- Experiment with hyperparameters
- Create your own Bangla programming tasks
- Contribute to the project!

Happy coding! 🚀
