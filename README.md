# PyBanglaCodeActAgent

<p align="center">
  <img src="https://github.com/user-attachments/assets/711b6064-0844-490e-879d-697b12b0c488" alt="Profile image" width="200" height="200">
</p>

## What is this project?

PyBanglaCodeActAgent implements **BanglaCodeAct** — a CodeAct/REACT-style agent that:

* Accepts programming problems written in **Bangla** (Bengali),
* Prompts a multilingual LLM (e.g., Qwen3-8B) to produce a plan/thought and Python code,
* Executes the code in a sandboxed REPL, captures execution feedback (errors / test failures),
* Iteratively self-corrects (Thought → Code → Observation loop) until tests pass or a maximum iteration limit is reached.

This approach sets a new benchmark for Bangla→Python on the mHumanEval dataset (reported pass@1: **94.0%** on dev with Qwen3-8B using the BanglaCodeAct agent), and demonstrates the value of multi-agent prompting + iterative execution feedback for low-resource languages (paper authors: Jahidul Islam, Md Ataullha, Saiful Azad). The paper text supplied with the request was used to compose the summary above and other descriptions.

## Highlights / Features

* Agent-driven code generation with an internal Bangla-language **Thought** step (explain plan in Bangla), followed by code generation and execution-based **Observation** feedback.
* Safe retry handler (`safe_run`) and configurable iteration limits for better robustness.
* Scripts/notebooks for:

  * prompting (zero-/few-shot and agent loop),
  * finetuning (if you have model resources),
  * scoring (compute pass@1 / test-based evaluation),
  * running evaluation on the provided `dev.csv` (mHumanEval-style examples).
* Example notebooks included: `Sample_Code_Prompting_v2.ipynb`, `Sample_Code_Finetuning_v2.ipynb`.
* Submission artifacts included (`submission.json`, `submission.zip`) for shared-task style evaluation.

## Repo structure

*derived from the repo index — open the repository to inspect full contents.* ([GitHub][1])

* `README.md` — (this file in the repo; note: you are currently reading a generated README).
* `Sample_Code_Prompting_v2.ipynb` — example prompting workflow notebook.
* `Sample_Code_Finetuning_v2.ipynb` — notebook showing fine-tuning experiments.
* `dev.csv` — development examples (mHumanEval-like Bangla problems + tests).
* `finetune.py` — (script) fine-tuning helper (likely uses Hugging Face / transformers training loop).
* `prompt.py` — (script) prompting / agent loop orchestration (core of BanglaCodeAct behaviour).
* `scoring.py` — (script) scoring / test harness for pass@1 / pass@k computation.
* `test.py` — (script) small runner / sanity tests for generated code or agent.
* `submission.json` / `submission.zip` — submission files for the shared task.
* `*.csv` — data files / variants.

## Requirements

The repository does not appear to ship an exact `requirements.txt`. Based on the code artifacts and typical stacks for this work, you will likely need:

* Python 3.9+ (3.10 recommended)
* `pandas`
* `numpy`
* `pytest` (if tests)
* `transformers` (if finetuning or model wrappers used)
* `torch` and/or `accelerate` (if running finetuning or PyTorch-based inference)
* `vllm` (the authors mention vLLM for inference)
* `tqdm`
* a sandboxed Python execution helper (the repo likely includes its own)
* Optional: `datasets`, `huggingface_hub` if you load models from HF Hub

Tip: create a virtualenv and then `pip install` what you need. If you want, I can scan the repository for `import` lines and produce an exact `requirements.txt`.

## Quick start — suggested workflow

> Before running anything, inspect the top of `prompt.py` and `test.py` to confirm arguments and paths.

1. Clone the repo:

```bash
git clone https://github.com/jahidulzaid/PyBanglaCodeActAgent.git
cd PyBanglaCodeActAgent
```

2. Create and activate venv (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate    # Windows PowerShell
pip install --upgrade pip
```

3. Install likely dependencies (adjust as needed):

```bash
pip install pandas numpy torch transformers vllm tqdm
```

4. Run the prompting / agent script (example — check `prompt.py` for exact flags):

```bash
python prompt.py --input dev.csv --model qwen3-8b --output submission.json --agent-mode banglacodeact
```

*(This is a suggested command — open `prompt.py` to confirm available CLI args.)*

5. Score generated submissions:

```bash
python scoring.py --gold dev.csv --pred submission.json --metric pass@1
```

6. Explore the example notebooks:

* `Sample_Code_Prompting_v2.ipynb` shows how to run the agent interactively and inspect Thought / Code / Observation cycles.
* `Sample_Code_Finetuning_v2.ipynb` shows an example fine-tuning workflow (requires compute).

## Reproducing the paper's experiments (high-level)

The paper reports experiments on the **mHumanEval** dataset for Bangla NL2Code and compares several models & prompting strategies. If you want to reproduce the results:

1. Prepare mHumanEval/Bangla dataset (the repo includes `dev.csv` which is likely the dev split used).
2. Select a multilingual LLM (the paper found **Qwen3-8B** performed best for BanglaCodeAct).
3. Use the agent-loop in `prompt.py` (Thought → Code → Observation). Key ideas:

   * Generate a Bangla thought/plan before writing code.
   * Enclose generated code in a code block and run tests immediately in a sandboxed REPL.
   * Feed execution trace/error messages back into the model; repeat until test pass or max iterations.
4. Use `scoring.py` to compute pass@1 (paper uses pass@1 as primary metric).
5. The paper mentions inference settings used (e.g., max_tokens, temperature, top-p, repetition_penalty, timeouts, retries); tune your inference engine to reproduce behaviour. (Parameters summarized in the paper text you provided.)

**Note:** Running the exact experiments in the paper will require the model checkpoints (Qwen3-8B etc.), the same vLLM configuration, and sufficient GPU memory. The paper also used vLLM inference engine and tensor parallelism.

## Results

* **BanglaCodeAct (Qwen3-8B)** — pass@1 reported: **94.0%** (development) and **71.6%** (blind test set).
* The paper compares Zero-Shot, Few-Shot, Self-Consistency, and other baselines across models (Llama-3.1-8B, Qwen variants, DeepSeek-Coder, TigerLLM, etc.). See the paper excerpt for the full table of results.

## How the agent works (concise)

1. **Thought**: model writes its plan/intent in Bangla (explain reasoning).
2. **Code**: model generates Python code corresponding to the plan.
3. **Observation**: sandbox executes code. Execution result (tracebacks, failed assertions) is collected.
4. **Self-correct**: agent re-prompts itself with the Observation and revised Thought; loop repeats.

The implementation includes retry/safe-run logic and timeout controls to prevent runaway execution.

## Files of interest (what to open first)

* `prompt.py` — core driver for generation & agent loop.
* `scoring.py` — evaluator for test cases / pass@k computation.
* `test.py` — quick-run harness / unit tests for the agent.
* `dev.csv` / `trial.csv` — examples of input problem format (Bangla instruction + tests).
* `Sample_Code_Prompting_v2.ipynb` — readable, runnable notebook for experimentation.

## Citation (paper)
