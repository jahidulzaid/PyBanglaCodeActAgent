import re
from collections import Counter
import vllm
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import set_seed

# model = "md-nishat-008/TigerLLM-1B-it"
model = "Qwen/Qwen2.5-Coder-7B-Instruct"

llm = vllm.LLM(
    model,
    # quantization="awq",
    max_model_len=4096,
    enable_prefix_caching=True,
    tensor_parallel_size=torch.cuda.device_count(),
)

tokenizer = llm.get_tokenizer()

def llm_engine(messages, stop_sequences=None, start_sequence=None) -> str:
    sampling_params = vllm.SamplingParams(
        temperature=0,
        # use_beam_search=True,
        # num_beams=3,
        best_of=1,
        max_tokens=2048,
        stop=stop_sequences,
        include_stop_str_in_output=True,
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if start_sequence:
        prompt += start_sequence
    output = llm.generate([prompt], sampling_params, use_tqdm=False)
    response = output[0].outputs[0].text

    if start_sequence:
        response = start_sequence + response
    return response

def extract_answer(response):
    # Regex pattern to match content inside \boxed{...}
    pattern = r'\\boxed{(-?\d+)}'

    # Search for the match
    match = re.search(pattern, response)

    if match:
        answer = int(match.group(1))  # Get the content inside the curly braces
    else:
        answer = -1
    return answer


def cot_sc(question: str, num_paths=16):
    sampling_params = vllm.SamplingParams(
        n=num_paths,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=2048
    )

    prompt = question
    messages = [
        {"role": "system", "content": "Please reason step by step in English, and put your final answer within \\boxed{}."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = llm.generate([text], sampling_params, use_tqdm=False)
    outputs = [output.text for output in outputs[0].outputs]
    answers = [extract_answer(output) for output in outputs]
    answers = [answer for answer in answers if answer >= 0]

    if answers:
        answer, _ = Counter(answers).most_common(1)[0]
    else:
        answer = 0

    return answer

CODEACT_PROMPT = """
You are a helpful coding assistant assigned to solve algorithmic problems in Python.

For each row in the dataset, you will be given:
- An **instruction** describing the task.
- A **test_list** (Python assertions).

**Your Workflow for each task:**

1. **Thought Process**:
   Explain your reasoning before coding.
   - Wrap your explanation in `<thought>` tags.
   Example: `<thought>I need to compute the smallest number divisible by all numbers from 1 to n. I can use LCM iteratively.</thought>`

2. **Write Python Code**:
   Implement the python program according to the instruction. Must use the exact given name for Class, Object and function name.
   - Place your implementation and provided tests in `<code>` tags.
   Example:
<code>

</code>

3. **Observation**:
After executing, confirm if all tests passed or debugging is needed.
Example: `<observation>All tests passed successfully.</observation>`

4. **Final Answer**:
Provide only the clean python program (without test assertions).
- Wrap in `<answer>` tags.
Example:
<answer>

</answer>
"""


import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    bold_yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_green = "\x1b[32;20;1m"
    bold_red = "\x1b[31;1m"
    bold_white = "\x1b[37;1m"
    orange = "\x1b[38;5;214m"
    bold_orange = "\x1b[38;5;214;1m"
    reset = "\x1b[0m"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: format,
        logging.WARNING: bold_yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
        31: reset + format + reset,
        32: green + format + reset,
        33: bold_green + format + reset,
        34: bold_white + format + reset,
        35: orange + format + reset,
        36: bold_orange + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger(__name__)
logger.propagate = False
ch = logging.StreamHandler()
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

import ast


def execute(script, globals=None, locals=None):
    """Execute a script and return the value of the last expression."""
    if globals is None:
        globals = {}
    if locals is None:
        locals = globals

    # Parse the script into AST nodes
    stmts = list(ast.iter_child_nodes(ast.parse(script)))
    if not stmts:
        return None

    if isinstance(stmts[-1], ast.Expr):
        # The last statement is an expression; we evaluate it
        if len(stmts) > 1:
            # Execute all but the last statement
            exec(compile(ast.Module(body=stmts[:-1], type_ignores=[]), filename="<ast>", mode="exec"), globals, locals)
        # Evaluate the last expression
        return eval(compile(ast.Expression(body=stmts[-1].value), filename="<ast>", mode="eval"), globals, locals)
    else:
        # If the last statement is not an expression, just execute the entire code
        exec(script, globals, locals)
        return None

import signal


class PythonREPL:
    def __init__(self, timeout=None):
        self.state = {}
        self.print_output = []
        self.timeout = timeout  # Set a default timeout value

    def _handle_timeout(self, signum, frame):
        raise TimeoutError(f"Exceeded the time limit of {self.timeout} seconds.")

    def run(self, code):
        # Reset print_output for each run
        self.print_output = []

        # Prepare the environment for execution
        env = self.state.copy()  # Create a local copy of the state

        # Define a custom print function to capture print statements
        def print_capture(*args, **kwargs):
            self.print_output.append(" ".join(map(str, args)))

        # Add the custom print function to the local environment
        env["print"] = print_capture

        # Set the signal handler for timeouts
        if self.timeout:
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.alarm(self.timeout)  # Set the timeout

        try:
            final_output = execute(code, env, env)
            signal.alarm(0)  # Cancel the alarm if execution completes successfully
        except TimeoutError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)
        finally:
            self.state.update(env)
            if self.timeout:
                signal.alarm(0)  # Ensure the alarm is canceled

        # Update the state with any new variables defined
        print_output = "\n".join(self.print_output) if self.print_output else None
        return final_output, print_output

    def reset(self):
        self.state = {}

import re

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer


class CodeActAgent:
    def __init__(self, llm_engine, max_iterations=10):
        self.llm_engine = llm_engine
        self.max_iterations = max_iterations
        self.repl = PythonREPL(timeout=5)

    def run(self, task: str):
        self.repl.reset()

        system_message = {"role": "system", "content": CODEACT_PROMPT}
        task_message = {"role": "user", "content": task}

        messages = [system_message, task_message]

        # Regular expressions to capture the content within tags
        thought_pattern = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
        code_pattern = re.compile(r"<code>(.*?)</code>", re.DOTALL)
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        logger.log(33, "======== New task ========")
        logger.log(34, task)

        final_answer = None

        for _ in range(self.max_iterations):
            response = self.llm_engine(messages, stop_sequences=["</code>", "</answer>"], start_sequence="<thought>\n")

            # Extract the content
            thoughts = thought_pattern.findall(response)
            codes = code_pattern.findall(response)
            answers = answer_pattern.findall(response)

            # If no action was taken, resample
            if len(codes) == 0 and len(answers) == 0:
                logger.error("Agent did not take any action.")
                logger.log(36, f"Raw LLM response: {response}")
                # Try to extract a Python function as a last resort
                func_match = re.search(r'(def [\s\S]+?\n)(?=\n|$)', response)
                if func_match:
                    final_answer = func_match.group(1).strip()
                    logger.log(33, "Fallback: Extracted function from raw response.")
                    logger.log(32, final_answer)
                    return final_answer
                return None

            if thoughts:
                logger.log(35, "=== Agent thoughts:")
                logger.log(31, thoughts[0].strip())

            if codes:
                code = codes[0].strip()
                code_highlight = highlight(
                    code,
                    PythonLexer(ensurenl=False),
                    Terminal256Formatter(style="nord"),
                )

                logger.log(35, ">>> Agent is executing the code below:")
                logger.log(31, code_highlight)
                logger.log(35, "====")

                final_output, print_output = self.repl.run(code)
                logger.log(35, "Print outputs:")

                total_output = ""

                if print_output:
                    logger.log(31, print_output)
                    total_output += print_output + "\n"
                if final_output is not None:
                    logger.log(31, final_output)
                    total_output += str(final_output) + "\n"

                logger.log(31, "")

                output = f"<output>\n{total_output}</output>"
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": output})

            # Prefer <answer> if present, else fallback to last <code> block
            if answers and answers[0].strip():
                final_answer = answers[0].strip()
            else:
                # As a last fallback, extract a Python function from the raw response
                func_match = re.search(r'(def [\s\S]+?\n)(?=\n|$)', response)
                if func_match:
                    final_answer = func_match.group(1).strip()
                else:
                    final_answer = None
            if final_answer:
                break

        else:
            logger.error("Reached max iterations.")
            return None

        logger.log(33, "Final answer:")
        logger.log(32, final_answer)


        return final_answer

agent = CodeActAgent(
    llm_engine=llm_engine,
    max_iterations=4,
)


# === New logic: process dev.csv and output submission.json (id, response) ===
import json, os, re, zipfile

set_seed(42)

df = pd.read_csv("dev.csv")  # expects columns: id, instruction
assert {"id", "instruction"}.issubset(df.columns), "CSV must have columns: id, instruction"

results = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    question = str(row["instruction"])
    response = agent.run(question)
    # If agent.run returns None, blank the response
    if not isinstance(response, str):
        response = ""
    # if response is None:
    #     response = ""
    results.append({"id": int(row["id"]), "response": str(response)})


# Save as JSON list
with open("submission.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"✅ Wrote submission.json with {len(results)} rows (id, response).")

# --- Validation and zipping ( from prompt.py) ---
SUB_PATH = "submission.json"
def file_format_check(path: str) -> bool:
    if os.path.basename(path) != "submission.json":
        print("Error: File name must be exactly 'submission.json'")
        return False
    if not path.lower().endswith(".json"):
        print("Error: File must have .json extension")
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        print("Note: The file must be in proper JSON format (not JSONL)")
        return False
    if not isinstance(data, list):
        print("Error: The root element should be a list of objects")
        return False
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Error: Item at index {idx} is not a dictionary")
            return False
        keys = set(item.keys())
        if keys != {"id", "response"}:
            print(f"Error: Item at index {idx} must contain only keys 'id' and 'response', found: {keys}")
            return False
        if not isinstance(item["id"], int):
            print(f"Error: 'id' field at index {idx} must be an integer")
            return False
        if not isinstance(item["response"], str):
            print(f"Error: 'response' field at index {idx} must be a string")
            return False
    print("Format check passed successfully!")
    return True

# Fencing/format validation and blanking invalids
with open(SUB_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)





# # fence_pat = re.compile(r"^```python[\s\S]*```$", re.MULTILINE)
# fence_pat = re.compile(r"^```python\s[\s\S]*?```$", re.MULTILINE)
# valid_format = []
# valid_fence  = []
# valid_both   = []
# def item_format_ok(item):
#     return (
#         isinstance(item, dict)
#         and set(item.keys()) == {"id", "response"}
#         and isinstance(item["id"], int)
#         and isinstance(item["response"], str)
#     )
# for item in data:
#     vfmt = item_format_ok(item)
#     vf   = bool(fence_pat.match(item["response"])) if vfmt else False
#     valid_format.append(vfmt)
#     valid_fence.append(vf)
#     valid_both.append(vfmt and vf)
# nf = sum(valid_fence)
# nm = sum(valid_format)
# nb = sum(valid_both)
# den = max(len(data), 1)
# print(f"Fencing valid: {nf}/{len(data)} ({nf*100.0/den:.1f}%)")
# print(f"Format valid:  {nm}/{len(data)} ({nm*100.0/den:.1f}%)")
# print(f"Both valid:    {nb}/{len(data)} ({nb*100.0/den:.1f}%)")
# for i, ok in enumerate(valid_both):
#     if not ok and isinstance(data[i], dict) and "response" in data[i]:
#         data[i]["response"] = ""





# using encoding utf-8

with open(SUB_PATH, "w", encoding="utf-8") as f:
    json.dump(
        [{"id": item["id"], "response": item["response"]} for item in data],
        f, ensure_ascii=False, indent=2
    )

# using manual method for building json

# output = '[\n'
# for i, item in enumerate(results):
#     output += f'  {{ "id": {item["id"]}, "response": "{item["response"].replace("\"", "\\\"")}" }}'
#     if i != len(results) - 1:
#         output += ',\n'
#     else:
#         output += '\n'
# output += ']'


# with open("submission.json", "w", encoding="utf-8") as f:
#     f.write(output)




print("✅ Updated submission.json after checks (invalid responses blanked).")
_ = file_format_check(SUB_PATH)
with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(SUB_PATH)
print("📦 Created submission.zip containing submission.json.")