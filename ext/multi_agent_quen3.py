
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# ! pip install -U transformers==4.45.2
# ! pip install -U torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# ! pip uninstall -y pynvml
# ! pip install nvidia-ml-py
# ! pip install vllm

import re
import csv
from collections import Counter

import vllm
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import set_seed

model_id = "Qwen/Qwen3-8B"

llm = vllm.LLM(
    model_id,
    max_model_len=4096,
    enable_prefix_caching=True,
    tensor_parallel_size=torch.cuda.device_count(),
)

tokenizer = llm.get_tokenizer()

TEMP = 0.7

def llm_engine(list_of_messages, stop_sequences=None, start_sequence=None) -> list[str]:
    sampling_params = vllm.SamplingParams(
        temperature=TEMP,
        top_p=0.9,
        max_tokens=2048,
        stop=stop_sequences,
        include_stop_str_in_output=True,
    )

    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in list_of_messages
    ]
    if start_sequence:
        prompts = [prompt + start_sequence for prompt in prompts]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    responses = [o.outputs[0].text for o in outputs]

    if start_sequence:
        responses = [start_sequence + response for response in responses]
    return responses

CODEACT_PROMPT = """
You are a helpful coding assistant assigned to solve algorithmic problems in Python.

For each row in the dataset, you will be given:
- An **instruction** describing the task.
- A **test_list** (Python assertions).

**Your Workflow for each task:**

1. **Thought Process**:
   Explain your reasoning step by step before coding.
   - Wrap your explanation in `<thought>` tags.
   - Consider edge cases (e.g., empty inputs, zero values, large inputs) in your reasoning.
   Example:
   <thought>I need to compute the smallest number divisible by all numbers from 1 to n. I can use LCM iteratively.</thought>

2. **Write Python Code**:
   Implement the Python program according to the instruction.
   - You must use the exact names provided for **class**, **function**, and **variables** in the task. Do **not** rename them.
   - Include both the solution and the provided test assertions.
   - Wrap the complete code in `<code>` tags.
   Example:
   <code>
    def smallest_multiple(n):
        if n <= 2:
            return n
        i = n * 2
        factors = [number for number in range(n, 1, -1) if number * 2 > n]
        while True:
            for a in factors:
                if i % a != 0:
                    i += n
                    break
                if a == factors[-1] and i % a == 0:
                    return i

    # Run tests
    assert smallest_multiple(13) == 360360
    assert smallest_multiple(2) == 2
    assert smallest_multiple(1) == 1
    </code>

3. **Observation**:
   After executing, confirm whether all tests passed or if debugging is needed.
   - Wrap your observation in `<observation>` tags.
   Example:
   <observation>All tests passed successfully.</observation>

4. **Final Answer**:
   Provide only the final clean Python program (without test assertions).
   - Wrap your final answer in `<answer>` tags.
   Example:
    <answer>
    def smallest_multiple(n):
        if n <= 2:
            return n
        i = n * 2
        factors = [number for number in range(n, 1, -1) if number * 2 > n]
        while True:
            for a in factors:
                if i % a != 0:
                    i += n
                    break
                if a == factors[-1] and i % a == 0:
                    return i
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
    def __init__(self, timeout=5, additional_vars=None):
        self.state = {}
        self.print_output = []
        self.additional_vars = additional_vars
        self.timeout = timeout  # Set a default timeout value
        self.reset()

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
        if self.additional_vars is not None:
            self.state.update(self.additional_vars)

import re

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer


class AsyncCodeActAgent:
    def __init__(self, max_iterations=4):
        self.max_iterations = max_iterations
        self.repl = PythonREPL(timeout=5)
        self.running = False

    def runner(self, task: str):
        self.repl.reset()
        self.running = True

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
            response = yield messages

            # Extract the content
            thoughts = thought_pattern.findall(response)
            codes = code_pattern.findall(response)
            answers = answer_pattern.findall(response)

            # If no action was taken, resample
            if len(codes) == 0 and len(answers) == 0:
                continue

            if thoughts:
                logger.log(35, "=== Agent thoughts:")
                logger.log(31, thoughts[0].strip())

            if codes:
                code = codes[0].strip()
                code = highlight(
                    code,
                    PythonLexer(ensurenl=False),
                    Terminal256Formatter(style="nord"),
                )

                logger.log(35, ">>> Agent is executing the code below:")
                logger.log(31, code)
                logger.log(35, "====")

                final_output, print_output = self.repl.run(codes[0].strip())
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

            final_answer = answers[0].strip() if answers else None

            if final_answer is not None:
                break

        else:
            logger.error("Reached max iterations.")
            self.running = False
            return None

        logger.log(33, "Final answer:")
        logger.log(32, final_answer)

        self.running = False
        return final_answer

def sc_codeact(question: str, num_agents: int, max_iterations: int):
    stop_sequences = ["</code>", "</answer>"]
    start_sequence = "<thought>\n"
    agents = [AsyncCodeActAgent(max_iterations=max_iterations) for _ in range(num_agents)]

    states = [(agent, agent.runner(question)) for agent in agents]

    list_of_messages = []

    # Init
    for agent, runner in states:
        list_of_messages.append(runner.send(None))

    responses = llm_engine(list_of_messages, stop_sequences=stop_sequences, start_sequence=start_sequence)

    answers = []

    while states:
        list_of_messages = []

        for (agent, runner), response in zip(states, responses):
            try:
                list_of_messages.append(runner.send(response))
            except StopIteration as e:
                answers.append(e.value)

        states = [(agent, runner) for agent, runner in states if agent.running]
        responses = llm_engine(list_of_messages, stop_sequences=stop_sequences, start_sequence=start_sequence)

    return answers


def to_integer(data, epsilon=1e-2):
    # Check if the data is None or cannot be converted to a float
    try:
        num = float(data)
        if abs(num) == float('inf'):
            return None
    except (ValueError, TypeError):
        return None

    # Check if the number is close enough to an integer within the given epsilon
    rounded_num = round(num)
    if abs(num - rounded_num) <= epsilon:
        return int(rounded_num)

    return None

from collections import Counter

def majority_vote(answers, epsilon=1e-2):
    # Convert all elements in the list to integers using to_integer
    int_answers = [to_integer(ans, epsilon) for ans in answers]

    # Filter out None and negative values
    int_answers = [ans for ans in int_answers if ans is not None and ans >= 0]

    # Count occurrences of each unique integer
    counts = Counter(int_answers)

    # Get the two most common integers
    most_common = counts.most_common(2)

    # Check for a tie
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return None, True  # Tie detected

    # Return the majority integer, or None if the list is empty
    return (most_common[0][0], False) if most_common else (None, False)

def analysis_conditions(question):
    """Determine the conditions and objectives of a question."""
    sampling_params = vllm.SamplingParams(
        temperature=0,
        # use_beam_search=True,
        best_of=3,
        max_tokens=2048,
    )

    prompt = f"{question}\nEnsure that the final answer is a python program (may have class structure)."
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text], sampling_params, use_tqdm=False)
    response = output[0].outputs[0].text

    parts = response.split("Objective:")
    conditions_text = parts[0].replace("Conditions:", "").strip()
    conditions = re.findall(r'\d\.\s*(.*)', conditions_text)
    conditions = [condition.strip() for condition in conditions]

    objectives_text = parts[1].strip()
    if re.search(r'\d\.\s+', objectives_text):
        # Extract objectives with numbers
        objectives = re.findall(r'\d\.\s*(.*)', objectives_text)
    else:
        # Split objectives by newline for unnumbered items
        objectives = objectives_text.split('\n')
    objectives = [objective.strip() for objective in objectives]

    return response, conditions, objectives



def sc_codeact_with_thinker(question: str, num_agents: int, max_iterations: int):
    stop_sequences = ["</code>", "</answer>"]
    start_sequence = "<thought>\n"
    agents = [AsyncCodeActAgent(max_iterations=max_iterations) for _ in range(num_agents)]

    sampling_params = vllm.SamplingParams(
        n=num_agents,
        temperature=TEMP,
        top_p=0.9,
        max_tokens=2048,
    )

    sampling_params_beam = vllm.SamplingParams(
        temperature=0,
        # use_beam_search=True,
        best_of=1,
        max_tokens=2048,
        stop=["</python>"],
        include_stop_str_in_output=True,
    )

    prompt = f"{question}\nEnsure that the final answer is a python program (may have class structure)."
    messages = [{"role": "user", "content": prompt}]

    repl = PythonREPL(timeout=5, additional_vars={"words_to_number_bangla": words_to_number_bangla})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text], sampling_params_beam, use_tqdm=False)
    response = output[0].outputs[0].text

    match = re.search(r"<python>(.*?)</python>", response, re.DOTALL)

    if match:
        code = match.group(1).strip()

        final_output, print_output = repl.run(code)
        output = "<output>\n"

        if print_output:
            output += print_output + "\n"

        if final_output is not None:
            output += str(final_output) + "\n"

        output += "</output>"

        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": output})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = llm.generate([text], sampling_params, use_tqdm=False)
    responses = [o.text for o in output[0].outputs]

    states = [(agent, agent.runner(question + "\n" + response)) for response, agent in zip(responses, agents)]

    list_of_messages = []

    # Init
    for agent, runner in states:
        list_of_messages.append(runner.send(None))

    responses = llm_engine(list_of_messages, stop_sequences=stop_sequences, start_sequence=start_sequence)

    answers = []

    while states:
        list_of_messages = []

        for (agent, runner), response in zip(states, responses):
            try:
                list_of_messages.append(runner.send(response))
            except StopIteration as e:
                answers.append(e.value)

        states = [(agent, runner) for agent, runner in states if agent.running]
        responses = llm_engine(list_of_messages, stop_sequences=stop_sequences, start_sequence=start_sequence)

    return answers

# Don't show logs
logger.setLevel(50)

test_df = pd.read_csv("dev.csv")
test_df.head()
test_df.iloc[1]

# # === New logic: process dev.csv and output submission.json (id, response) ===
# import json, os, re, zipfile

# set_seed(42)
# set_seed(42)
# num_agents = 16
# max_iterations = 4



# df = pd.read_csv("dev.csv")  # expects columns: id, instruction
# assert {"id", "instruction"}.issubset(df.columns), "CSV must have columns: id, instruction"

# results = []
# for i, row in tqdm(df.iterrows(), total=len(df)):
#     question = str(row["instruction"])

#     response = agent.run(question)
#     # response = run_with_self_consistency(agent, question, num_paths=5)


#     # If agent.run returns None, blank the response
#     if not isinstance(response, str):
#         response = ""
#     # if response is None:
#     #     response = ""
#     results.append({"id": int(row["id"]), "response": str(response)})





# # Save as JSON list
# with open("submission.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)
# print(f"✅ Wrote submission.json with {len(results)} rows (id, response).")

# # --- Validation and zipping ( from prompt.py) ---
# SUB_PATH = "submission.json"
# def file_format_check(path: str) -> bool:
#     if os.path.basename(path) != "submission.json":
#         print("Error: File name must be exactly 'submission.json'")
#         return False
#     if not path.lower().endswith(".json"):
#         print("Error: File must have .json extension")
#         return False
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#     except json.JSONDecodeError as e:
#         print(f"Error: Invalid JSON format - {e}")
#         print("Note: The file must be in proper JSON format (not JSONL)")
#         return False
#     if not isinstance(data, list):
#         print("Error: The root element should be a list of objects")
#         return False
#     for idx, item in enumerate(data):
#         if not isinstance(item, dict):
#             print(f"Error: Item at index {idx} is not a dictionary")
#             return False
#         keys = set(item.keys())
#         if keys != {"id", "response"}:
#             print(f"Error: Item at index {idx} must contain only keys 'id' and 'response', found: {keys}")
#             return False
#         if not isinstance(item["id"], int):
#             print(f"Error: 'id' field at index {idx} must be an integer")
#             return False
#         if not isinstance(item["response"], str):
#             print(f"Error: 'response' field at index {idx} must be a string")
#             return False
#     print("Format check passed successfully!")
#     return True

# # Fencing/format validation and blanking invalids
# with open(SUB_PATH, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # using encoding utf-8

# with open(SUB_PATH, "w", encoding="utf-8") as f:
#     json.dump(
#         [{"id": item["id"], "response": item["response"]} for item in data],
#         f, ensure_ascii=False, indent=2
#     )


# print("✅ Updated submission.json after checks (invalid responses blanked).")
# _ = file_format_check(SUB_PATH)
# with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
#     zf.write(SUB_PATH)
# print("📦 Created submission.zip containing submission.json.")

import json
from tqdm import tqdm

set_seed(42)
num_agents = 16
max_iterations = 4

SUB_PATH = "submission.json"

results = []  # <--- Collect answers here

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    id = row["id"]
    prompt = row["instruction"]
    prompt += "\nEnsure that the final answer is a python program(Might have Class)."

    print(f" INPUT FOR PROBLEM ID {id} ===\n{prompt}\n")

    answers = sc_codeact_with_thinker(prompt, num_agents, max_iterations)
    print(f" --- CANDIDATE ANSWERS ({len(answers)}) ===\n{answers}\n")

    answer, tie = majority_vote(answers)

    if tie:
        print("--- THERE'S A TIE ---")
        answers += sc_codeact_with_thinker(prompt, num_agents, max_iterations)
        print(f"=== NEW CANDIDATE ANSWERS ({len(answers)}) ===\n{answers}\n")
        answer, _ = majority_vote(answers)

    answer = answer if answer is not None else "0"


     # ✅ TEST CASE VALIDATION
    try:
        test_cases = ast.literal_eval(row["test_list"])  # safe parsing
    except Exception as e:
        print(f"⚠️ Could not parse test cases for ID {id}: {e}")
        test_cases = []

    all_passed = True
    if test_cases:
        repl = PythonREPL(timeout=5)  # you likely already have this somewhere

        # Run generated code
        repl.run(answer)

        # Run each test
        for test in test_cases:
            try:
                result, print_output = repl.run(test)
                print(f"✅ Test passed: {test}")
            except Exception as e:
                print(f"❌ Test failed: {test} | Error: {e}")
                all_passed = False

    print(f"--- MAJORITY ANSWER ---\n{answer}\n")
    print(f"--- TEST RESULT: {'PASSED' if all_passed else 'FAILED'} ---\n")

    results.append({
        "id": id,
        "response": answer,
        # "passed_all_tests": all_passed  # optional metadata
    })

    print(f"--- MAJORITY ANSWER ---\n{answer}\n")

    results.append({"id": id, "response": answer})  # <--- Store for JSON

# ✅ Write all results to a JSON file
with open(SUB_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ Updated submission.json.")
_ = file_format_check(SUB_PATH)
with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(SUB_PATH)
print("📦 Created submission.zip containing submission.json.")

