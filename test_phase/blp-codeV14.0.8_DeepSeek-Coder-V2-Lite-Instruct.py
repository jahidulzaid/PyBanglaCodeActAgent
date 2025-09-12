# couldn't run on GPU, needed more memory


import re
from collections import Counter
import vllm
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import set_seed
import json
import os

from vllm import LLM
# ---------- CONFIG ----------
NUM_PATHS = 10   # number of samples per problem
MODEL_NAME = "nm-testing/DeepSeek-Coder-V2-Lite-Instruct-FP8"

llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    max_model_len=32768,   # try 16k; should be safer than putting full 32‑128k
    enable_prefix_caching=True,
    tensor_parallel_size=torch.cuda.device_count(),  # likely =1
    dtype="float16",   # vLLM may still need a higher precision dtype for non‑quantized parts
)
tokenizer = llm.get_tokenizer()

def llm_engine(messages, stop_sequences=None, start_sequence=None) -> str:
    sampling_params = vllm.SamplingParams(
        temperature=0.6,
        top_p=0.9,
        # use_beam_search=True,
        # num_beams=3,
        best_of=1,
        max_tokens=32768,
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

# ---------- CONFIG ----------
NUM_PATHS = 10   # number of samples per problem
MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-FP8"




CODEACT_PROMPT = """
You are a helpful coding assistant assigned to write OOP program in Python.  

For each row in the dataset, you will be given:  
- An **instruction** describing the task.  
- A **test_list** (Python assertions).  

**Your Workflow for each task:**

1. **Thought Process**:  
   Explain your reasoning.  
   - Wrap your explanation in `<thought>` tags.  
   - Consider edge cases (e.g., empty inputs, zero values, large inputs).  
   Example:  
   <thought>I need to compute gcd.</thought>  

2. **Write Python Code**:  
   Implement the Python program according to the instruction.  
   - You must use the exact names provided for **class**, **function**, and **variables** in the task. Do **not** rename them.  
   - Include both the solution and the provided test assertions.  
   - Wrap the complete code in `<code>` tags.  
   Example:  
   <code>
   # Your implementation here

   # Provided test_list (assertions)
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
   # Final clean solution
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
                # logger.log(36, f"Raw LLM response: {response}")
                # Try to extract a Python function as a last resort

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

            final_answer = answers[0].strip() if answers else None

            if final_answer is not None:
                break

        else:
            logger.error("Reached max iterations.")
            return None

        logger.log(33, "Final answer:")
        logger.log(32, final_answer)


        return final_answer

agent = CodeActAgent(
    llm_engine=llm_engine,
    max_iterations=8,
)


# ---------- run_codeact with majority voting ----------
def run_codeact(agent, instruction: str, test_list: list[str]) -> str:
    """Generate multiple candidates, test them, and pick by majority voting."""
    candidates = []

    for _ in range(NUM_PATHS):
        # Step 1: Generate code
        code = agent.run(instruction)
        if not code:
            continue

        # Normalize candidate for voting
        normalized_code = re.sub(r"\s+", " ", code.strip())

        # Step 2: Run candidate with all visible tests
        passed_all = True
        for test_code in test_list:
            final_output, _ = agent.repl.run(code + "\n" + test_code)
            if final_output is None:
                passed_all = False
                break

        if passed_all:
            candidates.append(normalized_code)

    # Step 3: Majority voting
    if candidates:
        final = Counter(candidates).most_common(1)[0][0]
    else:
        # Fallback: generate one candidate (ignore tests)
        code = agent.run(instruction)
        final = code.strip() if code else ""

    return final



# ---------- Main ----------
def main():
    SUB_PATH = "submission.json"

    if os.path.exists(SUB_PATH):
        with open(SUB_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    # Load test dataset
    # test_df = pd.read_csv("test_v1.csv")
    # existing_ids = {item["id"] for item in existing}
    # test_df = test_df[~test_df["id"].isin(existing_ids)]

    test_df = pd.read_csv("test_v1.csv")  # expects columns: id, instruction, test_list
    assert {"id", "instruction"}.issubset(test_df.columns), "CSV must have columns: id, instruction, test_list"



    # Process problems
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        instruction = row["instruction"]
        test_list = eval(row["test_list"]) if isinstance(row["test_list"], str) else []

        final_code = run_codeact(agent, instruction, test_list)
        results.append({"id": row["id"], "response": final_code})

        # Save incrementally
        with open(SUB_PATH, "w", encoding="utf-8") as f:
            json.dump(existing + results, f, ensure_ascii=False, indent=2)

    # Final cleaning
    with open(SUB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    for item in data:
        try:
            idx = int(item["id"])
            resp = str(item["response"])
            cleaned.append({"id": idx, "response": resp})
        except Exception:
            continue

    with open(SUB_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished. Saved {len(cleaned)} results to {SUB_PATH}")

# ---------- Entry ----------
if __name__ == "__main__":
    main()