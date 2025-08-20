

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# pip install -U transformers==4.45.2
# pip install -U torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# pip uninstall -y pynvml
# pip install nvidia-ml-py
# pip install vllm

import re
from collections import Counter

import vllm
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import set_seed

model_id = "Qwen/Qwen2.5-32B-Instruct-AWQ"

llm = vllm.LLM(
    model_id,
    quantization="awq",
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

CODEACT_PROMPT = """You are a helpful assistant assigned to solve mathematical and coding tasks.
To achieve this, you'll use an interactive coding environment and work through each problem in structured steps: 'Thought:', 'Code:', and 'Observation:' sequences.

**Instructions for each turn:**
1. **Thought Process**: Start by explaining your step-by-step reasoning for solving the task.
   - Enclose this in `<thought>` tags. For example: `<thought>I need to print "Hello World!"</thought>`.

2. **Action Options**:
   - **Option 1**: Execute code in a Python environment to obtain output.
     - Enclose your code within `<code>` tags. For example: `<code>print("Hello World!")</code>`.
   - **Option 2**: Provide a final answer directly if calculations are complete.
     - Enclose your answer in `<answer>` tags. For example: `<answer>3.1415</answer>`.

---

**Example Tasks and Responses:**

Task: Convert the point \((0, -3 \sqrt{3}, 3)\) from rectangular to spherical coordinates, in the form \((\\rho, \\theta, \phi)\) where \(\\rho > 0\), \(0 \leq \\theta < 2\pi\), and \(0 \leq \phi \leq \pi\).

<thought>
To convert \((x, y, z)\) from rectangular to spherical coordinates \((\\rho, \\theta, \phi)\), use these formulas:
1. \(\\rho = \sqrt{x^2 + y^2 + z^2}\)
2. \(\\theta = \\arctan\\frac{y}{x}\)
3. \(\phi = \\arccos\\frac{z}{\\rho}\)

I'll implement these calculations in code.
</thought>
<code>
from sympy import sqrt, atan2, acos, pi

def rectangular_to_spherical():
    x, y, z = 0, -3*sqrt(3), 3
    rho = sqrt(x**2 + y**2 + z**2)
    theta = atan2(y, x)
    phi = acos(z/rho)
    return rho, theta, phi

spherical_coordinates = rectangular_to_spherical()
print(spherical_coordinates)
</code><end_action/>
<output>
(6, -pi/2, pi/3)
</output>
<thought>
To fit the required range for \(\\theta\), add \(2\pi\) to adjust \(\\theta = -\pi/2\). The spherical coordinates are \((6, \\frac{3\pi}{2}, \\frac{\pi}{3})\).
</thought>
<answer>
(6, \\frac{3\pi}{2}, \\frac{\pi}{3})
</answer><end_action/>

---

Task: Calculate \(1011_2 + 101_2 - 1100_2 + 1101_2\) in binary.

<thought>
I'll define a function to handle binary operations by converting each value to decimal, performing the addition and subtraction, and converting the result back to binary.
</thought>
<code>
def binary_sum_diff():
    num1 = int("1011", 2)
    num2 = int("101", 2)
    num3 = int("1100", 2)
    num4 = int("1101", 2)

    result = num1 + num2 - num3 + num4
    result_binary = format(result, "b")
    return result_binary

result = binary_sum_diff()
print(result)
</code><end_action/>
<output>
10001
</output>
<thought>
The answer in base 2 is \(10001_2\).
</thought>
<answer>
\(10001_2\)
</answer><end_action/>
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
    def __init__(self, llm_engine, max_iterations=4):
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
                return None

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
            return None

        logger.log(33, "Final answer:")
        logger.log(32, final_answer)

        return final_answer

agent = CodeActAgent(
    llm_engine=llm_engine,
    max_iterations=4,
)

test_df = pd.read_csv("test.csv", index_col="ID")
test_df.head()

set_seed(42)

data = []

for id, row in tqdm(test_df.iterrows(), total=len(test_df)):
    question = row["Problem"]
    question += "\nEnsure that the final answer is an integer without any units."

    answer = agent.run(question)

    try:
        answer = int(answer)
    except (TypeError, ValueError):
        answer = 0

    data.append({
        "ID": id,
        "Answer": answer
    })

submit = pd.DataFrame.from_records(data)
submit.to_csv("submission.csv", index=False)