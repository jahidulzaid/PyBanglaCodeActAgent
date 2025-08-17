
# %%
from unsloth import FastModel
import torch
from transformers import AutoConfig # Import AutoConfig

# Choose any model of your choosing, based on your GPU RAM
# default = "md-nishat-008/TigerLLM-1B-it"
# gemma3_1b = "google/gemma-3-1b-it"
# gemma3_4b = "google/gemma-3-4b-it"
gemma3_12b = "google/gemma-3-12b-it"
# gemma3_27b = "google/gemma-3-27b-it"

# Load model with the corrected config
model, tokenizer = FastModel.from_pretrained(
    model_name = gemma3_12b,
    max_seq_length = 1024,
    load_in_4bit = False,
    load_in_8bit = True,
    full_finetuning = False,
    # token = "hf_...", # In case you need it
)


# Note - You will find your Hugging Face Access Token in your Hugging Face Account.
# This will be necessary for Gemma & LLaMA models.

# %% [markdown]
# ## Step 2 - Play with the Model

# %%
from unsloth.chat_templates import get_chat_template

# Attach the TigerLLM & Gemma-3 chat template to the tokenizer
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# Compose the conversation
bangla_q = (
    "মৌলিক সংখ্যা বের করার জন্য একটি পাইথন ফাংশন লিখুন"
)


messages = [
    {
        "role": "user",
        "content": bangla_q,
    }
]

# Render the chat template to plain text
chat_text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,  # must be True for generation
    tokenize=False               # return a plain string
)

# Tokenize and generate
inputs = tokenizer(chat_text, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    use_cache=False,
    max_new_tokens=256,           # increase for longer outputs
    temperature=0.3,
    top_p=0.95,
    top_k=64,
)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

# %% [markdown]
# ## Step 3 - Try the Dev Set

# %%
import pandas as pd
import torch
from tqdm.auto import tqdm
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

# Load prompts
df = pd.read_csv("dev.csv")  # expects columns: id, instruction
assert {"id", "instruction"}.issubset(df.columns), "CSV must have columns: id, instruction"

responses = []
for prompt in tqdm(df["instruction"], desc="Generating"):
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt)}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            use_cache=False,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.95,
            top_k=64,
        )

    gen_ids  = out[0][inputs["input_ids"].shape[1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    responses.append(gen_text)

# Save ONLY id + response as JSON
out_df = pd.DataFrame({"id": df["id"], "response": responses})
out_df.to_json("submission.json", orient="records", force_ascii=False, indent=2)
print(f"✅ Wrote submission.json with {len(out_df)} rows (id, response).")


# %% [markdown]
# ## Step 4 - Preparing Submission File

# %%
import json, os, re, zipfile

SUB_PATH = "submission.json"

def file_format_check(path: str) -> bool:
    # name + extension
    if os.path.basename(path) != "submission.json":
        print("Error: File name must be exactly 'submission.json'")
        return False
    if not path.lower().endswith(".json"):
        print("Error: File must have .json extension")
        return False

    # must be valid JSON (not JSONL) and root must be a list
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

    # each item: dict with ONLY keys {'id','response'}; id=int; response=str
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

# ---------- Load, compute per-item validity, blank invalids, save, zip ----------
# Load JSON list
with open(SUB_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

n = len(data)
fence_pat = re.compile(r"^```python[\s\S]*```$", re.MULTILINE)

valid_format = []
valid_fence  = []
valid_both   = []

# Per-item validation mirrors file checker semantics
def item_format_ok(item):
    return (
        isinstance(item, dict)
        and set(item.keys()) == {"id", "response"}
        and isinstance(item["id"], int)
        and isinstance(item["response"], str)
    )

for item in data:
    vfmt = item_format_ok(item)
    vf   = bool(fence_pat.match(item["response"])) if vfmt else False
    valid_format.append(vfmt)
    valid_fence.append(vf)
    valid_both.append(vfmt and vf)

# Report stats
nf = sum(valid_fence)
nm = sum(valid_format)
nb = sum(valid_both)
den = max(n, 1)
print(f"Fencing valid: {nf}/{n} ({nf*100.0/den:.1f}%)")
print(f"Format valid:  {nm}/{n} ({nm*100.0/den:.1f}%)")
print(f"Both valid:    {nb}/{n} ({nb*100.0/den:.1f}%)")

# Strict policy: blank responses that fail ANY check
for i, ok in enumerate(valid_both):
    if not ok and isinstance(data[i], dict) and "response" in data[i]:
        data[i]["response"] = ""

# Overwrite submission.json (id+response only)
with open(SUB_PATH, "w", encoding="utf-8") as f:
    json.dump(
        [{"id": item["id"], "response": item["response"]} for item in data],
        f, ensure_ascii=False, indent=2
    )
print("✅ Updated submission.json after checks (invalid responses blanked).")

# Final file-level check (should pass)
_ = file_format_check(SUB_PATH)

# Zip as submission.zip (Jupyter-friendly, no shell commands)
with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(SUB_PATH)
print("📦 Created submission.zip containing submission.json.")


# %% [markdown]
# # Submit the submission.zip file in CodaBench


