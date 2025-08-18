

# %% [markdown]
# ## Step 1 - Choose a Base Model

# %%
from unsloth import FastModel
import torch
from transformers import AutoConfig # Import AutoConfig

# Choose any model of your choosing, based on your GPU RAM
default = "md-nishat-008/TigerLLM-1B-it"
# gemma3_1b = "google/gemma-3-1b-it"
# gemma3_4b = "google/gemma-3-4b-it"
# gemma3_12b = "google/gemma-3-12b-it"
# gemma3_27b = "google/gemma-3-27b-it"

# Load model with the corrected config
model, tokenizer = FastModel.from_pretrained(
    model_name = default,
    max_seq_length = 1024,
    load_in_4bit = False,
    load_in_8bit = True,
    full_finetuning = False,
    # token = "hf_...", # In case you need it
)


# Note - You will find your Hugging Face Access Token in your Hugging Face Account.
# This will be necessary for Gemma & LLaMA models.

# %% [markdown]
# ## Step 2 - Play with the Base Model

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
# ## Step 3 - Set the LoRA Adapters

# %%
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 32,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

# %% [markdown]
# ## Step 4 - Data Prep

# %% [markdown]
# ### Chat Template

# %%
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

# %% [markdown]
# ### Use Your Own Instruction Dataset
# ### You will find a sample dataset here - https://noshinulfat.github.io/blp25_code_generation_task/#/task-announcement

# %%
## We will use a very small sample dataset
from datasets import load_dataset

# Load the local CSV (will appear as the single default “train” split)
dataset = load_dataset(
    "csv",
    data_files="trial.csv",   # path relative to your current directory
)["train"]                               # pull the one split that load_dataset creates

print(len(dataset), "rows loaded.")


# %% [markdown]
# ### Map

# %%
def to_finetome(example):
    return {
        "conversations": [
            {"role": "user",  "content": example["instruction"]},
            {"role": "model", "content": example["response"]},   # ← key change
        ]
    }

dataset = dataset.map(
    to_finetome,
    num_proc=4,
    remove_columns=["instruction", "response"]
)

# %% [markdown]
# ### Check the First Instance

# %%
dataset[0]

# %% [markdown]
# ### Formatting

# %%
def formatting_prompts_func(batch):
    texts = []
    for convo in batch["conversations"]:
        try:
            serialized = tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False,
            ).removeprefix("<bos>")
        except Exception:          # catches TemplateError, etc.
            serialized = ""        # placeholder keeps list length in sync
        texts.append(serialized)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)


# %% [markdown]
# ### Check the First Instance Again

# %%
dataset[0]["text"]

# %% [markdown]
# ## Step 5 - Finetuning Config
# ### Adjust as you see fit

# %%
from trl import SFTTrainer, SFTConfig
import math

# --- constants ----------------------------------------------------------
DATASET_SIZE                = 74
PER_DEV_BATCH               = 16
GRAD_ACC_STEPS              = 4
EPOCHS                      = 2

# derived values (no longer passed as kwargs)
EFFECTIVE_BATCH             = PER_DEV_BATCH * GRAD_ACC_STEPS
STEPS_PER_EPOCH             = math.ceil(DATASET_SIZE / EFFECTIVE_BATCH)
MAX_STEPS                   = EPOCHS * STEPS_PER_EPOCH

cfg = SFTConfig(
    dataset_text_field          = "text",
    # ── memory / speed knobs ────────────────────────────────────────────
    packing                     = False,
    per_device_train_batch_size = PER_DEV_BATCH,
    gradient_accumulation_steps = GRAD_ACC_STEPS,
    gradient_checkpointing      = True,
    bf16                        = True,
    optim                       = "adamw_8bit",

    # ── schedule / optimisation ────────────────────────────────────────
    num_train_epochs            = EPOCHS,       # renamed
    max_steps                   = MAX_STEPS,    # keeps the same target
    warmup_steps                = 10,
    lr_scheduler_type           = "cosine",
    learning_rate               = 1e-4,
    weight_decay                = 0.01,

    # ── misc ───────────────────────────────────────────────────────────
    logging_steps               = 1,  ## Update this
    dataset_num_proc            = 8,
    seed                        = 3407,
    report_to                   = "none",
)

trainer = SFTTrainer(
    model          = model,        # load with attn_implementation="flash_attention_2"
    tokenizer      = tokenizer,
    train_dataset  = dataset,
    args           = cfg,
)


# %% [markdown]
# ### Masking

# %%
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# %% [markdown]
# ### Verify the Masking

# %%
tokenizer.decode(trainer.train_dataset[0]["input_ids"])

# %% [markdown]
# ### Now let's print the masked out example - you should see only the answer is present:

# %%
tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " ")

# %% [markdown]
# ## Step 6 - Let's Finetune 🔥🔥

# %%
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# %% [markdown]
# ### Start Finetuning....... 🔥🔥

# %%
trainer_stats = trainer.train()

# %% [markdown]
# ## Step 7 - Let's Use Our New Finetuned Model

# %%
from unsloth.chat_templates import get_chat_template

# Attach the Gemma-3 chat template to the tokenizer
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
    max_new_tokens=1012,           # increase for longer outputs
    temperature=0.3,
    top_p=0.95,
    top_k=64,
)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])


# %% [markdown]
# ## Step 8 - Use New Model on Dev Set

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
# ## Step 9 - Preparing Submission File

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

# %% [markdown]
# ### Save the NEW model...if it's good :)

# %%
model.save_pretrained("New_Model")  # Local saving
tokenizer.save_pretrained("New_Model")


