import json, ast, re, difflib, types
import pandas as pd
from pathlib import Path

# ---- paths (adjust if needed) ----
CSV_PATH = Path("test_v1.csv")
SUB_PATH = Path("submission.json")

# ---- helpers (no intermediate prints/logs) ----
def strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if s.startswith("```"):
        body = s[3:]
        if "\n" in body:
            body = body.split("\n", 1)[1]
        if "```" in body:
            body = body.rsplit("```", 1)[0]
        return body.strip()
    return s

def parse_tests(raw) -> list:
    raw = str(raw)
    x = ast.literal_eval(raw)
    if isinstance(x, str):
        x = ast.literal_eval(x)
    if not isinstance(x, (list, tuple)):
        raise ValueError("test_list parsed to non-list")
    return [str(t) for t in x]

FN_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
def expected_function_names(assert_list):
    from collections import Counter
    names = []
    for a in assert_list:
        for m in FN_RE.finditer(a):
            names.append(m.group(1))
    return [n for n, _ in Counter(names).most_common()]

def ensure_function(ns: dict, expected: str):
    if expected in ns and isinstance(ns[expected], types.FunctionType):
        return expected, False
    fns = [k for k, v in ns.items() if isinstance(v, types.FunctionType)]
    if not fns:
        return None, False
    def norm(s): return re.sub(r"_+", "", s).lower()
    target = norm(expected)
    for cand in sorted(fns, key=lambda x: (norm(x) != target, x)):
        if norm(cand) == target:
            ns[expected] = ns[cand]
            return cand, True
    close = difflib.get_close_matches(expected, fns, n=1, cutoff=0.82)
    if close:
        cand = close[0]
        ns[expected] = ns[cand]
        return cand, True
    if len(fns) == 1:
        cand = fns[0]
        ns[expected] = ns[cand]
        return cand, True
    return None, False

# ---- load data ----
df = pd.read_csv(CSV_PATH)
with open(SUB_PATH, "r", encoding="utf-8") as f:
    sub = json.load(f)

# id -> response
if isinstance(sub, list):
    id_to_resp = {}
    for row in sub:
        if isinstance(row, dict):
            id_key = next((k for k in ("id","ID","sample_id","idx") if k in row), None)
            resp_key = next((k for k in ("response","output","code","generated_code","prediction") if k in row), None)
            if id_key is not None and resp_key is not None:
                id_to_resp[str(row[id_key])] = row[resp_key]
else:
    id_to_resp = {str(k): v for k, v in sub.items()}

# ---- run tests (silent) ----
counts = {
    "PASS": 0,
    "FAIL_ASSERT": 0,
    "RUNTIME_ERROR": 0,
    "COMPILE_ERROR": 0,
    "MISSING_CODE": 0,
    "PARSE_FAIL": 0,
}
total = len(df)

for _, r in df.iterrows():
    rid = str(r.get("id", ""))
    try:
        tests = parse_tests(r["test_list"])
    except Exception:
        counts["PARSE_FAIL"] += 1
        continue

    resp = id_to_resp.get(rid)
    if resp is None:
        counts["MISSING_CODE"] += 1
        continue

    code = strip_code_fences(resp)
    ns = {}
    try:
        exec(code, ns, ns)
    except Exception:
        counts["COMPILE_ERROR"] += 1
        continue

    names = expected_function_names(tests)
    expected = names[0] if names else None
    if expected:
        ensure_function(ns, expected)  # silent alias if possible

    ok = True
    for a in tests:
        try:
            exec(a, ns, ns)
        except AssertionError:
            counts["FAIL_ASSERT"] += 1
            ok = False
            break
        except Exception:
            counts["RUNTIME_ERROR"] += 1
            ok = False
            break

    if ok:
        counts["PASS"] += 1

# ---- single fancy print with details ----
passes = counts["PASS"]
pct = (passes / total) * 100 if total else 0.0
fails = total - passes
bar_len = 28
filled = int(round(bar_len * pct / 100))
bar = "█" * filled + "░" * (bar_len - filled)

# ANSI styles
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

print(
    f"\n{BOLD}Pass@1 Summary{RESET}\n"
    f"{BOLD}{pct:6.2f}%{RESET}  {bar}  {BOLD}{passes}{RESET}/{total} passed\n"
    f"{DIM}Failures: {fails} = "
    f"Assertion {counts['FAIL_ASSERT']} | "
    f"Runtime {counts['RUNTIME_ERROR']} | "
    f"Compile {counts['COMPILE_ERROR']} | "
    f"Missing {counts['MISSING_CODE']} | "
    f"Parse {counts['PARSE_FAIL']}{RESET}\n"
)
