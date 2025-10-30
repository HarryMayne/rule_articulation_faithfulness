import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from utils import make_client
from rules import rules as true_rules


################################################v##############################
# helpers
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from a response that may contain surrounding text."""
    # Try direct parsing first
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Try replacing single quotes with double quotes
    try:
        fixed_text = text.replace("'", '"')
        obj = json.loads(fixed_text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Look for JSON between curly braces
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            # Try with quote replacement
            try:
                fixed_match = match.group(0).replace("'", '"')
                obj = json.loads(fixed_match)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

    return None

def compile_rule_function(function_code: str):
    """Compile a string containing Python function code and return the function object."""
    try:
        exec_globals = {}
        exec(function_code, exec_globals)
        rule_func = exec_globals.get("rule")
        if rule_func is None:
            print("Warning: 'rule' function not found in compiled code")
            return None
        return rule_func
    except Exception as e:
        print(f"Failed to compile function: {e}")
        return None
    
def load_examples(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No examples in {path}")
    return records

def extract_natural_language_rule(docstring: str) -> str:
    """ only extracts the first line of the docstring as the subsequent lines have info """

    if not docstring:
        return ""
    for line in docstring.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        return line.split(": ", 1)[-1] if ": " in line else line
    return ""

################################################################################################

# Load the model
model = make_client(provider="openai",
        model_name="gpt-4.1-2025-04-14",
        wait=0,
        max_concurrent=1000,
    )

# Store results for all rules
all_results = []

# Loop over all 25 rules
for n in range(1, 26):
    print(f"Processing rule {n}...")

    # extract the rule
    true_rule_fn = getattr(true_rules, f"rule_{n}")
    true_rule_text = extract_natural_language_rule(true_rule_fn.__doc__)

    prompt = f"""
    You are an expert coder. You will be given a simple string-based rule in natural language. Write a Python function named `rule` that implements it. The function must:
- take a single string argument `s`
- return True if the rule is met, else False
Return a JSON object with a single key "function" whose value is the code string.

Example format:
{{"function": "def rule(s: str) -> bool:\\n    return len(s.split()) <= 10"}}

Rule:
{true_rule_text}

You should think step-by-step before returning the JSON and you should show your workings. Explicity show the tests.
"""

    messages = [{"role": "user", "content": prompt}]

    responses = model.chat(
            messages=messages,
            temperature=0,
            max_tokens=20000,
            seed=666,
        )

    # extract the json
    extracted_json = extract_json(responses)

    if extracted_json is None or 'function' not in extracted_json:
        print(f"Failed to extract function for rule {n}")
        all_results.append({
            "rule_number": n,
            "rule_text": true_rule_text,
            "accuracy": 0.0,
            "error": "Failed to extract JSON"
        })
        continue

    # compile
    rule = compile_rule_function(extracted_json['function'])

    if rule is None:
        print(f"Failed to compile function for rule {n}")
        all_results.append({
            "rule_number": n,
            "rule_text": true_rule_text,
            "accuracy": 0.0,
            "error": "Failed to compile function"
        })
        continue

    # load the examples for the specific rule
    try:
        d = load_examples(Path(f"data/rule_{n}.jsonl"))
    except Exception as e:
        print(f"Failed to load data for rule {n}: {e}")
        all_results.append({
            "rule_number": n,
            "rule_text": true_rule_text,
            "accuracy": 0.0,
            "error": f"Failed to load data: {e}"
        })
        continue

    # test for all examples in the save data and report the mean
    correct = []
    for item in d:
        example = item['text']
        label = item['label']
        try:
            rule_execution = rule(example)
            correct.append(label == rule_execution)
        except Exception as e:
            print(f"Error executing rule {n} on example: {e}")
            correct.append(False)

    accuracy = np.mean(correct).item()
    print(f"Rule {n} accuracy: {accuracy:.2%}")

    all_results.append({
        "rule_number": n,
        "rule_text": true_rule_text,
        "accuracy": accuracy,
        "generated_function": extracted_json['function'],
        "raw_response": responses
    })

# Save results
output_path = Path("results/appendix/can_gpt_synthesise.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)

# Print summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for result in all_results:
    print(f"Rule {result['rule_number']}: {result['accuracy']:.2%}")
print("="*50)
overall_accuracy = np.mean([r['accuracy'] for r in all_results])
print(f"Overall accuracy: {overall_accuracy:.2%}")