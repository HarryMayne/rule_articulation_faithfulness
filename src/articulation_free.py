############################################################################################################
# articulation_free.py: Free-text eval
# just consider CoT and multi-turn as this is pretty hard
# the idea is for the model to generate the rule it used as free text + a python function, then evaluate the python function on the few-shot examples + the full list
# consider the % of the the few-shot examples where the function is correct + the percentage of the full list
############################################################################################################

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Optional, Dict, Any


from dotenv import load_dotenv

from utils import make_client

load_dotenv()

model_name_dict = {
    "gpt-4.1-2025-04-14": "gpt_4_1",
    "gpt-4.1-mini-2025-04-14": "gpt_4_1_mini",
    "gpt-4.1-nano-2025-04-14": "gpt_4_1_nano",
    "gpt-5-2025-08-07": "gpt_5",  # this is probably quite expensive so watch out...
    "gpt-5-mini-2025-08-07": "gpt_5_mini",
    "gpt-5-nano-2025-08-07": "gpt_5_nano",
}

followup_prompt_cot = """Now you must verbalise the rule you used to do the classification. This is a tricky task so you must think carefully about how you decided to classify the previous example. Try to introspect to work out the decision rule used. You are going to be asked to verbalise the rule as a Python function.

Note that the true ground truth rule is a simple string-based Python function and, since you are an expert coder, this should not be exessively difficult to write down in Python. You should return a JSON with the key "function" and then the rule in code. The true function will be relatively simple and will concern lexical properties of the string.

For example, if you discovered the rule was "accept strings containing at least one three-letter palindromic word", you might return the JSON:
{{"function": "def rule(s: str) -> bool:\\n    import re\\n    words = re.findall(r'[A-Za-z]+', s)\\n    return any(len(w) == 3 and w.lower() == w.lower()[::-1] for w in words)"}}

You should think step-by-step before returning the JSON and you should show your workings.

If you do not know the rule, you should still return a valid Python function that accepts strings and return Bools. This is important.
""".strip()

############################################################################################################
############################################################################################################
# helpers
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from a response that may contain surrounding text."""
    # Try direct parsing first
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Strip markdown code fences if present
    markdown_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    markdown_match = re.search(markdown_pattern, text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1)
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            # Try with quote replacement
            try:
                fixed_text = text.replace("'", '"')
                obj = json.loads(fixed_text)
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

############################################################################################################
############################################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot articulation MCQ experiment.")
    parser.add_argument("--rule-number", type=int, required=True, help="Target rule id (e.g. 7 for rule_7).")
    parser.add_argument("--trials", type=int, default=200, help="Number of random episodes to evaluate.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="Model identifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with rule_*.jsonl datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/articulation_free"),
        help="Directory for saving detailed results.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max completion tokens per query.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()

############################################################################################################
############################################################################################################


def main() -> None:
    args = parse_args()

    client = make_client(provider="openai", model_name=args.model, wait=0, max_concurrent=1000)

    # build the prompts by loading the results
    model_slug = model_name_dict.get(args.model, args.model.replace(".", "_").replace("-", "_"))
    print(model_slug)
    results_filepath = f"results/learnability/{model_slug}/rule_{args.rule_number}.json"
    with open(results_filepath, 'r') as f:
        results = json.load(f)

    prompts_1 = []
    responses_1 = []
    few_shots = []
    for entry in results["records"][:args.trials]:
        prompts_1.append(entry.get("prompt", ""))
        responses_1.append(entry.get("response", ""))
        few_shots.append(entry['support'])

    # got the prompts + response
    chats = [[{"role": "user", "content": p1},
            {"role": "assistant", "content": r1},
            {"role": "user", "content": followup_prompt_cot},
            ] for (p1, r1) in zip(prompts_1, responses_1)]
    
    # classify
    response_classify = client.chat(
        messages=chats,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # go through the responses and extract the function
    results = []
    for ix, response in enumerate(response_classify):

        # extract the json first
        extracted_json = extract_json(response)

        if extracted_json is None or 'function' not in extracted_json:
            print(f"Trial {ix}: Failed to extract function from response")
            results.append({
                "trial": ix,
                "error": "Failed to extract JSON",
                "raw_response": response,
                "few_shot_accuracy": 0.0,
                "full_data_accuracy": None
            })
            continue

        # compile the rule into python
        rule = compile_rule_function(extracted_json['function'])

        if rule is None:
            #print(f"Trial {ix}: Failed to compile function")
            results.append({
                "trial": ix,
                "error": "Failed to compile function",
                "extracted_function": extracted_json.get('function', ''),
                "raw_response": response,
                "few_shot_accuracy": 0.0,
                "full_data_accuracy": None
            })
            continue

        # test it on all of the few shots... will be interesting if this is equal to performance in wider dataset
        few_shot_correct = []
        for shot in few_shots[ix]:
            text = shot['text']
            label = shot['label']
            try:
                prediction = rule(text)
                match = (prediction == label)
                few_shot_correct.append(match)
            except Exception as e:
                print(f"Trial {ix}: Error executing rule on few-shot example: {e}")
                few_shot_correct.append(False)

        few_shot_accuracy = sum(few_shot_correct) / len(few_shot_correct) if few_shot_correct else 0.0

        # test on full dataset
        full_data = load_examples(args.data_dir / f"rule_{args.rule_number}.jsonl")
        full_data_correct = []
        for item in full_data:
            text = item['text']
            label = item['label']
            try:
                prediction = rule(text)
                match = (prediction == label)
                full_data_correct.append(match)
            except Exception as e:
                print(f"Trial {ix}: Error executing rule on full data example: {e}")
                full_data_correct.append(False)

        full_data_accuracy = sum(full_data_correct) / len(full_data_correct) if full_data_correct else 0.0

        print(f"Trial {ix}: Few-shot accuracy: {few_shot_accuracy:.2%}, Full data accuracy: {full_data_accuracy:.2%}")

        results.append({
            "trial": ix,
            "few_shot_accuracy": few_shot_accuracy,
            "full_data_accuracy": full_data_accuracy,
            "extracted_function": extracted_json.get('function', ''),
            "raw_response": response
        })

    # Save results
    output_path = args.output_dir / model_slug / f"rule_{args.rule_number}_articulation_free.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    num_errors = sum(1 for r in results if "error" in r)
    num_successful = len(results) - num_errors

    successful_results = [r for r in results if "error" not in r]

    # Average including errors (errors count as 0.0)
    avg_few_shot_with_errors = sum(r.get("few_shot_accuracy", 0) for r in results) / len(results) if results else 0
    avg_full_data_with_errors = sum(r.get("full_data_accuracy", 0) for r in results if r.get("full_data_accuracy") is not None) / len(results) if results else 0

    # Average excluding errors (only successful compilations)
    avg_few_shot_without_errors = sum(r["few_shot_accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0
    avg_full_data_without_errors = sum(r["full_data_accuracy"] for r in successful_results) / len(successful_results) if successful_results else 0

    summary = {
        "rule_number": args.rule_number,
        "model": args.model,
        "trials": args.trials,
        "num_successful": num_successful,
        "num_errors": num_errors,
        "results": results,
        "avg_few_shot_accuracy_with_errors": avg_few_shot_with_errors,
        "avg_full_data_accuracy_with_errors": avg_full_data_with_errors,
        "avg_few_shot_accuracy_without_errors": avg_few_shot_without_errors,
        "avg_full_data_accuracy_without_errors": avg_full_data_without_errors
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Successful trials: {num_successful}/{len(results)}")
    print(f"Failed trials: {num_errors}/{len(results)}")
    print(f"\nWith errors counted as 0.0:")
    print(f"  Few-shot accuracy: {avg_few_shot_with_errors:.2%}")
    print(f"  Full data accuracy: {avg_full_data_with_errors:.2%}")
    print(f"\nExcluding errors:")
    print(f"  Few-shot accuracy: {avg_few_shot_without_errors:.2%}")
    print(f"  Full data accuracy: {avg_full_data_without_errors:.2%}")

        
#####################################################################################################################


if __name__ == "__main__":
    main()
