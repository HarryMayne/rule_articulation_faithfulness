############################################################################################################
# make_data.py: Helper that asks GPT-5 for JSONL examples of a chosen rule_* function.
# No validation here. Simple.
# Usage: python3 make_data.py --rule-number 1 --examples 10
############################################################################################################

import argparse
import os
import re
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# main prompt.
main_prompt = """
You are a research assistant for an AI safety paper being conducted at the University of Oxford. The goal of the paper is to test whether LLMs can articule rules they learn from few-shot examples. The goal of the project is to make meaningful contributions to AI safety and to write the results up into a NeurIPS paper. Your task in the project is to consider one of the rules and generate lots of diverse data satisfying or failing the rule. You should create a balanced JSONL datasets for the rule. Note that your generations will be checked with the Python implementation of the rule.

Rule definition:
```
{rule_source}
```

Please return {total_examples} total examples (balanced: {positives} True / {negatives} False).
Make sure positives satisfy rule_{rule_number}(text) and negatives clearly do not.
Keep each entry natural and distinct. No placeholders or numbered lists. It should be a standard phrase, sentence or sentences.

Output format:
- newline-delimited JSON (JSONL)
- each line: {{"text": "example sentence", "label": true}} (or false)
- no prose or code fences—just the JSON objects
- The text should NOT refer to the rule. It should never give the rule away. This is very important.

Begin immediately with the first JSON object.
""".strip()

# args. 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a quick JSONL dataset for a specific rule_* function."
    )
    parser.add_argument(
        "--rule-number",
        type=int,
        required=True,
        help="Target rule number (e.g., --rule-number 4 for rule_4).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Total examples to request (use an even number to keep the split balanced).",
    )
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "rules" / "rules.py",
        help="Path to the canonical rules.py file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder where rule_{n}.jsonl will be written (default: ./data).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5-2025-08-07"),
        help="OpenAI model name (defaults to env OPENAI_MODEL or gpt-5-2025-08-07).",
    )
    return parser.parse_args()

# pull the rule out of the .py file with regex.
def extract_rule_source(rules_text: str, rule_number: int, rules_path: Path) -> str:
    """
    Grab the exact rule_* function from the full rules.py file via a simple regex.
    The file follows a regular format, so this is reliable enough for quick scripts.
    """
    pattern = re.compile(
        rf"(def\s+rule_{rule_number}\s*\(.*?)(?=^\s*def\s+rule_\d+\s*\(|\Z)",
        re.S | re.M,
    )
    match = pattern.search(rules_text)
    if not match:
        raise ValueError(f"Could not find rule_{rule_number} inside {rules_path}")
    return match.group(0).strip()

# do the request
def request_examples(client: OpenAI, model: str, prompt: str) -> str:
    """
    Lightweight wrapper around the chat completions API.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful data assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    args = parse_args()

    if args.examples <= 0 or args.examples % 2 != 0:
        raise ValueError("--examples must be a positive even number for balance.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Please export it before running.")

    # Read the rules file and slice out the function we're interested in.
    rules_text = args.rules_path.read_text(encoding="utf-8")
    rule_source = extract_rule_source(rules_text, args.rule_number, args.rules_path)

    # Double braces keep the prompt format stable when the rule itself contains braces.
    safe_rule_source = rule_source.replace("{", "{{").replace("}", "}}")
    prompt = main_prompt.format(
        rule_source=safe_rule_source,
        rule_number=args.rule_number,
        total_examples=args.examples,
        positives=args.examples // 2,
        negatives=args.examples // 2,
    )

    # GPT-5 and store whatever it returns 
    # ..,.
    client = OpenAI()
    output_text = request_examples(client, args.model, prompt)

    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Model response was empty; nothing to write.")
    random.shuffle(lines)
    shuffled_text = "\n".join(lines) + "\n"

    output_path = args.output_dir / f"rule_{args.rule_number}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(shuffled_text, encoding="utf-8")

    print(f"Wrote examples to {output_path}.")


if __name__ == "__main__":
    main()
