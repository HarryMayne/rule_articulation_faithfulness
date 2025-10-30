############################################################################################################
# make_data.py: Helper that asks GPT-5 for JSONL examples of a chosen rule_* function.
# Usage: python3 make_data.py --rule-number 1 --examples 10
# just does it all in one generation for ££ effeciency 
############################################################################################################

import argparse
import json
import os
import random
import re
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

{existing_examples}

Please return {total_examples} total examples (balanced: {positives} True / {negatives} False).
Make sure positives satisfy rule_{rule_number}(text) and negatives clearly do not.
Keep each entry natural and distinct. No placeholders or numbered lists. It should be a standard phrase, sentence or sentences.
Do not repeat any earlier examples (if any are shown above); generate fresh text only.

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

# Load the existing records....
def read_existing_dataset(path: Path):
    """
    Return (records, raw_text) for any existing dataset so we can include it in the prompt
    and guard against duplicates.
    """
    if not path.exists():
        return [], ""
    records = []
    raw_lines = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            raw_lines.append(stripped)
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                pass
    return records, "\n".join(raw_lines)

# parse the new example...
def parse_new_examples(raw: str):
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records

# do the request to OAI
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


################################################################################################
def main() -> None:
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Please export it before running.")

    output_path = args.output_dir / f"rule_{args.rule_number}.jsonl"
    existing_records, existing_raw = read_existing_dataset(output_path)
    existing_block = ""
    if existing_raw:
        safe_existing = existing_raw.replace("{", "{{").replace("}", "}}")
        existing_block = (
            "Existing dataset already collected (avoid repeating these JSONL lines):\n"
            "```\n"
            f"{safe_existing}\n"
            "```\n"
        )

    # Read the rules file and slice out the function we're interested in.
    rules_text = args.rules_path.read_text(encoding="utf-8")
    rule_source = extract_rule_source(rules_text, args.rule_number, args.rules_path)

    # Double braces keep the prompt format stable when the rule itself contains braces.
    safe_rule_source = rule_source.replace("{", "{{").replace("}", "}}")
    prompt = main_prompt.format(
        rule_source=safe_rule_source,
        existing_examples=existing_block,
        rule_number=args.rule_number,
        total_examples=args.examples,
        positives=args.examples // 2,
        negatives=args.examples // 2,
    )

    # GPT-5 and store whatever it returns 
    # ..,.
    client = OpenAI()
    output_text = request_examples(client, args.model, prompt)

    new_records = parse_new_examples(output_text)
    #validate_new_examples(new_records, args.examples, existing_texts)
    random.shuffle(new_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    needs_separator = False
    if file_exists:
        size = output_path.stat().st_size
        if size > 0:
            with output_path.open("rb") as existing_fh:
                existing_fh.seek(-1, os.SEEK_END)
                last_char = existing_fh.read(1)
            needs_separator = last_char != b"\n"

    with output_path.open("a" if file_exists else "w", encoding="utf-8") as fh:
        if needs_separator:
            fh.write("\n")
        for record in new_records:
            fh.write(json.dumps(record, ensure_ascii=False))
            fh.write("\n")

    # Append the new examples to the end of the existing jso nl file
    print(f"Appended {len(new_records)} fresh examples to {output_path}.")


if __name__ == "__main__":
    main()
