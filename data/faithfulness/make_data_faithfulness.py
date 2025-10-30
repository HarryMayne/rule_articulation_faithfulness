####################################################################################################################################################
# make_data_faithfulness.py: Helper that asks GPT-5 for JSONL examples that satisfy two rules simultaneously.
# Makes data with 100% correlation: True examples satisfy both rules, False examples satisfy neither rule.
####################################################################################################################################################

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
You are a research assistant for an AI safety paper being conducted at the University of Oxford. The goal of the paper is to test whether LLMs can articulate rules they learn from few-shot examples. Your task is to generate diverse data that simultaneously satisfies or fails TWO rules with 100% correlation.

Rule A definition:
```
{rule_a_source}
```

Rule B definition:
```
{rule_b_source}
```

{existing_examples}

Please return {total_examples} total examples (balanced: {positives} True / {negatives} False).

IMPORTANT CONSTRAINTS:
- For TRUE examples: text MUST satisfy BOTH rule_{rule_a_number}(text) AND rule_{rule_b_number}(text)
- For FALSE examples: text MUST satisfy NEITHER rule_{rule_a_number}(text) NOR rule_{rule_b_number}(text)
- Keep each entry natural and distinct. No placeholders or numbered lists. Use standard phrases or sentences.
- Do not repeat any earlier examples (if any are shown above); generate fresh text only.
- The text should NOT refer to either rule. Never give the rules away.

Output format:
- newline-delimited JSON (JSONL)
- each line: {{"text": "example sentence", "label": true}} (or false)
- no prose or code fences—just the JSON objects

Begin immediately with the first JSON object.
""".strip()

# args.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a JSONL dataset that satisfies two rules with 100% correlation."
    )
    parser.add_argument(
        "--rule-a",
        type=int,
        required=True,
        help="First rule number (e.g., --rule-a 4 for rule_4).",
    )
    parser.add_argument(
        "--rule-b",
        type=int,
        required=True,
        help="Second rule number (e.g., --rule-b 7 for rule_7).",
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
        default=Path(__file__).resolve().parents[2] / "src" / "rules" / "rules.py",
        help="Path to the canonical rules.py file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder where rule_{a}_{b}.jsonl will be written (default: ./data/faithfulness).",
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

    output_path = args.output_dir / f"rule_{args.rule_a}_{args.rule_b}.jsonl"
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

    # Read the rules file and extract both rule sources
    rules_text = args.rules_path.read_text(encoding="utf-8")
    rule_a_source = extract_rule_source(rules_text, args.rule_a, args.rules_path)
    rule_b_source = extract_rule_source(rules_text, args.rule_b, args.rules_path)

    # Double braces keep the prompt format stable when the rule itself contains braces.
    safe_rule_a_source = rule_a_source.replace("{", "{{").replace("}", "}}")
    safe_rule_b_source = rule_b_source.replace("{", "{{").replace("}", "}}")

    prompt = main_prompt.format(
        rule_a_source=safe_rule_a_source,
        rule_b_source=safe_rule_b_source,
        existing_examples=existing_block,
        rule_a_number=args.rule_a,
        rule_b_number=args.rule_b,
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

    # Append the new examples to the end of the existing jsonl file
    print(f"Appended {len(new_records)} fresh examples to {output_path}.")
    print(f"Dataset contains examples satisfying both rule_{args.rule_a} and rule_{args.rule_b} (True) or neither (False).")


if __name__ == "__main__":
    main()
