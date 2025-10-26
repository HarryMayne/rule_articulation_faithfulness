############################################################################################################
# validate_data.py
# Simple checker that runs each rule_* function over its matching rule_{n}.jsonl dataset.
# Reports class balance and correctness stats per rule.
############################################################################################################

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

# Regex to capture rule number from filenames and from rules.py content.
RULE_FILE_PATTERN = re.compile(r"rule_(\d+)\.jsonl$")
RULE_FUNC_PATTERN = re.compile(
    r"(def\s+rule_(\d+)\s*\(.*?)(?=^\s*def\s+rule_\d+\s*\(|\Z)",
    re.S | re.M,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate balance and correctness for rule_{n}.jsonl files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing rule_{n}.jsonl datasets (default: ./data).",
    )
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "rules" / "rules.py",
        help="Path to rules.py containing the rule_* functions.",
    )
    return parser.parse_args()


def load_rule_source(rules_text: str, rule_number: int) -> str:
    for match in RULE_FUNC_PATTERN.finditer(rules_text):
        if int(match.group(2)) == rule_number:
            return match.group(0).strip()
    raise ValueError(f"rule_{rule_number} not found in rules.py")


def build_rule_callable(rule_source: str, rule_number: int):
    namespace: Dict[str, object] = {}
    exec(rule_source, {}, namespace)
    fn = namespace.get(f"rule_{rule_number}")
    if fn is None:
        raise RuntimeError(f"rule_{rule_number} could not be constructed.")
    return fn


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def evaluate_rule(records: List[Dict[str, object]], rule_fn) -> Tuple[bool, bool, int, int, int]:
    total = len(records)
    positives = sum(1 for rec in records if rec.get("label") is True)
    negatives = total - positives
    balanced = positives == negatives

    correct = 0
    for rec in records:
        text = rec.get("text", "")
        label = bool(rec.get("label"))
        predicted = bool(rule_fn(text))
        if predicted == label:
            correct += 1
    return balanced, correct == total, positives, negatives, total - correct


def pretty_print(results: List[Tuple[int, str, bool, bool, int, int, int]]) -> None:
    header = f"{'Rule':<6}{'Samples':<10}{'Pos/Neg':<12}{'Balanced':<10}{'Correct':<10}{'Errors':<8}{'File'}"
    print(header)
    print("-" * len(header))
    for rule_number, filename, balanced, correct, positives, negatives, errors in results:
        balance_str = "yes" if balanced else "no"
        correct_str = "yes" if correct else "no"
        print(
            f"{rule_number:<6}{positives+negatives:<10}{positives}/{negatives:<12}"
            f"{balance_str:<10}{correct_str:<10}{errors:<8}{filename}"
        )


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist.")
    if not args.rules_path.exists():
        raise FileNotFoundError(f"rules.py not found at {args.rules_path}.")

    rules_text = args.rules_path.read_text(encoding="utf-8")
    jsonl_files = sorted(
        p for p in args.data_dir.iterdir() if p.is_file() and RULE_FILE_PATTERN.match(p.name)
    )
    if not jsonl_files:
        print("No rule_{n}.jsonl files found. Nothing to validate.")
        return

    results = []
    for path in jsonl_files:
        rule_number = int(RULE_FILE_PATTERN.match(path.name).group(1))
        rule_source = load_rule_source(rules_text, rule_number)
        rule_fn = build_rule_callable(rule_source, rule_number)
        records = read_jsonl(path)
        balanced, correct, positives, negatives, errors = evaluate_rule(records, rule_fn)
        results.append((rule_number, path.name, balanced, correct, positives, negatives, errors))

    pretty_print(results)


if __name__ == "__main__":
    main()
