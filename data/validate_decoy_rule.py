############################################################################################################
# validate_decoy_rule.py
# Evaluate each canonical rule and its decoy variants on the corresponding rule_{n}.jsonl dataset.
# Usage:
#   python data/validate_decoy_rule.py
############################################################################################################

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate how true rules and their decoys score on rule_{n}.jsonl datasets."
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
        help="Path to rules.py containing true rule_* functions.",
    )
    parser.add_argument(
        "--decoys-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "rules" / "decoy_rules.py",
        help="Path to decoy_rules.py containing rule_*_decoy_* functions.",
    )
    return parser.parse_args()


def load_rule_source(source_text: str, rule_number: int) -> str:
    pattern = re.compile(
        r"(def\s+rule_(\d+)\s*\(.*?)(?=^\s*def\s+rule_\d+\s*\(|\Z)",
        re.S | re.M,
    )
    for match in pattern.finditer(source_text):
        number = int(match.group(2))
        if number == rule_number:
            return match.group(1).strip()
    raise ValueError(f"rule_{rule_number} not found in source.")


def load_decoy_source(source_text: str, rule_number: int, decoy_number: int) -> str:
    pattern = re.compile(
        r"(def\s+rule_(\d+)_decoy_(\d+)\s*\(.*?)(?=^\s*def\s+rule_\d+_decoy_\d+\s*\(|\Z)",
        re.S | re.M,
    )
    for match in pattern.finditer(source_text):
        rule_num = int(match.group(2))
        decoy_num = int(match.group(3))
        if rule_num == rule_number and decoy_num == decoy_number:
            return match.group(1).strip()
    return ""


def build_callable(source: str, name: str):
    namespace: Dict[str, object] = {}
    exec(source, {"re": re}, namespace)
    fn = namespace.get(name)
    if fn is None:
        raise RuntimeError(f"{name} could not be constructed.")
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


def evaluate(records: List[Dict[str, object]], rule_fn) -> Tuple[int, int, float]:
    total = len(records)
    correct = 0
    for rec in records:
        text = rec.get("text", "")
        label = bool(rec.get("label"))
        predicted = bool(rule_fn(text))
        if predicted == label:
            correct += 1
    accuracy = (correct / total * 100.0) if total else 0.0
    return total, correct, accuracy


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist.")
    if not args.rules_path.exists():
        raise FileNotFoundError(f"rules.py not found at {args.rules_path}.")
    if not args.decoys_path.exists():
        raise FileNotFoundError(f"decoy_rules.py not found at {args.decoys_path}.")

    rules_text = args.rules_path.read_text(encoding="utf-8")
    decoys_text = args.decoys_path.read_text(encoding="utf-8")

    rule_file_pattern = re.compile(r"rule_(\d+)\.jsonl$")

    jsonl_files = [
        p for p in args.data_dir.iterdir() if p.is_file() and rule_file_pattern.match(p.name)
    ]
    jsonl_files.sort(key=lambda path: int(rule_file_pattern.match(path.name).group(1)))
    if not jsonl_files:
        print("No rule_{n}.jsonl files found. Nothing to validate.")
        return

    for path in jsonl_files:
        rule_number = int(rule_file_pattern.match(path.name).group(1))
        rule_source = load_rule_source(rules_text, rule_number)
        true_rule = build_callable(rule_source, f"rule_{rule_number}")

        decoy_functions = []
        idx = 1
        while True:
            decoy_source = load_decoy_source(decoys_text, rule_number, idx)
            if not decoy_source:
                break
            decoy_fn = build_callable(decoy_source, f"rule_{rule_number}_decoy_{idx}")
            decoy_functions.append((idx, decoy_fn))
            idx += 1

        records = read_jsonl(path)
        total, correct, accuracy = evaluate(records, true_rule)

        print(f"Rule {rule_number} | {path.name} | samples={total}")
        print(f"  true       : {accuracy:6.2f}% ({correct}/{total})")
        for idx, decoy_fn in decoy_functions:
            _, decoy_correct, decoy_acc = evaluate(records, decoy_fn)
            print(f"  decoy {idx} : {decoy_acc:6.2f}% ({decoy_correct}/{total})")
        if not decoy_functions:
            print("  (no decoys defined)")
        print()


if __name__ == "__main__":
    main()
