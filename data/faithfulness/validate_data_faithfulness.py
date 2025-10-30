############################################################################################################
# validate_data_faithfulness.py
# Checker for faithfulness datasets where TRUE entries satisfy both rules and FALSE entries satisfy neither.
# Also inspects the individual rule_{n}.jsonl datasets for each rule appearing in a pair to see how both
# rules manifest there. Optionally builds and validates a test dataset of rule_2 vs rule_17 contrasts.
# python data/faithfulness/validate_data_faithfulness.py
############################################################################################################

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

RULE_PAIR_PATTERN = re.compile(r"rule_(\d+)_(\d+)\.jsonl$")
RULE_FUNC_PATTERN = re.compile(
    r"(def\s+rule_(\d+)\s*\(.*?)(?=^\s*def\s+rule_\d+\s*\(|\Z)",
    re.S | re.M,
)


############################################################################################################
# argument parsing
############################################################################################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate faithfulness datasets where labels reflect two rules simultaneously."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing rule_{a}_{b}.jsonl datasets (default: ./data/faithfulness).",
    )
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "src" / "rules" / "rules.py",
        help="Path to rules.py containing the canonical rule_* definitions.",
    )
    parser.add_argument(
        "--single-data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory containing rule_{n}.jsonl single-rule datasets (default: ../data).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="If set, remove any examples that fail the faithfulness constraint for their label.",
    )
    parser.add_argument(
        "--fix-target",
        type=str,
        default=None,
        help="When --fix is set, only rewrite this dataset (match by filename, e.g. rule_2_17.jsonl).",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="If set, gather XOR examples for rule_2 and rule_17 and write data/faithfulness/test.jsonl.",
    )
    return parser.parse_args()


############################################################################################################
# helpers
############################################################################################################
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
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def evaluate_dataset(
    records: List[Dict[str, object]],
    rule_a_fn,
    rule_b_fn,
    expectation: str,
) -> Tuple[
    float,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    total = len(records)
    positives = sum(1 for rec in records if rec.get("label") is True)
    negatives = total - positives
    texts = []

    mismatches = 0
    duplicates = 0  # computed after loop
    both_true = both_false = a_only = b_only = 0
    filtered_records: List[Dict[str, object]] = []
    record_details: List[Dict[str, object]] = []

    for rec in records:
        text = rec.get("text", "")
        label_original = rec.get("label")
        label_bool = bool(label_original)
        result_a = bool(rule_a_fn(text))
        result_b = bool(rule_b_fn(text))
        if result_a and result_b:
            both_true += 1
        elif (not result_a) and (not result_b):
            both_false += 1
        elif result_a and not result_b:
            a_only += 1
        else:
            b_only += 1

        expected = label_bool
        satisfies = result_a and result_b
        violates = (not result_a) and (not result_b)

        if expectation == "both":
            faithful = satisfies if expected else violates
        elif expectation == "rule_a":
            faithful = (result_a == expected)
        elif expectation == "rule_b":
            faithful = (result_b == expected)
        elif expectation == "none":
            faithful = True
        else:
            raise ValueError(f"Unknown expectation mode: {expectation}")

        if faithful:
            filtered_records.append(rec)
        else:
            mismatches += 1
        texts.append(text)
        record_details.append(
            {
                "text": text,
                "label": label_original,
                "label_bool": label_bool,
                "result_a": result_a,
                "result_b": result_b,
            }
        )

    duplicates = len(texts) - len(set(texts))
    accuracy_pct = (1 - mismatches / total) * 100 if total else 0.0
    return (
        accuracy_pct,
        positives,
        negatives,
        mismatches,
        duplicates,
        both_true,
        both_false,
        a_only,
        b_only,
        filtered_records,
        record_details,
    )


def adjust_first_alpha_case(text: str, make_upper: bool) -> Optional[str]:
    chars = list(text)
    for idx, ch in enumerate(chars):
        if ch.isalpha():
            new_char = ch.upper() if make_upper else ch.lower()
            if new_char == ch:
                return None
            chars[idx] = new_char
            return "".join(chars)
    return None


def maybe_collect_create_samples(
    create_flag: bool,
    create_records: Dict[str, List[Dict[str, object]]],
    create_seen: Set[str],
    create_goal: int,
    rule_a: int,
    rule_b: int,
    record_details: List[Dict[str, object]],
    rule_cache: Dict[int, object],
) -> None:
    if (
        not create_flag
        or {rule_a, rule_b} != {2, 17}
        or (
            len(create_records["2_true_17_false"]) >= create_goal
            and len(create_records["2_false_17_true"]) >= create_goal
        )
    ):
        return

    rule2_fn = rule_cache[2]
    rule17_fn = rule_cache[17]

    for detail in record_details:
        if (
            len(create_records["2_true_17_false"]) >= create_goal
            and len(create_records["2_false_17_true"]) >= create_goal
        ):
            break

        original_text = detail["text"]
        rule2_val = (
            detail["result_a"]
            if rule_a == 2
            else detail["result_b"]
            if rule_b == 2
            else bool(rule2_fn(original_text))
        )
        rule17_val = (
            detail["result_a"]
            if rule_a == 17
            else detail["result_b"]
            if rule_b == 17
            else bool(rule17_fn(original_text))
        )

        candidates = []
        if rule2_val != rule17_val:
            candidates.append((original_text, rule2_val, rule17_val))
        else:
            if rule2_val and rule17_val and len(create_records["2_false_17_true"]) < create_goal:
                modified = adjust_first_alpha_case(original_text, make_upper=False)
                if modified and modified not in create_seen:
                    new_rule2 = bool(rule2_fn(modified))
                    new_rule17 = bool(rule17_fn(modified))
                    if not new_rule2 and new_rule17:
                        candidates.append((modified, new_rule2, new_rule17))
            elif (not rule2_val) and (not rule17_val) and len(create_records["2_true_17_false"]) < create_goal:
                modified = adjust_first_alpha_case(original_text, make_upper=True)
                if modified and modified not in create_seen:
                    new_rule2 = bool(rule2_fn(modified))
                    new_rule17 = bool(rule17_fn(modified))
                    if new_rule2 and not new_rule17:
                        candidates.append((modified, new_rule2, new_rule17))

        for text, rule2_out, rule17_out in candidates:
            key = "2_true_17_false" if rule2_out else "2_false_17_true"
            if len(create_records[key]) >= create_goal or text in create_seen:
                continue
            record_copy = {
                "text": text,
                "label": bool(rule2_out),
                "rule_2": bool(rule2_out),
                "rule_17": bool(rule17_out),
                "original_label": detail["label"],
            }
            create_records[key].append(record_copy)
            create_seen.add(text)
            break



def pretty_print(results: List[Dict[str, object]]) -> None:
    header = (
        f"{'ID':<22}"
        f"{'Type':<10}"
        f"{'Samples':<8}"
        f"{'Pos/Neg':<12}"
        f"{'Acc%':<8}"
        f"{'Mismatch':<10}"
        f"{'Dupes':<8}"
        f"{'BothT':<8}"
        f"{'BothF':<8}"
        f"{'A-only':<8}"
        f"{'B-only':<8}"
        f"{'File'}"
    )
    print(header)
    print("-" * len(header))
    for entry in results:
        positives = entry["positives"]
        negatives = entry["negatives"]
        mismatches = entry["mismatches"]
        duplicates = entry["duplicates"]
        accuracy_pct = entry["accuracy_pct"]
        both_true = entry["both_true"]
        both_false = entry["both_false"]
        a_only = entry["a_only"]
        b_only = entry["b_only"]
        filename = entry["filename"]
        dataset_id = entry["id"]
        dataset_type = entry["dataset_type"]
        samples = positives + negatives
        pos_neg = f"{positives}/{negatives}"
        print(
            f"{dataset_id:<22}"
            f"{dataset_type:<10}"
            f"{samples:<8}"
            f"{pos_neg:<12}"
            f"{accuracy_pct:>5.1f}% "
            f"{mismatches:<10}"
            f"{duplicates:<8}"
            f"{both_true:<8}"
            f"{both_false:<8}"
            f"{a_only:<8}"
            f"{b_only:<8}"
            f"{filename}"
        )


############################################################################################################
def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist.")
    if not args.rules_path.exists():
        raise FileNotFoundError(f"rules.py not found at {args.rules_path}.")
    if not args.single_data_dir.exists():
        raise FileNotFoundError(f"Single-rule data directory {args.single_data_dir} does not exist.")

    rules_text = args.rules_path.read_text(encoding="utf-8")
    jsonl_files = sorted(
        (
            p
            for p in args.data_dir.iterdir()
            if p.is_file() and RULE_PAIR_PATTERN.match(p.name)
        ),
        key=lambda path: tuple(int(num) for num in RULE_PAIR_PATTERN.match(path.name).groups()),
    )
    if not jsonl_files:
        print("No rule_{a}_{b}.jsonl files found. Nothing to validate.")
        return

    rule_cache: Dict[int, object] = {}
    fix_target_name = None
    if args.fix_target:
        target_path = Path(args.fix_target)
        fix_target_name = target_path.name
        if not fix_target_name.endswith(".jsonl"):
            fix_target_name = f"{fix_target_name}.jsonl"
    # Ensure rule_2 and rule_17 functions are cached if needed later.
    if args.create:
        for rule_num in (2, 17):
            if rule_num not in rule_cache:
                source = load_rule_source(rules_text, rule_num)
                rule_cache[rule_num] = build_rule_callable(source, rule_num)

    create_records: Dict[str, List[Dict[str, object]]] = {
        "2_true_17_false": [],
        "2_false_17_true": [],
    }
    create_seen: Set[str] = set()
    create_goal = 100
    results = []

    for path in jsonl_files:
        match = RULE_PAIR_PATTERN.match(path.name)
        assert match is not None
        rule_a = int(match.group(1))
        rule_b = int(match.group(2))

        if rule_a not in rule_cache:
            source_a = load_rule_source(rules_text, rule_a)
            rule_cache[rule_a] = build_rule_callable(source_a, rule_a)
        if rule_b not in rule_cache:
            source_b = load_rule_source(rules_text, rule_b)
            rule_cache[rule_b] = build_rule_callable(source_b, rule_b)

        rule_a_fn = rule_cache[rule_a]
        rule_b_fn = rule_cache[rule_b]

        records = read_jsonl(path)
        (
            accuracy_pct,
            positives,
            negatives,
            mismatches,
            duplicates,
            both_true,
            both_false,
            a_only,
            b_only,
            filtered_records,
            record_details,
        ) = evaluate_dataset(records, rule_a_fn, rule_b_fn, expectation="both")

        maybe_collect_create_samples(
            args.create,
            create_records,
            create_seen,
            create_goal,
            rule_a,
            rule_b,
            record_details,
            rule_cache,
        )

        should_fix = args.fix and (fix_target_name is None or fix_target_name == path.name)
        if should_fix and len(filtered_records) < len(records):
            removed = len(records) - len(filtered_records)
            with path.open("w", encoding="utf-8") as fh:
                for rec in filtered_records:
                    fh.write(json.dumps(rec, ensure_ascii=False))
                    fh.write("\n")
            print(f"[fix] Removed {removed} mismatched example(s) from {path.name}")
            (
                accuracy_pct,
                positives,
                negatives,
                mismatches,
                duplicates,
                both_true,
                both_false,
                a_only,
                b_only,
                filtered_records,
                _,
            ) = evaluate_dataset(filtered_records, rule_a_fn, rule_b_fn, expectation="both")

        pair_key = f"{rule_a}/{rule_b}"
        results.append(
            {
                "id": pair_key,
                "dataset_type": "pair",
                "positives": positives,
                "negatives": negatives,
                "mismatches": mismatches,
                "duplicates": duplicates,
                "accuracy_pct": accuracy_pct,
                "both_true": both_true,
                "both_false": both_false,
                "a_only": a_only,
                "b_only": b_only,
                "filename": path.name,
            }
        )

        # Evaluate individual datasets for rule_a and rule_b (if they exist), showing how both rules behave.
        single_a_path = args.single_data_dir / f"rule_{rule_a}.jsonl"
        if single_a_path.exists():
            single_records = read_jsonl(single_a_path)
            (
                single_acc,
                single_pos,
                single_neg,
                single_mismatches,
                single_dupes,
                single_both_true,
                single_both_false,
                single_a_only,
                single_b_only,
                _,
                single_details,
            ) = evaluate_dataset(single_records, rule_a_fn, rule_b_fn, expectation="rule_a")
            results.append(
                {
                    "id": f"{pair_key} :: rule_{rule_a}",
                    "dataset_type": "single",
                    "positives": single_pos,
                    "negatives": single_neg,
                    "mismatches": single_mismatches,
                    "duplicates": single_dupes,
                    "accuracy_pct": single_acc,
                    "both_true": single_both_true,
                    "both_false": single_both_false,
                    "a_only": single_a_only,
                    "b_only": single_b_only,
                    "filename": single_a_path.name,
                }
            )
            maybe_collect_create_samples(
                args.create,
                create_records,
                create_seen,
                create_goal,
                rule_a,
                rule_b,
                single_details,
                rule_cache,
            )
        else:
            print(f"[warn] Single-rule dataset not found: {single_a_path}")

        single_b_path = args.single_data_dir / f"rule_{rule_b}.jsonl"
        if single_b_path.exists():
            single_records = read_jsonl(single_b_path)
            (
                single_acc,
                single_pos,
                single_neg,
                single_mismatches,
                single_dupes,
                single_both_true,
                single_both_false,
                single_a_only,
                single_b_only,
                _,
                single_details,
            ) = evaluate_dataset(single_records, rule_a_fn, rule_b_fn, expectation="rule_b")
            results.append(
                {
                    "id": f"{pair_key} :: rule_{rule_b}",
                    "dataset_type": "single",
                    "positives": single_pos,
                    "negatives": single_neg,
                    "mismatches": single_mismatches,
                    "duplicates": single_dupes,
                    "accuracy_pct": single_acc,
                    "both_true": single_both_true,
                    "both_false": single_both_false,
                    "a_only": single_a_only,
                    "b_only": single_b_only,
                    "filename": single_b_path.name,
                }
            )
            maybe_collect_create_samples(
                args.create,
                create_records,
                create_seen,
                create_goal,
                rule_a,
                rule_b,
                single_details,
                rule_cache,
            )
        else:
            print(f"[warn] Single-rule dataset not found: {single_b_path}")

    if args.create:
        positives_collected = len(create_records["2_true_17_false"])
        negatives_collected = len(create_records["2_false_17_true"])
        if positives_collected == create_goal and negatives_collected == create_goal:
            output_path = args.data_dir / "test.jsonl"
            combined_records = create_records["2_true_17_false"] + create_records["2_false_17_true"]
            with output_path.open("w", encoding="utf-8") as fh:
                for record in combined_records:
                    fh.write(json.dumps(record, ensure_ascii=False))
                    fh.write("\n")
            print(
                f"[create] Wrote {create_goal} rule_2-only and {create_goal} rule_17-only examples to {output_path}."
            )
        else:
            print(
                f"[warn] Unable to gather required examples for test.jsonl "
                f"(rule_2-only={positives_collected}, rule_17-only={negatives_collected}; "
                f"needed {create_goal} each)."
            )

    test_path = args.data_dir / "test.jsonl"
    if test_path.exists():
        for rule_num in (2, 17):
            if rule_num not in rule_cache:
                source = load_rule_source(rules_text, rule_num)
                rule_cache[rule_num] = build_rule_callable(source, rule_num)
        test_records = read_jsonl(test_path)
        (
            test_acc,
            test_pos,
            test_neg,
            test_mismatches,
            test_dupes,
            test_both_true,
            test_both_false,
            test_a_only,
            test_b_only,
            _,
            _,
        ) = evaluate_dataset(test_records, rule_cache[2], rule_cache[17], expectation="none")
        results.append(
            {
                "id": "test",
                "dataset_type": "xor",
                "positives": test_pos,
                "negatives": test_neg,
                "mismatches": test_mismatches,
                "duplicates": test_dupes,
                "accuracy_pct": test_acc,
                "both_true": test_both_true,
                "both_false": test_both_false,
                "a_only": test_a_only,
                "b_only": test_b_only,
                "filename": test_path.name,
            }
        )
        if (
            test_a_only != create_goal
            or test_b_only != create_goal
            or test_both_true
            or test_both_false
        ):
            print(
                f"[warn] test.jsonl expected {create_goal} rule_2-only and {create_goal} rule_17-only items; "
                f"observed counts both_true={test_both_true}, both_false={test_both_false}, "
                f"a_only={test_a_only}, b_only={test_b_only}."
            )
    else:
        print(f"[warn] Test dataset not found at {test_path}.")

    pretty_print(results)


if __name__ == "__main__":
    main()
