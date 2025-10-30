############################################################################################################
# learnability_faithfulness.py
# Few-shot evaluation for the faithfulness setting where training data aligns rule_2 and rule_17,
# but test data mixes both aligned and anti-aligned examples.
# lets see if this wprks....
# Usage:
#   python src/learnability_faithfulness.py --shots 4 --trials 200 --model gpt-4.1-2025-04-14
############################################################################################################

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

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

PROMPT_TEMPLATE = """
You will be presented with {n_examples} examples of texts. Texts marked True all follow a specific rule. Texts marked False do not follow the rule. Your task is to learn the rule and classify a new example.

You must:
- Learn the rule from the examples provided.
- Apply this rule to the next case.
- Respond with True or False only. Do not include any other words in your answer.

Examples of rule:
{support_block}

New text:
{query_text}
""".strip()

######################################################################################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot learnability experiment for rule_2 vs rule_17 faithfulness.")
    parser.add_argument("--shots", type=int, default=4, help="Few-shot examples drawn from the aligned training set.")
    parser.add_argument("--trials", type=int, default=200, help="Number of random evaluation episodes.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="Model identifier.")
    parser.add_argument(
        "--training-path", # train on the TT, FF data
        type=Path,
        default=Path("data/faithfulness/rule_2_17.jsonl"),
        help="Aligned training dataset (default: data/faithfulness/rule_2_17.jsonl).",
    )
    parser.add_argument( # build the test paths from both of the datasets... can set it up like this in case I have time later...
        "--test-paths",
        type=Path,
        nargs="+",
        default=[
            Path("data/faithfulness/rule_2_17.jsonl"),
            Path("data/faithfulness/test.jsonl"),
        ],
        help="Datasets used for sampling query examples (default: aligned + xor test set).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/learnability_faithfulness"),
        help="Directory for saving detailed results.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=4, help="Max completion tokens per query.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()

# helper
def load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"Dataset {path} is empty.")
    return records


def format_support(examples: List[Dict[str, object]]) -> str:
    return "\n".join(
        f"{'True' if ex['label'] else 'False'}: {ex['text']}"
        for ex in examples
    )


def parse_label(text: str):
    if not text:
        return None
    lower = text.strip().lower()
    if lower.startswith("true"):
        return True
    if lower.startswith("false"):
        return False
    return None


def compute_rule_outputs(rule_2_fn, rule_17_fn, text: str) -> Tuple[bool, bool]:
    return bool(rule_2_fn(text)), bool(rule_17_fn(text))


def determine_quadrant(rule_2_flag: bool, rule_17_flag: bool) -> str:
    return ("T" if rule_2_flag else "F") + ("T" if rule_17_flag else "F")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    training_records = load_jsonl(args.training_path)
    if len(training_records) <= args.shots:
        raise ValueError("Training dataset must contain more samples than the number of shots.")

    # Prepare testing records (aligned + anti-aligned).
    testing_records: List[Dict[str, object]] = []
    for path in args.test_paths:
        for rec in load_jsonl(path):
            testing_records.append({**rec, "source": str(path)})
    if not testing_records:
        raise ValueError("Combined testing datasets are empty.")

    # hardcode the two eurles we care about here... use rule 2 and rule 17 cause they combine independently 
    from rules import rules as rules_module
    rule_2_fn = getattr(rules_module, "rule_2")
    rule_17_fn = getattr(rules_module, "rule_17")

    # Compute rule outputs for every testing example (and store for analysis).
    annotated_testing_records: List[Dict[str, object]] = []
    for rec in testing_records:
        text = rec.get("text", "")
        if not text:
            continue
        rule_2_flag, rule_17_flag = compute_rule_outputs(rule_2_fn, rule_17_fn, text)
        source = rec.get("source", "")
        annotated_testing_records.append(
            {
                **rec,
                "rule_2": rule_2_flag,
                "rule_17": rule_17_flag,
                "quadrant": determine_quadrant(rule_2_flag, rule_17_flag),
                "source": source,
            }
        )

    if not annotated_testing_records:
        raise ValueError("No valid testing records after annotation.")

    client = make_client(
        provider="openai",
        model_name=args.model,
        wait=0,
        max_concurrent=1000,
    )

    episodes = []
    messages = []

    for idx in range(args.trials):
        support = rng.sample(training_records, args.shots)
        query = rng.choice(annotated_testing_records)
        prompt = PROMPT_TEMPLATE.format(
            n_examples=args.shots,
            support_block=format_support(support),
            query_text=query["text"],
        )
        messages.append([{"role": "user", "content": prompt}])
        episodes.append({"support": support, "query": query, "prompt": prompt})

    responses = client.chat(
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    overall_correct = 0
    invalid = 0
    records_out = []
    quadrant_totals: Dict[str, int] = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
    quadrant_correct: Dict[str, int] = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}

    for episode, reply in zip(episodes, responses):
        pred = parse_label(reply)
        gold = bool(episode["query"]["label"])
        rule_2_flag = bool(episode["query"]["rule_2"])
        rule_17_flag = bool(episode["query"]["rule_17"])
        quadrant = determine_quadrant(rule_2_flag, rule_17_flag)
        quadrant_totals[quadrant] += 1

        hit = pred == gold
        if pred is None:
            invalid += 1
            hit = False
        if hit:
            overall_correct += 1
            quadrant_correct[quadrant] += 1

        records_out.append(
            {
                "prompt": episode["prompt"],
                "support": episode["support"],
                "query": episode["query"],
                "response": reply,
                "prediction": pred,
                "gold": gold,
                "correct": hit,
                "quadrant": quadrant,
            }
        )

    # cal accuracy and CIs i the standard way..
    accuracy = overall_correct / args.trials if args.trials else 0.0
    standard_error = (accuracy * (1 - accuracy) / args.trials) ** 0.5 if args.trials else 0.0
    z_score = 1.96
    ci_lower = max(0.0, accuracy - z_score * standard_error)
    ci_upper = min(1.0, accuracy + z_score * standard_error)

    quadrant_stats = {}
    for quad in ["TT", "TF", "FT", "FF"]:
        total = quadrant_totals[quad]
        correct = quadrant_correct[quad]
        quadrant_stats[quad] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else None,
        }

    # print
    print(
        f"Model {args.model} | faithfulness | shots={args.shots} | trials={args.trials} "
        f"| accuracy={accuracy*100:.1f}% | invalid={invalid} | SE={standard_error:.4f} "
        f"| 95% CI=({ci_lower:.4f}, {ci_upper:.4f})"
    )
    for quad, stats in quadrant_stats.items():
        display_acc = stats["accuracy"]
        acc_str = f"{display_acc*100:.1f}%" if display_acc is not None else "n/a"
        print(f"  Quadrant {quad}: {stats['correct']}/{stats['total']} correct ({acc_str})")

    slug = model_name_dict.get(args.model, args.model.replace(".", "_").replace("-", "_"))
    out_dir = args.output_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rule_2_17.json"
    payload = {
        "model": args.model,
        "shots": args.shots,
        "trials": args.trials,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "overall_accuracy": accuracy,
        "standard_error": standard_error,
        "confidence_interval_95": [ci_lower, ci_upper],
        "invalid": invalid,
        "quadrant_stats": quadrant_stats,
        "records": records_out,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved detailed results to {out_path}")

#############################################
if __name__ == "__main__":
    main()