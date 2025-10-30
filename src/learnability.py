############################################################################################################
# learnability.py: Few-shot evaluation loop for the rule datasets.
# Each run samples random support/test splits, queries the model, and logs accuracy.
# Usage:
#   python src/learnability.py --rule-number 1 --shots 4 --trials 200 --model gpt-4.1-2025-04-14

# TODO
# - rewrite this code.
############################################################################################################

import argparse
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv

from utils import make_client

load_dotenv()


model_name_dict = {
    "gpt-4.1-2025-04-14": "gpt_4_1",
    "gpt-4.1-mini-2025-04-14": "gpt_4_1_mini",
    "gpt-4.1-nano-2025-04-14": "gpt_4_1_nano",
    "gpt-5-2025-08-07": "gpt_5", # this is probably quite expensive so watch out...
    "gpt-5-mini-2025-08-07": "gpt_5_mini",
    "gpt-5-nano-2025-08-07": "gpt_5_nano",
}

main_prompt = """
You will be presented with {n_examples} examples of texts. Texts marked True all follow a specific rule. Texts marked False do not follow the rule. Your task is to learn the rule and classify a new example.

You must:
- Learn the rule from the examples provided.
- Apply this rule to the next case.
- Respond with True or False only. Do not include any other words in your answer.

Examples of rule:
{}

New text:
{}
""".strip()

# args.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICL learnability sweep over stored rule datasets.")
    parser.add_argument("--rule-number", type=int, required=True, help="Target rule id (e.g. 7 for rule_7).")
    parser.add_argument("--shots", type=int, default=4, help="Few-shot examples provided to the model.")
    parser.add_argument("--trials", type=int, default=200, help="How many random episodes to evaluate.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="Model identifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where rule_*.jsonl datasets live.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/learnability"),
        help="Root folder for saving results.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=4, help="Max completion tokens per query.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
    return parser.parse_args()

# 
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


def format_support(examples):
    return "\n".join(
        f"{'True' if ex['label'] else 'False'}: {ex['text']}"
        for ex in examples
    )


def parse_label(text: str):
    lower = text.strip().lower()
    if lower.startswith("true"):
        return True
    if lower.startswith("false"):
        return False
    return None

########################################################################################################
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data_path = args.data_dir / f"rule_{args.rule_number}.jsonl"
    all_examples = load_examples(data_path)
    if len(all_examples) <= args.shots:
        raise ValueError("Dataset must contain more samples than the number of shots.")

    client = make_client(
        provider="openai",
        model_name=args.model,
        wait=0,
        max_concurrent=1000, # just do everything in parallel... should be fine...
    )

    episodes = []
    messages = []
    for idx in range(args.trials):
        batch = rng.sample(all_examples, args.shots + 1)
        support, query = batch[:-1], batch[-1]
        prompt = main_prompt.format(
            format_support(support),
            query["text"],
            n_examples=args.shots,
        )
        messages.append([{"role": "user", "content": prompt}])
        episodes.append({"support": support, "query": query, "prompt": prompt})

    responses = client.chat(
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    results = []
    correct = 0
    invalid = 0
    for episode, reply in zip(episodes, responses):
        pred = parse_label(reply or "")
        gold = bool(episode["query"]["label"])
        hit = pred == gold
        if pred is None:
            invalid += 1
            hit = False
        if hit:
            correct += 1
        results.append(
            {
                "prompt": episode["prompt"],
                "support": episode["support"],
                "query": episode["query"],
                "response": reply,
                "prediction": pred,
                "gold": gold,
                "correct": hit,
            }
        )

    accuracy = correct / args.trials if args.trials else 0.0
    # Standard error uses the Bernoulli variance estimate.
    standard_error = (accuracy * (1 - accuracy) / args.trials) ** 0.5 if args.trials else 0.0
    z_score = 1.96  # Approximate two-sided 95% normal quantile.
    ci_lower = max(0.0, accuracy - z_score * standard_error)
    ci_upper = min(1.0, accuracy + z_score * standard_error)
    print(
        f"Model {args.model} | rule_{args.rule_number} | shots={args.shots} | trials={args.trials} "
        f"| accuracy={accuracy*100:.1f}% | invalid={invalid} | SE={standard_error:.4f} | 95% CI=({ci_lower:.4f}, {ci_upper:.4f})"
    )

    slug = model_name_dict.get(args.model, args.model.replace(".", "_").replace("-", "_"))
    out_dir = args.output_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rule_{args.rule_number}.json"
    payload = {
        "model": args.model,
        "rule_number": args.rule_number,
        "shots": args.shots,
        "trials": args.trials,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "accuracy": accuracy,
        "standard_error": standard_error,
        "confidence_interval_95": [ci_lower, ci_upper],
        "invalid": invalid,
        "records": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved detailed results to {out_path}")

########################################################################################################
if __name__ == "__main__":
    main()
