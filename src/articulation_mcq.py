############################################################################################################
# articulation_mcq.py: Multiple-choice eval
# python src/articulation_mcq.py --rule-number 1 --model gpt-4.1-2025-04-14
# run from the root -- not robust to being run elsewhere yet....
# options to run in single turn or multiturn + with/without CoT
############################################################################################################

import argparse
import json
import os
import random
import re
from pathlib import Path

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

classification_prompt = """
You will be presented with {n_examples} examples of texts. Texts marked True all follow a specific rule. Texts marked False do not follow the rule. Your task is to learn the rule and classify a new example.

You must:
- Learn the rule from the examples provided.
- Apply this rule to the next case.
- Respond with True or False only. Do not include any other words in your answer.

Examples of rule:
{examples}

New text:
{query}

Choose which rule explains the examples. One of the rules is correct and the other is incorrect. You must pick one of the rules.

Option A: {desc_one}
Option B: {desc_two}

Respond with JSON: {{"answer":"A"}} or {{"answer":"B"}}.""".strip()

classification_prompt_cot = """
You will be presented with {n_examples} examples of texts. Texts marked True all follow a specific rule. Texts marked False do not follow the rule. Your task is to learn the rule and classify a new example.

You must:
- Learn the rule from the examples provided.
- Apply this rule to the next case.
- Respond with True or False only. Do not include any other words in your answer.

Examples of rule:
{examples}

New text:
{query}

Choose which rule explains the examples. One of the rules is correct and the other is incorrect. You must pick one of the rules.

Option A: {desc_one}
Option B: {desc_two}

Respond with JSON: {{"answer":"A"}} or {{"answer":"B"}}, but make sure to think step-by-step first and show all of your reasoning.""".strip()

followup_prompt = """Choose which rule explains the examples. One of the rules is correct and the other is incorrect. You must pick one of the rules.

Option A: {desc_one}
Option B: {desc_two}

Respond with JSON: {{"answer":"A"}} or {{"answer":"B"}}.""".strip()

followup_prompt_cot = """Choose which rule explains the examples. One of the rules is correct and the other is incorrect. You must pick one of the rules.

Option A: {desc_one}
Option B: {desc_two}

Respond with JSON: {{"answer":"A"}} or {{"answer":"B"}}, but make sure to think step-by-step first and show all of your reasoning.""".strip()

###########################################################################################################################################
# Extract first line of the docstring (the proper rule in natural language)
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


# extract the real rule/decoy rule from the doc

###########################################################################################################################################
# main code

def parse_choice_response(text: str):
    if not text:
        return None
    snippets = []
    for match in re.finditer(r"\{[^{}]*\"answer\"\s*:\s*\"[AB]\"[^{}]*\}", text, flags=re.IGNORECASE):
        snippets.append(match.group(0))
    if not snippets:
        return None
    for snippet in reversed(snippets):
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        answer = data.get("answer")
        if isinstance(answer, str):
            upper = answer.strip().upper()
            if upper in {"A", "B"}:
                return upper
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot articulation MCQ experiment.")
    parser.add_argument("--rule-number", type=int, required=True, help="Target rule id (e.g. 7 for rule_7).")
    parser.add_argument("--shots", type=int, default=64, help="Few-shot examples provided to the model.")
    parser.add_argument("--trials", type=int, default=200, help="Number of random episodes to evaluate.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="Model identifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with rule_*.jsonl datasets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/articulation_mcq"),
        help="Directory for saving detailed results.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max completion tokens per query.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--cot", type=bool, default=False, help="To use CoT or not.")
    parser.add_argument("--multiturn", type=str, default="True")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print(f"CoT activate: {args.cot}. Multiturn activate: {args.multiturn}")
    rng = random.Random(args.seed)

    data_path = args.data_dir / f"rule_{args.rule_number}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    records = []
    with data_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) <= args.shots:
        raise ValueError("Dataset must contain more samples than the number of shots.")

    from rules import rules as true_rules
    from rules import decoy_rules

    true_rule_fn = getattr(true_rules, f"rule_{args.rule_number}")
    true_rule_text = extract_natural_language_rule(true_rule_fn.__doc__)

    # get the decoy option

    name = f"rule_{args.rule_number}_decoy_1"
    decoy_option = None
    if hasattr(decoy_rules, name):
        decoy_fn = getattr(decoy_rules, name)
        decoy_option = (name, extract_natural_language_rule(decoy_fn.__doc__))
    if not decoy_option:
        raise ValueError(f"No decoy rules found for rule {args.rule_number}.")

    # load the client. Just do it all simulataneously....
    client = make_client(provider="openai", model_name=args.model, wait=0, max_concurrent=1000)

    print("loaded the client")

    if args.multiturn == "False":
    
        prompts = []
        correct_labels = []
        for _ in range(args.trials):
            batch = rng.sample(records, args.shots + 1) # randomly sample.
            few_shots, query = batch[:-1], batch[-1] # get the few_shots and the query
            support_block = "\n".join(f"{'True' if item['label'] else 'False'}: {item['text']}" for item in few_shots) # add the correct answer

            # get the decoy - just a single decoy
            decoy_name, decoy_text = decoy_option

            # get the labels and randomise the assignment. Set as true label
            true_label = rng.choice(["A", "B"])
            if true_label == "A":
                desc_one, desc_two = true_rule_text, decoy_text
            else:
                desc_one, desc_two = decoy_text, true_rule_text
            correct_label = true_label
            correct_labels.append(correct_label)

            if args.cot == False:
                prompt_template = classification_prompt
            else:
                prompt_template = classification_prompt_cot


            full_prompt = prompt_template.format(
                n_examples=args.shots,
                examples=support_block,
                query=query["text"],
                desc_one=desc_one,
                desc_two=desc_two,
            )

            prompts.append(full_prompt)


        ################################################################################################################################

        chats = [[{"role": "user", "content": x}] for x in prompts]

        response_classify = client.chat(
            messages=chats,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

        # extract the top choices
        choices = [parse_choice_response(x) for x in response_classify]
        correct = [x==y for (x,y) in zip(choices, correct_labels)]

        # calculate the stats.
        total = len(choices)
        num_correct = sum(1 for flag in correct if flag)
        invalid = sum(1 for choice in choices if choice is None)
        accuracy = num_correct / total if total else 0.0
        standard_error = (accuracy * (1 - accuracy) / total) ** 0.5 if total else 0.0
        z_score = 1.96
        ci_lower = max(0.0, accuracy - z_score * standard_error)
        ci_upper = min(1.0, accuracy + z_score * standard_error)

        print(
            f"Model {args.model} | rule_{args.rule_number} | shots={args.shots} | trials={args.trials} "
            f"| temperature={args.temperature} | max_tokens={args.max_tokens} | seed={args.seed} "
            f"| accuracy={accuracy*100:.1f}% | invalid={invalid} | SE={standard_error:.4f} "
            f"| 95% CI=({ci_lower:.4f}, {ci_upper:.4f}) | CoT={args.cot} | multiturn={args.multiturn}"
        )

        results = []
        for prompt, response, choice, gold, hit in zip(prompts, response_classify, choices, correct_labels, correct):
            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "prediction": choice,
                    "gold": gold,
                    "correct": hit,
                    "decoy_rule": decoy_option[0],
                }
            )

        slug = model_name_dict.get(args.model, args.model.replace(".", "_").replace("-", "_"))
        out_dir = Path(f"results/articulation_mcq_single/{slug}")
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

    else:

        # build the prompts by loading the results
        model_slug = model_name_dict.get(args.model, args.model.replace(".", "_").replace("-", "_"))
        print(model_slug)
        results_filepath = f"results/learnability/{model_slug}/rule_{args.rule_number}.json"
        with open(results_filepath, 'r') as f:
            results = json.load(f)

        prompts_1 = []
        responses_1 = []
        for entry in results["records"]:
            prompts_1.append(entry.get("prompt", ""))
            responses_1.append(entry.get("response", ""))

        # followups
        prompts_2 = []
        correct_labels = []
        for _ in range(args.trials):

            # get the decoy - just a single decoy
            decoy_name, decoy_text = decoy_option

            # get the labels and randomise the assignment. Set as true label
            true_label = rng.choice(["A", "B"])
            if true_label == "A":
                desc_one, desc_two = true_rule_text, decoy_text
            else:
                desc_one, desc_two = decoy_text, true_rule_text
            correct_label = true_label
            correct_labels.append(correct_label)

            # add prompt depending on cot
            if args.cot == False:
                prompt_template = followup_prompt
            else:
                prompt_template = followup_prompt_cot


            full_prompt = prompt_template.format(
                desc_one=desc_one,
                desc_two=desc_two,
            )

            prompts_2.append(full_prompt)

        # make prompts
        chats = [[{"role": "user", "content": p1},
                  {"role": "assistant", "content": r1},
                  {"role": "user", "content": p2},
                  ] for (p1, r1, p2) in zip(prompts_1, responses_1, prompts_2)]
        
        # classify
        response_classify = client.chat(
            messages=chats,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )

        # extract the top choices
        choices = [parse_choice_response(x) for x in response_classify]
        correct = [x == y for (x, y) in zip(choices, correct_labels)]

        # calculate the stats.
        total = len(correct_labels)
        num_correct = sum(1 for flag in correct if flag)
        invalid = sum(1 for choice in choices if choice is None)
        accuracy = num_correct / total if total else 0.0
        standard_error = (accuracy * (1 - accuracy) / total) ** 0.5 if total else 0.0
        z_score = 1.96
        ci_lower = max(0.0, accuracy - z_score * standard_error)
        ci_upper = min(1.0, accuracy + z_score * standard_error)

        print(
            f"Model {args.model} | rule_{args.rule_number} | shots={args.shots} | trials={args.trials} "
            f"| temperature={args.temperature} | max_tokens={args.max_tokens} | seed={args.seed} "
            f"| accuracy={accuracy*100:.1f}% | invalid={invalid} | SE={standard_error:.4f} "
            f"| 95% CI=({ci_lower:.4f}, {ci_upper:.4f}) | CoT={args.cot} | multiturn={args.multiturn}"
        )

        results_records = []
        for prompt1, response1, prompt2, response, choice, gold, hit in zip(
            prompts_1, responses_1, prompts_2, response_classify, choices, correct_labels, correct
        ):
            results_records.append(
                {
                    "learnability_prompt": prompt1,
                    "learnability_response": response1,
                    "followup_prompt": prompt2,
                    "response": response,
                    "prediction": choice,
                    "gold": gold,
                    "correct": hit,
                    "decoy_rule": decoy_option[0],
                }
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
            "records": results_records,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
