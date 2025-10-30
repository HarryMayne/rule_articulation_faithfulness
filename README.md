# Rule Articulation Code
Code for takehome for Owain Evans application. Harry Mayne

- `src`
    - `rules`: Code to generate new rules. Use the `make_rule.py` script to generate an additional rule. Rules stored in `rules.py`
    - `learnability`: Test whether the models can learn the rules in context with few-shot sampling.
- `data`: 
    - `make_data.py`: Takes a rule and uses GPT-5 to generate n example of that rule as a jsonl
    - `validate_data.py`: Neat script that uses the python function to check for rule adherence in the data generation
- `results`:
    - `learnability`: Can the LLMs learn the rule and apply to a new example (results by model + rule)
    