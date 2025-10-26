############################################################################################################
# learnability.py: Generate responses for ICL
# The modeles are given n few-shot examples and (with no CoT) told to classify the next case.
# results are saved to data/model/rule_{n}.json and 
# does it for m examples. i.e. I might want to consider 1000 repeats.
############################################################################################################
 
import argparse
import os
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from utils import make_client

############################################################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14"),
        help="gpt-4.1-2025-04-14, gpt-4.1-mini-2025-04-14, gpt-4.1-nano-2025-04-14",
    )
    return parser.parse_args()

model_name_dict = {
    'gpt-4.1-2025-04-14':'gpt_4_1',
    'gpt-4.1-mini-2025-04-14':'gpt_4_1_mini',
    'gpt-4.1-nano-2025-04-14':'gpt_4_1_nano',
    }

# write the main prompt
main_prompt = """
You will be presented with {n_examples} examples of texts. Texts marked True all follow a specific rule. Texts marked False do not follow the rule. Your task is to learn the rule and classify a new example.

You must:
- Learn the rule from the examples provided.
- Apply this rule to the next case.
- Respond with True or False only. Do not include any other words in your answer 

Examples of rule:
{}

New text:
{}
"""

############################################################################################################
# helpers
############################################################################################################





############################################################################################################

# randomly select n few-shot examples (without replacement) and 1 test example 

# create the prompt with xthe n few-shot examples and the test example

# wrap this up into m prompts...

# put all m prompts through the OpenAI API. Use the make_client function from utils to do this and print everything nicely.... 

# Get the answers + compare to ground truth


# Print the accuracy... If scoring is turned on, save json of everything to data/model/rule_{n}.json
