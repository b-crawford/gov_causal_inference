import argparse
import pandas as pd
import os
from tqdm import tqdm
import ollama
import sys
from fuzzywuzzy import fuzz
import mlflow
import metrics

# import local modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils

tqdm.pandas()

DEFAULT_OLLAMA_MODEL = "llama3"

# load identification prompt
identification_prompt_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "prompts",
    "causal_identification.txt",
)

# Using a context manager to open the prompt and read its contents
with open(identification_prompt_path, "r", encoding="utf-8") as file:
    identification_prompt = file.read()


# define a function for runing ollama
def ollama_run(model, prompt):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]


def ollama_single_identification(model, project_description, prompt):

    run_prompt = prompt.format(project_description=project_description)

    return ollama_run(model, run_prompt)


def ollama_response_to_label(model_response):

    match_cutoff = 80

    if fuzz.ratio(model_response.lower(), "possibly") > match_cutoff:
        return "possibly"
    elif fuzz.ratio(model_response.lower(), "definitely_not") > match_cutoff:
        return "unlikely"
    # elif fuzz.ratio(model_response.lower(), "unclear") > match_cutoff:
    #     return "unclear"
    else:
        return pd.NA


def ollama_inference(descriptions_df, model, prompt):

    print(f"Running Ollama identification with model: {model}")
    descriptions_df["ollama_response"] = descriptions_df["scraped_text"].progress_apply(
        lambda x: ollama_single_identification(model, x, prompt)
    )

    descriptions_df["predicted_label"] = descriptions_df["ollama_response"].apply(
        ollama_response_to_label
    )

    return descriptions_df
