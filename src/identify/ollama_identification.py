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
import keyword_search

tqdm.pandas()

DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "prompts",
    "causal_identification.txt",
)


# define a function for runing ollama
def ollama_run(model, prompt):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]


def ollama_single_identification(model, inputted_data, prompt):

    run_prompt = prompt.format(inputted_data=inputted_data)

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


def ollama_inference(descriptions_df, model, prompt, filter_to_keyword_sentences):

    if filter_to_keyword_sentences:
        descriptions_df = keyword_search.extract_keyword_sentences(
            descriptions_df,
            positive_keywords=keyword_search.POSITIVE_KEYWORDS,
            negative_keywords=keyword_search.NEGATIVE_KEYWORDS,
        )
        descriptions_df["prompt_input_info"] = descriptions_df.apply(
            lambda row: (
                row["scraped_text"]
                if len(row["keyword_sentences"]) == 0
                else "\n".join(row["keyword_sentences"])
            ),
            axis=1,
        )
    else:
        descriptions_df["prompt_input_info"] = descriptions_df["scraped_text"]

    print(f"Running Ollama identification with model: {model}")
    descriptions_df["ollama_response"] = descriptions_df[
        "prompt_input_info"
    ].progress_apply(lambda x: ollama_single_identification(model, x, prompt))

    descriptions_df["predicted_label"] = descriptions_df["ollama_response"].apply(
        ollama_response_to_label
    )

    return descriptions_df
