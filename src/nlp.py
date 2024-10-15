import spacy
from spacy.cli import download
import argparse
import pandas as pd
import os
from tqdm import tqdm
import ollama
import sys
from transformers import pipeline
from fuzzywuzzy import fuzz


# import local modules
sys.path.append(os.path.dirname(__file__))
import utils

tqdm.pandas()

# set up parameters
POSITIVE_KEYWORDS = [
    "impact",
    "drivers",
    "effect",
    "affect",
    "causal",
    "influence",
    "relation",
    "relationship",
    "contribute",
    "factor",
]
NEGATIVE_KEYWORDS = ["political influence"]

DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_HUGGINGFACE_MODEL = "facebook/bart-large-mnli"

# load identification prompt
identification_prompt_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompts", "causal_identification.txt"
)

# Using a context manager to open the file and read its contents
with open(identification_prompt_path, "r", encoding="utf-8") as file:
    identification_prompt = file.read()


# Function to ensure spaCy model is downloaded
def ensure_model_downloaded(model_name):
    try:
        # Try to load the spaCy model
        nlp = spacy.load(model_name)
        print(f"Loaded model: {model_name}")
        return nlp
    except OSError:
        # If the model is not found, download it
        print(f"Model {model_name} not found. Downloading...")
        download(model_name)
        # Load the model after downloading
        nlp = spacy.load(model_name)
        print(f"Downloaded and loaded model: {model_name}")
        return nlp


def find_keywords(nlp, description, keywords):
    """Search for causal keywords in the description."""
    doc = nlp(description)
    found_keywords = [token.text for token in doc if token.text.lower() in keywords]
    return found_keywords


def extract_keywords(descriptions_df, positive_keywords, negative_keywords):

    nlp = ensure_model_downloaded("en_core_web_sm")

    print("Running positive keyword search.")
    descriptions_df["positive_keywords"] = descriptions_df[
        "scraped_text"
    ].progress_apply(lambda x: find_keywords(nlp, x, positive_keywords))

    print("Running negative keyword search.")
    descriptions_df["negative_keywords"] = descriptions_df[
        "scraped_text"
    ].progress_apply(lambda x: find_keywords(nlp, x, negative_keywords))

    return descriptions_df


def ollama_run(model, prompt):
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"]


def ollama_single_identification(model, project_description):

    prompt = identification_prompt.format(project_description=project_description)

    return ollama_run(model, prompt)


def ollama_response_to_label(model_response):

    match_cutoff = 90

    # if fuzz.ratio(model_response.lower(), "true")>match_cutoff:
    #     return 1
    # elif fuzz.ratio(model_response.lower(), "false")>match_cutoff:
    #     return -1
    # elif fuzz.ratio(model_response.lower(), "insufficient information")>match_cutoff:
    #     return 0
    # else:
    #     return pd.NA

    if (
        fuzz.ratio(model_response.lower(), "potentially_includes_causal_inference")
        > match_cutoff
    ):
        return 1
    elif (
        fuzz.ratio(model_response.lower(), "does_not_include_causal_inference")
        > match_cutoff
    ):
        return -1
    else:
        return pd.NA


def ollama_inference(descriptions_df, model):

    print(f"Running Ollama identification with model: {model}")
    descriptions_df["ollama_response"] = descriptions_df["scraped_text"].progress_apply(
        lambda x: ollama_single_identification(model, x)
    )

    descriptions_df["ollama_label"] = descriptions_df["ollama_response"].apply(
        ollama_response_to_label
    )

    return descriptions_df


def huggingface_single_identification(classifier, project_description):

    candidate_labels = [
        "A piece of research involving causal inference. i.e. a piece of research attempting to identify the factors contributing to some phenomenon.",
        "Non-causal research.",
        "Not enough information to describe the research.",
    ]

    return classifier(project_description, candidate_labels)["scores"][0]


def huggingface_inference(descriptions_df, model):

    print(f"Running HuggingFace identification with model: {model}")
    classifier = pipeline("zero-shot-classification", model=model, device="mps")

    descriptions_df["huggingface_prob_causal"] = descriptions_df[
        "scraped_text"
    ].progress_apply(lambda x: huggingface_single_identification(classifier, x))

    return descriptions_df


def prediction(descriptions_df):

    descriptions_df["predicted_label"] = descriptions_df["ollama_label"]

    return descriptions_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_id",
        type=int,
        help="The run ID for the data over which we should process.",
    )
    parser.add_argument(
        "-o",
        "--ollama_model",
        type=str,
        help="The ollama model we should use.",
        default=DEFAULT_OLLAMA_MODEL,
    )
    parser.add_argument(
        "-hf",
        "--huggingface_model",
        type=str,
        help="The HuggingFace model we should use.",
        default=DEFAULT_HUGGINGFACE_MODEL,
    )
    args = parser.parse_args()

    # load data
    folder_path = utils.get_run_folder_path(args.run_id)
    descriptions = pd.read_csv(os.path.join(folder_path, "basic_descriptions.csv"))

    # run keyword search
    descriptions = extract_keywords(descriptions, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS)

    # run ollama processing
    descriptions = ollama_inference(descriptions, args.ollama_model)

    # run huggingface processing
    # descriptions = huggingface_inference(descriptions, args.huggingface_model)

    # final prediction
    descriptions = prediction(descriptions)

    # save data
    descriptions.to_csv(os.path.join(folder_path, "nlp.csv"), index=False)
