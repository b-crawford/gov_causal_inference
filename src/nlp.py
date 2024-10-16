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
