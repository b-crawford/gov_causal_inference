import requests
from bs4 import BeautifulSoup
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import argparse
import utils
import os
import textwrap


# Set the width for text wrapping
def wrap_tab_print(string, width=50):
    if pd.isna(string):
        print("\tNone found.")
    else:
        for line in textwrap.wrap(string, width=width):
            print("\t" + line)


# Function to ask for command line input
def user_labelling():

    options = {
        "1": "The description suggests the research attempts to answer a causal question.",
        "0": "There is not enough information present to decide whether the research attempts to answer a causal question.",
        "-1": "The description suggests that no causal inference was done or required.",
    }

    while True:

        print("\nPlease choose between one of the following options")
        for key, value in options.items():
            wrap_tab_print(f"{key}: {value}")

        user_input = input(
            f"\nPlease choose an option by entering the label code: ({list(options.keys())}): "
        )

        if user_input in options:
            print(f"You selected: {options[user_input]}")
            return user_input
        else:
            print(
                f"Invalid option. Please choose a valid option: {list(options.keys())}."
            )


def main(run_id):

    run_path = utils.get_run_folder_path(args.run_id)

    # get data
    manually_labelled_examples = utils.get_manually_labelled_examples(run_path)
    descriptions = utils.get_descriptions(run_path)

    # merge together
    if manually_labelled_examples is not None:
        merged = descriptions.merge(
            manually_labelled_examples[["url"]], on="url", how="left", indicator=True
        )
        unlabelled_indices = merged["_merge"] == "left_only"

        print(f"Number of previously labelled examples: {sum(~unlabelled_indices)}")
        print(f"Number to label: {sum(unlabelled_indices)}")

        # isolate those which need labelling
        merged = merged.loc[unlabelled_indices].drop(labels=["_merge"], axis=1)
    else:
        merged = descriptions

    # randomise the order
    merged = merged.sample(frac=1, replace=False)

    for row in merged.to_dict("records"):

        print("-" * 80)
        print("-" * 80)
        print("-" * 80)

        print("The url for the next example is:")
        wrap_tab_print(row["url"])

        print("The publishing organisation is:")
        wrap_tab_print(row["publishing_institution"])

        print("The title is:")
        wrap_tab_print(row["title"])

        print("The context is:")
        wrap_tab_print(row["context"])

        print("The description reads:")
        wrap_tab_print(row["description"])

        print("The details section reads:")
        wrap_tab_print(row["details"])

        # ask the user to label it
        row["label"] = user_labelling()

        # save the time
        row["labelled_at"] = datetime.now()

        # save to file
        pd.DataFrame(row, index=[0]).to_csv_or_append(
            utils.get_manually_labelled_filepath(run_path)
        )


if __name__ == "__main__":

    # get the date from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_id",
        type=int,
        help="The run ID to gather examples from.",
        default=None,
    )
    parser.add_argument("--rescrape", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # run main function
    main(args.run_id)