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
width = 50


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
            for line in textwrap.wrap(f"\t{key}: {value}", width=width):
                print("\t" + line)

        user_input = input(
            f"\nPlease choose an option by entering the label code: ({list(options.keys())}): "
        )

        if user_input in options:
            print(f"You selected: {options[user_input]}")
            return user_input
        else:
            print(f"Invalid option. Please choose a valid option: {options_string}.")


def main(run_id):

    run_path = utils.get_run_folder_path(args.run_id)

    # get data
    manually_labelled_examples = utils.get_manually_labelled_examples(run_path)
    descriptions = utils.get_descriptions(run_path)

    # merge together
    merged = descriptions.merge(
        manually_labelled_examples, on=["url", "details"], how="left", indicator=True
    )

    unlabelled_indices = merged["_merge"] == "left_only"

    print(f"Number of previously labelled examples: {sum(~unlabelled_indices)}")
    print(f"Number to label: {sum(unlabelled_indices)}")

    # isolate those which need labelling
    merged = merged.loc[unlabelled_indices]

    # randomise the order
    merged = merged.sample(frac=1, replace=False)

    for i, row in merged.iterrows():

        print("-" * 80)
        print("-" * 80)
        print("-" * 80)

        print("The details for the next example read:")

        # Wrap the text and print each line tabbed
        for line in textwrap.wrap(row["details"], width=width):
            print("\t" + line)

        # ask the user to label it
        user_label = user_labelling()

        # save to file
        new_row = pd.DataFrame(
            {
                "url": [row["url"]],
                "details": [row["details"]],
                "user_label": [user_label],
            }
        )

        new_row.to_csv(
            utils.get_manually_labelled_filepath(run_path),
            mode="a",
            header=False,
            index=False,
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
