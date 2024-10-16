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


def ollama_single_identification(model, project_description):

    prompt = identification_prompt.format(project_description=project_description)

    return ollama_run(model, prompt)


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


def ollama_inference(descriptions_df, model):

    print(f"Running Ollama identification with model: {model}")
    descriptions_df["ollama_response"] = descriptions_df["scraped_text"].progress_apply(
        lambda x: ollama_single_identification(model, x)
    )

    descriptions_df["ollama_label"] = descriptions_df["ollama_response"].apply(
        ollama_response_to_label
    )

    return descriptions_df


def log_metrics(prediction_df, run_id, ollama_model):

    # Calculate precision and recall for each label
    precision, recall, f1, n_manually_labelled, n_failed_prediction = (
        metrics.get_metrics(
            prediction_df=prediction_df,
            predict_label_column="ollama_label",
            actual_label_column="label",
            labels=metrics.LABELS,
        )
    )

    # set the mlflow server
    utils.set_mlflow_file_location(run_id)

    with mlflow.start_run():

        # log the system
        mlflow.log_param("system", "Ollama")

        # Log the model name
        mlflow.log_param("model_name", ollama_model)

        # Log the prompt used for the classification
        mlflow.log_param("prompt", identification_prompt)

        # log the number of manually labelled in the dataset
        mlflow.log_param("n_manually_labelled", n_manually_labelled)

        # log the number for which prediction failed
        mlflow.log_param("n_failed_prediction", n_failed_prediction)

        # Log precision and recall for each label
        for i, label in enumerate(metrics.LABELS):
            mlflow.log_metric(f"precision_label_{label}", precision[i])
            mlflow.log_metric(f"recall_label_{label}", recall[i])
            mlflow.log_metric(f"f1_label_{label}", f1[i])


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
        "--evaluation_mode",
        action="store_true",
        help="Only predict for manually labelled responses.",
    )
    args = parser.parse_args()

    # load data
    run_path = utils.get_run_folder_path(args.run_id)
    descriptions = pd.read_csv(os.path.join(run_path, "basic_descriptions.csv"))

    if args.evaluation_mode:
        # get the manually labelled
        manually_labelled = utils.get_manually_labelled_examples(run_path)

        run_df = manually_labelled[["project_hash", "label"]].merge(
            descriptions,
            how="left",
            on="project_hash",
        )

        output_filename = "ollama_evaluation.csv"
    else:
        run_df = descriptions
        output_filename = "ollama_identification.csv"

    # run ollama processing
    prediction_df = ollama_inference(run_df, args.ollama_model)

    prediction_df.to_csv(os.path.join(run_path, output_filename), index=False)

    if args.evaluation_mode:
        log_metrics(prediction_df, args.run_id, args.ollama_model)
