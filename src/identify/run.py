import argparse
import pandas as pd
import os
import sys
import mlflow
from functools import partial

# import local modules
import metrics
import huggingface_identification
import keyword_identification
import ollama_identification
import baseline_identification

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import keyword_search


def log_metrics(prediction_df, run_id, labels, system, **kwargs):

    # Calculate precision and recall for each label
    precision, recall, f1, n_manually_labelled, n_failed_prediction = (
        metrics.get_metrics(
            prediction_df=prediction_df,
            predict_label_column="predicted_label",
            actual_label_column="label",
            labels=labels,
        )
    )

    # set the mlflow server
    utils.set_mlflow_file_location(run_id)

    with mlflow.start_run():

        # log the system
        mlflow.log_param("system", system)

        # log the number of manually labelled in the dataset
        mlflow.log_param("n_manually_labelled", n_manually_labelled)

        # log the number for which prediction failed
        mlflow.log_param("n_failed_prediction", n_failed_prediction)

        # Log precision and recall for each label
        for i, label in enumerate(metrics.LABELS):
            mlflow.log_metric(f"precision_label_{label}", precision[i])
            mlflow.log_metric(f"recall_label_{label}", recall[i])
            mlflow.log_metric(f"f1_label_{label}", f1[i])

        # Loop through the kwargs and log each key-value pair to MLflow
        for key, value in kwargs.items():
            mlflow.log_param(key, value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_id",
        type=int,
        help="The run ID for the data over which we should process.",
    )
    parser.add_argument(
        "system",
        type=str,
        help="Which system to run the predictions for. ",
    )
    parser.add_argument(
        "--evaluation_mode",
        action="store_true",
        help="Only predict for manually labelled responses.",
    )
    parser.add_argument(
        "-hf",
        "--huggingface_model",
        type=str,
        help="HuggingFace parameter: the model we should use.",
        default=huggingface_identification.DEFAULT_HUGGINGFACE_MODEL,
    )
    parser.add_argument(
        "-c",
        "--causal_cutoff",
        type=float,
        help="HuggingFace parameter: the minimum probability to assign a project as causal inference.",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--min_keyword_diff",
        type=float,
        help="Keyword parameter: the minimum number of positive keywords more than negative for a project to be identified as causal.",
        default=1,
    )
    parser.add_argument(
        "-o",
        "--ollama_model",
        type=str,
        help="The ollama model we should use.",
        default=ollama_identification.DEFAULT_OLLAMA_MODEL,
    )
    args = parser.parse_args()

    # load data
    run_path = utils.get_run_folder_path(args.run_id)
    descriptions = pd.read_csv(os.path.join(run_path, "basic_descriptions.csv"))

    # load the prediction function
    if args.system == "huggingface":
        system_kwargs = {
            "model": args.huggingface_model,
            "causal_cutoff": args.causal_cutoff,
        }
        predict_function = partial(
            huggingface_identification.huggingface_inference, **system_kwargs
        )
    elif args.system == "baseline":
        system_kwargs = {}
        predict_function = baseline_identification.baseline_inference
    elif args.system == "keyword":
        system_kwargs = {
            "min_positive_to_negative_keyword_diff": args.min_keyword_diff,
            "positive_keywords": keyword_search.POSITIVE_KEYWORDS,
            "negative_keywords": keyword_search.NEGATIVE_KEYWORDS,
        }
        predict_function = partial(
            keyword_identification.keyword_inference, **system_kwargs
        )
    elif args.system == "ollama":
        system_kwargs = {
            "model": args.ollama_model,
            "prompt": ollama_identification.identification_prompt,
        }
        predict_function = partial(
            ollama_identification.ollama_inference, **system_kwargs
        )
    else:
        raise ValueError(f"Invalid system declared.")

    if args.evaluation_mode:
        # get the manually labelled
        manually_labelled = utils.get_manually_labelled_examples(run_path)

        run_df = manually_labelled[["project_hash", "label"]].merge(
            descriptions,
            how="left",
            on="project_hash",
        )

        output_filename = f"{args.system}_evaluation.csv"
    else:
        run_df = descriptions
        output_filename = f"{args.system}_identification.csv"

    # run processing
    prediction_df = predict_function(run_df)

    prediction_df.to_csv(os.path.join(run_path, output_filename), index=False)

    if args.evaluation_mode:
        log_metrics(
            prediction_df=prediction_df,
            run_id=args.run_id,
            labels=metrics.LABELS,
            system=args.system,
            **system_kwargs,
        )
