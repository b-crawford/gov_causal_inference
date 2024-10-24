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
import finetune_identification

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import keyword_search


def log_metrics(prediction_df, run_id, labels, system, **kwargs):

    # Calculate precision and recall for each label
    precision, recall, f1, n_eval, n_failed_prediction = metrics.get_metrics(
        prediction_df=prediction_df,
        predict_label_column="predicted_label",
        actual_label_column="label",
        labels=labels,
    )

    # set the mlflow server
    utils.set_mlflow_file_location(run_id)

    with mlflow.start_run():

        # log the system
        mlflow.log_param("system", system)

        # log the number of manually labelled in the eval dataset
        mlflow.log_param("n_evaluation_set", n_eval)

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
    parser.add_argument(
        "-p",
        "--prompt_path",
        type=str,
        help="The prompt we should use.",
        default=ollama_identification.DEFAULT_PROMPT_PATH,
    )
    parser.add_argument(
        "-f",
        "--filter_to_keyword_sentences",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "-ft",
        "--finetune_model_path",
        type=str,
        help="The path to the finetuned model.",
        default=finetune_identification.BASE_MODEL_PATH,
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
        # Using a context manager to open the prompt and read its contents
        with open(args.prompt_path, "r", encoding="utf-8") as file:
            prompt = file.read()

        system_kwargs = {
            "model": args.ollama_model,
            "prompt": prompt,
            "filter_to_keyword_sentences": args.filter_to_keyword_sentences,
        }
        predict_function = partial(
            ollama_identification.ollama_inference, **system_kwargs
        )
    elif args.system == "finetune":

        if args.finetune_model_path == finetune_identification.BASE_MODEL_PATH:
            model_path = finetune_identification.train(
                run_path, args.finetune_model_path
            )
        else:
            model_path = args.finetune_model_path

        system_kwargs = {
            "model_path": model_path,
        }

        predict_function = partial(finetune_identification.predict, **system_kwargs)
    else:
        raise ValueError(f"Invalid system declared.")

    if args.evaluation_mode:
        # get the eval dataset
        eval_dataset = utils.get_eval_examples(run_path)

        run_df = eval_dataset[["project_hash", "label"]].merge(
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
