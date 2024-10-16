import argparse
import pandas as pd
import os
from tqdm import tqdm
from transformers import pipeline
import sys
import mlflow
import metrics

# import local modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils

tqdm.pandas()

DEFAULT_HUGGINGFACE_MODEL = "facebook/bart-large-mnli"
ZERO_SHOT_CATEGORIES = {
    "possibly": "A piece of research potentially involving causal inference. i.e. a piece of research attempting to quantitatively identify the factors contributing to some phenomenon.",
    "unlikely": "Non-causal research.",
    "unclear": "Not enough information to describe the research.",
}
reverse_category_map = {value: key for key, value in ZERO_SHOT_CATEGORIES.items()}


def huggingface_inference(descriptions_df, model, causal_cutoff=None):

    print(f"Running HuggingFace identification with model: {model}")
    classifier = pipeline("zero-shot-classification", model=model, device="mps")

    descriptions_df["huggingface_output"] = descriptions_df[
        "scraped_text"
    ].progress_apply(lambda x: classifier(x, list(ZERO_SHOT_CATEGORIES.values())))

    if causal_cutoff is not None:

        descriptions_df["huggingface_prob_causal"] = descriptions_df[
            "huggingface_output"
        ].apply(extract_possibly_probs)

        descriptions_df["huggingface_label"] = descriptions_df[
            "huggingface_prob_causal"
        ].apply(lambda x: "possibly" if x > causal_cutoff else "unlikely")

    else:

        descriptions_df["huggingface_label"] = descriptions_df[
            "huggingface_output"
        ].apply(result_to_max_index)

    return descriptions_df


def extract_possibly_probs(result):
    return result["scores"][0]


def result_to_max_index(result):
    # Find the label with the highest probability (score)
    max_score_index = result["scores"].index(max(result["scores"]))
    largest_label = result["labels"][max_score_index]
    return reverse_category_map[largest_label]


def log_metrics(prediction_df, run_id, huggingface_model, causal_cutoff):

    # Calculate precision and recall for each label
    precision, recall, f1, n_manually_labelled, n_failed_prediction = (
        metrics.get_metrics(
            prediction_df=prediction_df,
            predict_label_column="huggingface_label",
            actual_label_column="label",
            labels=metrics.LABELS,
        )
    )

    # set the mlflow server
    utils.set_mlflow_file_location(run_id)

    with mlflow.start_run():

        # log the system
        mlflow.log_param("system", "HuggingFace")

        # Log the model name
        mlflow.log_param("model_name", huggingface_model)

        # Log the categories used for zero-shot identification
        mlflow.log_param("zero_shot_categories", ZERO_SHOT_CATEGORIES)

        # # Log the minimum probability to assign a project as causal inference
        mlflow.log_param("mode", "cutoff" if causal_cutoff is not None else "max_prob")
        mlflow.log_param("causal_cutoff", causal_cutoff)

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
        "-m",
        "--model",
        type=str,
        help="The HuggingFace model we should use.",
        default=DEFAULT_HUGGINGFACE_MODEL,
    )
    parser.add_argument(
        "-c",
        "--causal_cutoff",
        type=float,
        help="The minimum probability to assign a projec as causal inference.",
        default=None,
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

        output_filename = "huggingface_evaluation.csv"
    else:
        run_df = descriptions
        output_filename = "huggingface_identification.csv"

    # run processing
    prediction_df = huggingface_inference(run_df, args.model, args.causal_cutoff)

    prediction_df.to_csv(os.path.join(run_path, output_filename), index=False)

    if args.evaluation_mode:
        log_metrics(prediction_df, args.run_id, args.model, args.causal_cutoff)
