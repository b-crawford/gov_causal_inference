import argparse
import pandas as pd
import os
import sys
import mlflow
import metrics

# import local modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import keyword_search


def found_words_to_label(
    positive_keywords, negative_keywords, min_positive_to_negative_keyword_diff
):
    if (
        len(positive_keywords) - len(negative_keywords)
    ) >= min_positive_to_negative_keyword_diff:
        return "possibly"
    else:
        return "unlikely"


def keyword_inference(descriptions_df, min_positive_to_negative_keyword_diff=1):

    descriptions_df = keyword_search.extract_keywords(
        descriptions_df,
        positive_keywords=keyword_search.POSITIVE_KEYWORDS,
        negative_keywords=keyword_search.NEGATIVE_KEYWORDS,
    )

    descriptions_df["keyword_label"] = descriptions_df.apply(
        lambda row: found_words_to_label(
            row["positive_keywords"],
            row["negative_keywords"],
            min_positive_to_negative_keyword_diff,
        ),
        axis=1,
    )

    return descriptions_df


def extract_possibly_probs(result):
    return result["scores"][0]


def log_metrics(prediction_df, run_id, min_positive_to_negative_keyword_diff):

    # Calculate precision and recall for each label
    precision, recall, f1, n_manually_labelled, n_failed_prediction = (
        metrics.get_metrics(
            prediction_df=prediction_df,
            predict_label_column="keyword_label",
            actual_label_column="label",
            labels=metrics.LABELS,
        )
    )

    # set the mlflow server
    utils.set_mlflow_file_location(run_id)

    with mlflow.start_run():

        # log the system
        mlflow.log_param("system", "Keyword search")

        # Log the categories used for zero-shot identification
        mlflow.log_param(
            "min_positive_to_negative_keyword_diff",
            min_positive_to_negative_keyword_diff,
        )

        # # Log the minimum probability to assign a project as causal inference
        mlflow.log_param("positive_keywords", keyword_search.POSITIVE_KEYWORDS)
        mlflow.log_param("negative_keywords", keyword_search.NEGATIVE_KEYWORDS)

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
        "--min_positive_to_negative_keyword_diff",
        type=float,
        help="The minimum number of positive keywords more than negative for a project to be identified as causal.",
        default=1,
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

        output_filename = "keyword_evaluation.csv"
    else:
        run_df = descriptions
        output_filename = "keyword_identification.csv"

    # run processing
    prediction_df = keyword_inference(
        run_df, args.min_positive_to_negative_keyword_diff
    )

    prediction_df.to_csv(os.path.join(run_path, output_filename), index=False)

    if args.evaluation_mode:
        log_metrics(
            prediction_df, args.run_id, args.min_positive_to_negative_keyword_diff
        )
