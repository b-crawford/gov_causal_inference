import argparse
import pandas as pd
import os
import sys
import mlflow
import metrics

# import local modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils


def baseline_inference(descriptions_df):

    descriptions_df["baseline_label"] = "possibly"

    return descriptions_df


def log_metrics(prediction_df, run_id):

    # Calculate precision and recall for each label
    precision, recall, f1, n_manually_labelled, n_failed_prediction = (
        metrics.get_metrics(
            prediction_df=prediction_df,
            predict_label_column="baseline_label",
            actual_label_column="label",
            labels=metrics.LABELS,
        )
    )

    # set the mlflow server
    utils.set_mlflow_file_location(run_id)

    with mlflow.start_run():

        # log the system
        mlflow.log_param("system", "Baseline search")

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

        output_filename = "baseline_evaluation.csv"
    else:
        run_df = descriptions
        output_filename = "baseline_identification.csv"

    # run processing
    prediction_df = baseline_inference(run_df)

    prediction_df.to_csv(os.path.join(run_path, output_filename), index=False)

    if args.evaluation_mode:
        log_metrics(prediction_df, args.run_id)
