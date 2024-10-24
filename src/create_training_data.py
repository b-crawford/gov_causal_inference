import pandas as pd
import argparse
import utils


def split_some_data_for_training(run_path, proportion_train):

    training = utils.get_training_examples(run_path)
    manually_labelled = utils.get_manually_labelled_examples(run_path)

    if manually_labelled is None:
        return None

    # work out the target length to hit the proportion
    target_len = round(len(manually_labelled) * proportion_train)

    # sample the remaining
    to_sample = target_len - len(training)
    if to_sample > 0:
        to_move_ids = manually_labelled.sample(n=to_sample)["project_hash"]

        # save the remaining evaluation data to csv
        resulting_eval = manually_labelled.loc[
            ~manually_labelled.project_hash.isin(to_move_ids)
        ]
        print(f"Evaluation examples: {len(resulting_eval)}")
        resulting_eval.to_csv(utils.get_eval_filepath(run_path), index=False)

        # save the new manually labelled data to csv
        to_move = manually_labelled.loc[
            manually_labelled.project_hash.isin(to_move_ids)
        ]
        resulting_train = pd.concat([training, to_move], axis=0)
        print(f"Training examples: {len(resulting_train)}")
        resulting_train.to_csv(utils.get_training_filepath(run_path), index=False)


if __name__ == "__main__":

    # get the date from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_id",
        type=int,
        help="The run ID to gather examples from.",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--proportion_train",
        type=float,
        help="The proportion of manually labelled data to use for training.",
        default=0.3,
    )
    args = parser.parse_args()

    run_path = utils.get_run_folder_path(args.run_id)

    # run main function
    split_some_data_for_training(run_path, args.proportion_train)
