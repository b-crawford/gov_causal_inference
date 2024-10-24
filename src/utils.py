import os
import re
import pandas as pd
import mlflow

runs_data_folder = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)


def set_mlflow_file_location(run_id):
    mlflow.set_tracking_uri(f"{runs_data_folder}/{run_id}/mlflow")


def get_max_integer_folder(parent_dir):
    # List all subfolders in the parent directory
    subfolders = [
        f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))
    ]

    # Use a regular expression to find subfolders with integer names
    integer_folders = [int(f) for f in subfolders if re.match(r"^\d+$", f)]

    # Find the maximum integer folder
    if integer_folders:
        return max(integer_folders)
    else:
        return 1


def get_run_id(declared_run_id, runs_data_folder=runs_data_folder):
    # if we have not declared a run the give this run a new id
    if declared_run_id is not None:
        return declared_run_id
    else:
        max_run_so_far = get_max_integer_folder(runs_data_folder)
        return max_run_so_far + 1


def get_run_folder_path(declared_run_id=None, runs_data_folder=runs_data_folder):

    run_id = get_run_id(declared_run_id)

    run_subfolder = f"{runs_data_folder}/{run_id}"

    # create subfolder in data directory for this run ID
    if not os.path.exists(run_subfolder):
        os.makedirs(run_subfolder)

    return run_subfolder


def get_manually_labelled_filepath(run_path):
    return os.path.join(run_path, "manually_labelled_examples.csv")


def get_manually_labelled_examples(run_path):

    filepath = get_manually_labelled_filepath(run_path)

    if not os.path.exists(filepath):
        return None

    return pd.read_csv(filepath)


def get_training_filepath(run_path):
    return os.path.join(run_path, "training_examples.csv")


def get_training_examples(run_path):

    filepath = get_training_filepath(run_path)

    if not os.path.exists(filepath):
        return pd.DataFrame({"project_hash": []})

    return pd.read_csv(filepath)


def get_eval_filepath(run_path):
    return os.path.join(run_path, "eval_examples.csv")


def get_eval_examples(run_path):

    filepath = get_eval_filepath(run_path)

    if not os.path.exists(filepath):
        return pd.DataFrame({"project_hash": []})

    return pd.read_csv(filepath)


def get_descriptions(run_path):
    filepath = os.path.join(run_path, "basic_descriptions.csv")
    return pd.read_csv(filepath)


def get_finetune_model_folder(run_path):
    return os.path.join(run_path, "finetuned_model")


def get_nlp(run_path):
    filepath = os.path.join(run_path, "nlp.csv")
    return pd.read_csv(filepath)


# Define a new method for DataFrame
def to_csv_or_append(self, file_name, index=False):
    """Writes or appends the DataFrame to a CSV file depending on file existence.

    Parameters:
    - file_name (str): The name of the CSV file.
    - index (bool): Whether to write row indices. Default is False.
    """
    if not os.path.exists(file_name):
        # If file does not exist, create it and write the header
        self.to_csv(file_name, mode="w", header=True, index=index)
        print(f"File created and data written to {file_name}")
    else:
        # Open the file and check if the last character is a newline
        with open(file_name, "rb+") as f:
            f.seek(-1, os.SEEK_END)  # Go to the last byte in the file
            last_char = f.read(1)
            if last_char != b"\n":
                f.write(b"\n")  # Append a newline if the last char is not a newline
        self.to_csv(file_name, mode="a", header=False, index=index)
        print(f"Data appended to {file_name}")


# Bind the method to the DataFrame class
pd.DataFrame.to_csv_or_append = to_csv_or_append
