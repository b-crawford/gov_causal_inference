from sklearn.metrics import precision_score, recall_score, f1_score
import sys
import os

# import local modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils

LABELS = ["possibly", "unclear", "unlikely"]


def get_metrics(prediction_df, predict_label_column, actual_label_column, labels):

    initial_length = len(prediction_df)

    prediction_df = prediction_df[[actual_label_column, predict_label_column]].dropna()

    y_test = prediction_df[actual_label_column]
    y_pred = prediction_df[predict_label_column]

    precision = precision_score(
        y_test, y_pred, average=None, labels=labels, zero_division=0
    )
    recall = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)

    f1 = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0)

    return precision, recall, f1, initial_length, initial_length - len(prediction_df)
