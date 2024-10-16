import sys
import os

# import local modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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


def keyword_inference(
    descriptions_df,
    min_positive_to_negative_keyword_diff,
    positive_keywords,
    negative_keywords,
):

    descriptions_df = keyword_search.extract_keywords(
        descriptions_df,
        positive_keywords=positive_keywords,
        negative_keywords=negative_keywords,
    )

    descriptions_df["predicted_label"] = descriptions_df.apply(
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
